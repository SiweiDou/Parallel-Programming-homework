#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <ctime>
#include "omp.h"
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{ 
    //打印运行时间戳
    time_t now = time(0);
    char* dt = ctime(&now);
    cout << "Test started at: " << dt;
    //下面代码用于测试MD5哈希的正确性
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };
    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        for (int i1 = 0; i1 < 4; i1 += 1) {
            ss << std::setw(8) << std::setfill('0') << hex << state[i1];
        }
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl; //请不要修改这一行

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    // 此处采用多优先队列进行并行化加速
    auto start_guess = system_clock::now();
    
    int num_threads = omp_get_max_threads();
    vector<PriorityQueue> queues(num_threads);

    // 多优先队列使用相同的模型进行预测,注意在model中将=重载为深拷贝
    for (int i = 0; i < num_threads; i++){
        queues[i].m = q.m;
    }

    // 根据线程数，为每个线程分配一份独立的 ordered_pts 副本
    vector<vector<PT>> partitioned_ordered_pts(num_threads);
    for (int i = 0; i < q.m.ordered_pts.size(); ++i) {
        partitioned_ordered_pts[i % num_threads].push_back(q.m.ordered_pts[i]);
    }

    // 分别存储并行生成的猜测口令
    vector<vector<string>> all_local_guesses(num_threads);

    // 每个线程在自己的队列里，先设置独立的 ordered_pts 副本，再 init()
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PriorityQueue &my_q = queues[tid];
        my_q.m.ordered_pts = partitioned_ordered_pts[tid];
        my_q.init();

        while (!my_q.priority.empty()){
            my_q.PopNext();
            // 此处与原逻辑有区别，我们仅将其存入缓存，等最后统一哈希
            all_local_guesses[tid].insert(all_local_guesses[tid].end(),my_q.guesses.begin(),my_q.guesses.end());
            my_q.guesses.clear();
        }
    }

    // 此处将所有猜测合并，之后统一hash
    vector<string> all_guesses;
    for (int i = 0; i < num_threads; i++){
        all_guesses.insert(all_guesses.end(), all_local_guesses[i].begin(), all_local_guesses[i].end());
    }
    auto end_guess = system_clock::now();
    auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
    time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

    auto start_hash = system_clock::now();
    bit32 state_batch[4][4];
    int count = 0;
    string input_arry[4];
    for(string pw : all_guesses)
    {
        input_arry[count++] = pw;
        if (count == 4){
            MD5HashBatch(input_arry, state_batch);
            count = 0;
        }
    }
    // 处理剩余不足4个的口令（用标量版本完成）
    if (count > 0)
    {
        for (int i = 0; i < count; ++i)
        {
            bit32 state[4];
            MD5Hash(input_arry[i], state); // 调用原始的标量 MD5Hash
        }
    }

    // 在这里对哈希所需的总时长进行计算
    auto end_hash = system_clock::now();
    auto duration = duration_cast<microseconds>(end_hash - start_hash);
    time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    cout << "Guess time:" << time_guess<< "seconds"<< endl;
    cout << "Hash time:" << time_hash << "seconds"<<endl;
    cout << "Train time:" << time_train <<"seconds"<<endl;

    // int curr_num = 0;
    // auto start = system_clock::now();
    // // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    // int history = 0;
    // // std::ofstream a("./files/results.txt");
    // while (!q.priority.empty())
    // {
    //     q.PopNext();
    //     q.total_guesses = q.guesses.size();
    //     if (q.total_guesses - curr_num >= 100000)
    //     {
    //         // cout << "Guesses generated: " <<history + q.total_guesses << endl;
    //         curr_num = q.total_guesses;

    //         // 在此处更改实验生成的猜测上限
    //         int generate_n=10000000;
    //         if (history + q.total_guesses > 10000000)
    //         {
    //             auto end = system_clock::now();
    //             auto duration = duration_cast<microseconds>(end - start);
    //             time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //             cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;//请不要修改这一行
    //             cout << "Hash time:" << time_hash << "seconds"<<endl;//请不要修改这一行
    //             cout << "Train time:" << time_train <<"seconds"<<endl;//请不要修改这一行
    //             break;
    //         }
    //     }
    //     // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
    //     // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
    //     // if (q.guesses.size() >= 8) 发现性能没有提升，可能是条件过于苛刻，导致没有进入分支，我们修改进入分支的条件，只要基类够8个就执行
    //     if (curr_num > 500000) 
    //     {
    //         auto start_hash = system_clock::now();
    //         bit32 state_batch[4][4];
    //         int count = 0;
    //         string input_arry[4];
    //         for(string pw : q.guesses)
    //         {
    //             input_arry[count++] = pw;
    //             if (count == 4){
    //                 MD5HashBatch(input_arry, state_batch);
    //                 count = 0;
    //             }
    //         }
    //         // 处理剩余不足4个的口令（用标量版本完成）
    //         if (count > 0)
    //         {
    //             for (int i = 0; i < count; ++i)
    //             {
    //                 bit32 state[4];
    //                 MD5Hash(input_arry[i], state); // 调用原始的标量 MD5Hash
    //             }
    //         }

    //         // 在这里对哈希所需的总时长进行计算
    //         auto end_hash = system_clock::now();
    //         auto duration = duration_cast<microseconds>(end_hash - start_hash);
    //         time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

    //         // 记录已经生成的口令总数
    //         history += curr_num;
    //         curr_num = 0;
    //         q.guesses.clear();
    //     }
    // }
}
