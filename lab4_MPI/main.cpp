#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <ctime>

// #include "omp.h"
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2 -fopenmp

int main(int argc, char* argv[])
{ 
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        auto now = system_clock::now();
        // 转换为 time_t 以便格式化输出
        std::time_t now_time = system_clock::to_time_t(now);
        cout << "Test started at: " << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << endl;
        
    }
    //打印运行时间戳
    if (rank == 0) {
       
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
        cout << "MD5Hash test passed!" << endl; // 请勿修改此行
    }

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;

    if (rank == 0) {
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

        q.init();
        cout << "here" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);  // 等待 rank 0

    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");

    // q.start_thread_pool();

    // while (true)
    // {
    //     bool empty;
    //     if (rank == 0) {
    //         empty = q.priority.empty();
    //         if (!empty) {
    //             // 取出队首 PT 的信息（但不执行 Generate！）
    //             current_pt = q.priority.front();
    //         }
    //     }
    //     MPI_Bcast(&empty, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    //     if (empty) break;

    //     // ===== Step 2: 广播 PT 信息 =====
    //     // 把 current_pt 的关键字段（max_indices, curr_indices, content types）广播出去
        
    //     // ===== Step 3: 全员 Generate =====
    //     q.Generate(current_pt);   // 内部按 rank 分工 + MPI 汇总
        
    //     // ===== Step 4: 后续队列操作（仅 rank 0）=====
    //     if (rank == 0) {
    //         q.total_guesses = q.guesses.size();
            
    //         // NewPTs + 插入
    //         vector<PT> new_pts = current_pt.NewPTs();
    //         for (PT pt : new_pts) {
    //             CalProb(pt);
    //             // 按概率插入 priority...
    //             for (auto iter = priority.begin(); iter != priority.end(); iter++)
    //             {
    //                 // 对于非队首和队尾的特殊情况
    //                 if (iter != priority.end() - 1 && iter != priority.begin())
    //                 {
    //                     // 判定概率
    //                     if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
    //                     {
    //                         priority.emplace(iter + 1, pt);
    //                         break;
    //                     }
    //                 }
    //                 if (iter == priority.end() - 1)
    //                 {
    //                     priority.emplace_back(pt);
    //                     break;
    //                 }
    //                 if (iter == priority.begin() && iter->prob < pt.prob)
    //                 {
    //                     priority.emplace(iter, pt);
    //                     break;
    //                 }
    //             }
    //         }
    //         priority.erase(priority.begin());
            
    //         // 哈希判断、停止条件...
    //         q.total_guesses = q.guesses.size();
    //         if (q.total_guesses - curr_num >= 100000)
    //         {
    //             // cout << "Guesses generated: " <<history + q.total_guesses << endl;
    //             curr_num = q.total_guesses;

    //             // 在此处更改实验生成的猜测上限
    //             int generate_n=10000000;
    //             if (history + q.total_guesses > 10000000)
    //             {
    //                 auto end = system_clock::now();
    //                 auto duration = duration_cast<microseconds>(end - start);
    //                 time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    //                 cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;//请不要修改这一行
    //                 cout << "Hash time:" << time_hash << "seconds"<<endl;//请不要修改这一行
    //                 cout << "Train time:" << time_train <<"seconds"<<endl;//请不要修改这一行

    //                 q.stop_thread_pool();

    //                 // cout << "PriorityQueue::Generate中，进入多线程分支 " << q.pthread_count << "次"<<endl;
    //                 // cout << "进入单线程分支" << q.serial_count << "次"<<endl;
    //                 break;
    //             }
    //         }
    //         // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
    //         // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
    //         // if (q.guesses.size() >= 8) 发现性能没有提升，可能是条件过于苛刻，导致没有进入分支，我们修改进入分支的条件，只要基类够8个就执行
    //         if (curr_num > 500000) 
    //         {
    //             auto start_hash = system_clock::now();
    //             bit32 state_batch[4][4];
    //             int count = 0;
    //             string input_arry[4];
    //             for(string pw : q.guesses)
    //             {
    //                 input_arry[count++] = pw;
    //                 if (count == 4){
    //                     MD5HashBatch(input_arry, state_batch);
    //                     count = 0;
    //                 }
    //             }
    //             // 处理剩余不足4个的口令（用标量版本完成）
    //             if (count > 0)
    //             {
    //                 for (int i = 0; i < count; ++i)
    //                 {
    //                     bit32 state[4];
    //                     MD5Hash(input_arry[i], state); // 调用原始的标量 MD5Hash
    //                 }
    //             }

    //             // 在这里对哈希所需的总时长进行计算
    //             auto end_hash = system_clock::now();
    //             auto duration = duration_cast<microseconds>(end_hash - start_hash);
    //             time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

    //             // 记录已经生成的口令总数
    //             history += curr_num;
    //             curr_num = 0;
    //             q.guesses.clear();
    //         }
    //     }
    // }

    while (true)
    {
        // ===== Step 1: 检查队列是否空 =====
        int empty;
        if (rank == 0) {
            empty = q.priority.empty() ? 1 : 0;
        }
        MPI_Bcast(&empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (empty) break;

        // ===== Step 2: 全员 Generate（内部按 rank 分工 + MPI 汇总）=====
        PT pt_front;  // rank 1~7 用空的 PT 也无妨，Generate 内部会收广播
        if (rank == 0) {
            pt_front = q.priority.front();
        }
        q.Generate(pt_front);

        int should_stop = 0;
        // ===== Step 3: 队列操作（仅 rank 0）=====
        if (rank == 0) {
            q.total_guesses = q.guesses.size();

            // NewPTs + 插入
            vector<PT> new_pts = pt_front.NewPTs();
            for (PT& new_pt : new_pts) {
                q.CalProb(new_pt);
                // 按概率插入 priority
                for (auto iter = q.priority.begin(); iter != q.priority.end(); iter++) {
                    if (iter != q.priority.end() - 1 && iter != q.priority.begin()) {
                        if (new_pt.prob <= iter->prob && new_pt.prob > (iter + 1)->prob) {
                            q.priority.emplace(iter + 1, new_pt);
                            break;
                        }
                    }
                    if (iter == q.priority.end() - 1) {
                        q.priority.emplace_back(new_pt);
                        break;
                    }
                    if (iter == q.priority.begin() && iter->prob < new_pt.prob) {
                        q.priority.emplace(iter, new_pt);
                        break;
                    }
                }
            }
            q.priority.erase(q.priority.begin());

            // 进度判断和停止条件
            if (q.total_guesses - curr_num >= 100000) {
                curr_num = q.total_guesses;
                int generate_n = 10000000;
                if (history + q.total_guesses > 10000000) {
                    auto end = system_clock::now();
                    auto duration = duration_cast<microseconds>(end - start);
                    time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    should_stop = 1;
                }
            }

            // 哈希
            if (curr_num > 500000) {
                auto start_hash = system_clock::now();
                bit32 state_batch[4][4];
                int count = 0;
                string input_arry[4];
                for (string pw : q.guesses) {
                    input_arry[count++] = pw;
                    if (count == 4) {
                        MD5HashBatch(input_arry, state_batch);
                        count = 0;
                    }
                }
                if (count > 0) {
                    for (int i = 0; i < count; ++i) {
                        bit32 state[4];
                        MD5Hash(input_arry[i], state);
                    }
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // ===== Step 3.5: 广播停止信号（所有进程参与）=====
        MPI_Bcast(&should_stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (should_stop) {
            break; 
        }
    }

    MPI_Finalize();
}
