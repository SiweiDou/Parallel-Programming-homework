#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <ctime>
#include <cstdlib>
using namespace std;
using namespace chrono;

int main(int argc, char **argv)
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
    string train_path = (argc > 1) ? argv[1] : "./guessdata/Rockyou-singleLined-full.txt";
    int generate_n = (argc > 2) ? atoi(argv[2]) : 10000000;
    cout << "Train data: " << train_path << endl;
    cout << "Generate limit: " << generate_n << endl;

    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train(train_path);
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;//请不要修改这一行
                cout << "Hash time:" << time_hash << "seconds"<<endl;//请不要修改这一行
                cout << "Train time:" << time_train <<"seconds"<<endl;//请不要修改这一行
                break;
            }
        }
        // 当累计达到一定数量时，批量进行MD5哈希（GPU加速）
        if (curr_num > 500000) 
        {
            auto start_hash = system_clock::now();

            int n = q.guesses.size();
            bit32* results = new bit32[n * 4];

            // 使用GPU批量哈希（自动跳过>55字符的口令，这类口令极罕见）
            int gpu_done = MD5HashBatch_GPU(q.guesses.data(), n, results);

            // CPU fallback: 处理被GPU跳过的超长口令
            if (gpu_done < n)
            {
                for (int i = 0; i < n; ++i)
                {
                    if (q.guesses[i].length() > 55)
                    {
                        MD5Hash(q.guesses[i], results + i * 4);
                    }
                }
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            delete[] results;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}