#include "PCFG.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "md5.h"

using namespace std;
using namespace chrono;

static string BuildDigestString(const bit32* state)
{
    stringstream ss;

    for (int i = 0; i < 4; i++) {
        ss << setw(8) << setfill('0') << hex << state[i];
    }

    return ss.str();
}

int main(int argc, char **argv)
{
    time_t now = time(0);
    char* dt = ctime(&now);

    cout << "Test started at: " << dt;
    cout << "Testing MD5Hash correctness..." << endl;

    string test_pws[8] = {
        "123456", "password", "12345678", "qwerty",
        "123456789", "12345", "1234", "111111"
    };

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

        if (BuildDigestString(state) != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!\n";
            return 1;
        }
    }

    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    string train_path = (argc > 1) ? argv[1] : "./guessdata/Rockyou-singleLined-full.txt";
    int generate_n = (argc > 2) ? atoi(argv[2]) : 10000000;
    int gpu_threshold = (argc > 3) ? atoi(argv[3]) : 4096;
    int batch_flush_size = (argc > 4) ? atoi(argv[4]) : 131072;

    ConfigureGPUGenerate(gpu_threshold, batch_flush_size);

    cout << "Train data: " << train_path << endl;
    cout << "Generate limit: " << generate_n << endl;
    cout << "GPU generate threshold: " << gpu_threshold << endl;
    cout << "GPU batch flush size: " << batch_flush_size << endl;

    PriorityQueue q;
    auto t0 = steady_clock::now();

    q.m.train(train_path);
    q.m.order();

    time_train = double(duration_cast<microseconds>(steady_clock::now() - t0).count()) * 1e-6;
    q.init();

    int curr_num = 0;
    int history = 0;
    auto start = steady_clock::now();

    while (!q.priority.empty()) {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000) {
            curr_num = q.total_guesses;

            if (history + q.total_guesses > generate_n) {
                FlushGPUBatch(q.guesses);
                time_guess = double(duration_cast<microseconds>(steady_clock::now() - start).count()) * 1e-6;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }

        if (curr_num > 500000) {
            FlushGPUBatch(q.guesses);
            auto t_hash = steady_clock::now();
            int n = q.guesses.size();
            bit32* results = new bit32[n * 4];
            int done = MD5HashBatch_GPU_Raw(q.guesses.data(), n, results);

            if (done < n) {
                for (int i = 0; i < n; ++i) {
                    if (q.guesses[i].length() > 55) {
                        MD5Hash(q.guesses[i], results + i * 4);
                    }
                }
            }

            // 使用 steady_clock 后理论上不会出现负值；这里保留防御性处理，防止异常影响累计值。
            double dt_hash = double(duration_cast<microseconds>(steady_clock::now() - t_hash).count()) * 1e-6;

            if (dt_hash < 0) {
                dt_hash = 0;
            }

            time_hash += dt_hash;
            delete[] results;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    GPUGenerateStats stats = GetGPUGenerateStats();

    cout << "GPU generate items:" << stats.gpu_items << endl;
    cout << "CPU serial generate items:" << stats.cpu_items << endl;
    cout << "GPU batch flush count:" << stats.flush_count << endl;
    cout << "GPU async flush count:" << stats.async_flush_count << endl;
    cout << "GPU wait time:" << stats.gpu_wait_time << "seconds" << endl;
    cout << "CPU/GPU overlap enabled:" << stats.overlap_enabled << endl;
    cout << "Cached GPU segments:" << stats.cached_segments << endl;
    cout << "Effective GPU threshold:" << stats.gpu_threshold << endl;
    cout << "Effective batch flush size:" << stats.batch_flush_size << endl;

    return 0;
}