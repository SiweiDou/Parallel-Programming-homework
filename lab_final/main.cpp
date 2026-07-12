#include "PCFG.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mpi.h>
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
    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    time_t now = time(0);
    char* dt = ctime(&now);

    if (mpi_rank == 0) {
        cout << "Test started at: " << dt;
        cout << "Testing MD5Hash correctness..." << endl;
    }

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
            cout << "Rank " << mpi_rank << " MD5Hash test failed for " << test_pws[i] << "!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    if (mpi_rank == 0) {
        cout << "MD5Hash test passed!" << endl;
    }

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    string train_path = (argc > 1) ? argv[1] : "./guessdata/Rockyou-singleLined-full.txt";
    int generate_n = (argc > 2) ? atoi(argv[2]) : 10000000;
    int local_generate_limit = (generate_n + mpi_size - 1) / mpi_size;
    int gpu_threshold = (argc > 3) ? atoi(argv[3]) : 4096;
    int batch_flush_size = (argc > 4) ? atoi(argv[4]) : 131072;

    ConfigureGPUGenerate(gpu_threshold, batch_flush_size);

    if (mpi_rank == 0) {
        cout << "MPI world size: " << mpi_size << endl;
        cout << "Train data: " << train_path << endl;
        cout << "Generate limit: " << generate_n << endl;
        cout << "Local rank generate limit: " << local_generate_limit << endl;
        cout << "GPU generate threshold: " << gpu_threshold << endl;
        cout << "GPU batch flush size: " << batch_flush_size << endl;
    }

    PriorityQueue q;
    auto t0 = steady_clock::now();

    q.m.train(train_path);
    q.m.order();

    time_train = double(duration_cast<microseconds>(steady_clock::now() - t0).count()) * 1e-6;
    q.init();

    int curr_num = 0;
    int history = 0;
    int popped_pts = 0;
    auto start = steady_clock::now();

    while (!q.priority.empty()) {
        bool rank_owns_pt = (popped_pts % mpi_size == mpi_rank);
        q.PopNext(rank_owns_pt);
        popped_pts += 1;
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000) {
            curr_num = q.total_guesses;

            if (history + q.total_guesses > local_generate_limit) {
                FlushGPUBatch(q.guesses);
                time_guess = double(duration_cast<microseconds>(steady_clock::now() - start).count()) * 1e-6;
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

    FlushGPUBatch(q.guesses);

    GPUGenerateStats stats = GetGPUGenerateStats();

    long long local_generated = history + q.total_guesses;
    double local_guess_time = time_guess > 0 ? (time_guess - time_hash) : double(duration_cast<microseconds>(steady_clock::now() - start).count()) * 1e-6 - time_hash;

    long long global_generated = 0;
    long long global_gpu_items = 0;
    long long global_cpu_items = 0;
    long long global_cpu_threaded_items = 0;
    long long global_cpu_serial_items = 0;
    long long global_flush_count = 0;
    double max_guess_time = 0;
    double sum_hash_time = 0;
    double max_train_time = 0;
    double sum_gpu_wait_time = 0;
    double sum_gpu_stream_time = 0;
    double sum_cpu_generate_time = 0;

    MPI_Reduce(&local_generated, &global_generated, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.gpu_items, &global_gpu_items, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cpu_items, &global_cpu_items, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cpu_threaded_items, &global_cpu_threaded_items, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cpu_serial_items, &global_cpu_serial_items, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.flush_count, &global_flush_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_guess_time, &max_guess_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_hash, &sum_hash_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_train, &max_train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.gpu_wait_time, &sum_gpu_wait_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.gpu_stream_time, &sum_gpu_stream_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.cpu_generate_time, &sum_cpu_generate_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    cout << "Rank " << mpi_rank << " local generated items:" << local_generated << endl;
    cout << "Rank " << mpi_rank << " local guess time:" << local_guess_time << "seconds" << endl;
    cout << "Rank " << mpi_rank << " local hash time:" << time_hash << "seconds" << endl;
    cout << "Rank " << mpi_rank << " local GPU generate items:" << stats.gpu_items << endl;
    cout << "Rank " << mpi_rank << " local CPU OpenMP generate items:" << stats.cpu_threaded_items << endl;

    if (mpi_rank == 0) {
        cout << "Global generated items:" << global_generated << endl;
        cout << "Global guess time(max rank):" << max_guess_time << "seconds" << endl;
        cout << "Global hash time(sum ranks):" << sum_hash_time << "seconds" << endl;
        cout << "Global train time(max rank):" << max_train_time << "seconds" << endl;
        cout << "Global GPU generate items:" << global_gpu_items << endl;
        cout << "Global CPU generate items:" << global_cpu_items << endl;
        cout << "Global CPU OpenMP generate items:" << global_cpu_threaded_items << endl;
        cout << "Global CPU serial generate items:" << global_cpu_serial_items << endl;
        cout << "Global GPU batch flush count:" << global_flush_count << endl;
        cout << "Global GPU wait time(sum ranks):" << sum_gpu_wait_time << "seconds" << endl;
        cout << "Global GPU submit stream time(sum ranks):" << sum_gpu_stream_time << "seconds" << endl;
        cout << "Global CPU generate time(sum ranks):" << sum_cpu_generate_time << "seconds" << endl;
        cout << "CPU/GPU overlap enabled:" << stats.overlap_enabled << endl;
        cout << "Effective GPU threshold:" << stats.gpu_threshold << endl;
        cout << "Effective batch flush size:" << stats.batch_flush_size << endl;
        cout << "Dynamic GPU threshold:" << stats.dynamic_gpu_threshold << endl;
        cout << "Adaptive batch target:" << stats.adaptive_batch_target << endl;
    }

    MPI_Finalize();

    return 0;
}