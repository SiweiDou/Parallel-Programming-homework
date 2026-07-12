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

// 将 MD5Hash 返回的 4 个 32 位状态字拼接为标准 32 位十六进制摘要字符串。
// 该函数只用于程序启动时的正确性自检，避免并行实验建立在错误 hash 实现之上。
static string BuildDigestString(const bit32* state)
{
    stringstream ss;

    for (int i = 0; i < 4; i++) {
        // 每个 state[i] 固定输出 8 个十六进制字符，不足部分补 0。
        ss << setw(8) << setfill('0') << hex << state[i];
    }

    return ss.str();
}

int main(int argc, char **argv)
{
    // 期末融合实验同时包含 MPI 维度，因此主程序必须先初始化 MPI 环境。
    // 即使单进程运行，mpi_size 也会是 1，后续分支仍可统一处理。
    MPI_Init(&argc, &argv);

    int mpi_rank = 0;
    int mpi_size = 1;
    // mpi_rank: 当前进程编号；mpi_size: 总进程数。候选 PT 按 popped_pts % mpi_size 划分。
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    time_t now = time(0);
    char* dt = ctime(&now);

    if (mpi_rank == 0) {
        // 只让 0 号 rank 打印全局启动信息，避免多进程重复输出。
        cout << "Test started at: " << dt;
        cout << "Testing MD5Hash correctness..." << endl;
    }

    // 使用常见弱口令及其公开 MD5 值做快速自检，覆盖 CPU 端单条 MD5Hash。
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

        // 任何 rank 的 hash 结果不一致都说明程序不可继续实验，直接 MPI_Abort 全局退出。
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
    // 命令行参数：argv[1]=训练集路径，argv[2]=全局生成上限，argv[3]=GPU 分流阈值，argv[4]=GPU batch 大小。
    // 不传参数时使用课程数据集和实验中默认的动态分流配置。
    string train_path = (argc > 1) ? argv[1] : "./guessdata/Rockyou-singleLined-full.txt";
    int generate_n = (argc > 2) ? atoi(argv[2]) : 10000000;
    // 每个 rank 的本地生成上限采用向上取整，保证全局生成量接近 generate_n。
    int local_generate_limit = (generate_n + mpi_size - 1) / mpi_size;
    int gpu_threshold = (argc > 3) ? atoi(argv[3]) : 4096;
    int batch_flush_size = (argc > 4) ? atoi(argv[4]) : 131072;

    // 将命令行参数传给 guessing.cu 中的 GPU/CPU 动态分流模块。
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

    // 所有 rank 都独立训练同一份 PCFG 模型，保证优先队列初始状态完全一致。
    // 这样 MPI 只需按 PT 序号划分生成任务，不需要广播复杂模型结构。
    q.m.train(train_path);
    q.m.order();

    time_train = double(duration_cast<microseconds>(steady_clock::now() - t0).count()) * 1e-6;
    // 根据排序后的 PT 初始化候选生成优先队列。
    q.init();

    int curr_num = 0;
    int history = 0;
    int popped_pts = 0;
    auto start = steady_clock::now();

    while (!q.priority.empty()) {
        // 所有 rank 都按同样顺序 PopNext，以保持 priority 队列一致；
        // 但只有满足 popped_pts % mpi_size == mpi_rank 的 rank 真正生成候选。
        bool rank_owns_pt = (popped_pts % mpi_size == mpi_rank);
        q.PopNext(rank_owns_pt);
        popped_pts += 1;
        q.total_guesses = q.guesses.size();

        // 每累计约 10 万个本地候选检查一次生成上限，避免每个 PT 都做昂贵判断。
        if (q.total_guesses - curr_num >= 100000) {
            curr_num = q.total_guesses;

            if (history + q.total_guesses > local_generate_limit) {
                // 到达本 rank 上限前，必须先等待异步 GPU 生成完成，否则 guesses 中可能仍有空字符串。
                FlushGPUBatch(q.guesses);
                time_guess = double(duration_cast<microseconds>(steady_clock::now() - start).count()) * 1e-6;
                break;
            }
        }

        // guesses 缓冲区超过 50 万后进行一次批量 hash，并清空缓冲区释放内存。
        // 生成阶段与 hash 阶段分开计时，便于报告中分析候选生成和 MD5 计算的占比。
        if (curr_num > 500000) {
            // hash 前同样需要 flush GPU 生成 batch，保证 CPU 端 guesses 已经完整。
            FlushGPUBatch(q.guesses);
            auto t_hash = steady_clock::now();
            int n = q.guesses.size();
            bit32* results = new bit32[n * 4];
            int done = MD5HashBatch_GPU_Raw(q.guesses.data(), n, results);

            if (done < n) {
                // GPU Raw kernel 只处理单块 MD5 的短口令。若出现超长口令，则回退 CPU MD5Hash。
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

            // history 记录已经完成 hash 的候选数；curr_num 和 guesses 清零后继续生成下一批。
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    // 主循环结束后，可能仍有最后一个未满 batch 的 GPU 生成任务，需要显式等待。
    FlushGPUBatch(q.guesses);

    // 获取本 rank 的 CPU/GPU 生成统计，后续通过 MPI_Reduce 汇总到 0 号 rank。
    GPUGenerateStats stats = GetGPUGenerateStats();

    long long local_generated = history + q.total_guesses;
    // 如果提前 break 已经记录 time_guess，则使用该值；否则用当前时刻补算生成总耗时。
    // local_guess_time 扣除 time_hash，近似得到纯候选生成阶段耗时。
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

    // 汇总策略：生成数量和 hash 时间使用 SUM；训练/生成阶段墙钟时间取 MAX，代表最慢 rank 决定并行总时间。
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

    // 每个 rank 打印本地统计，便于检查 MPI 负载是否均衡。
    cout << "Rank " << mpi_rank << " local generated items:" << local_generated << endl;
    cout << "Rank " << mpi_rank << " local guess time:" << local_guess_time << "seconds" << endl;
    cout << "Rank " << mpi_rank << " local hash time:" << time_hash << "seconds" << endl;
    cout << "Rank " << mpi_rank << " local GPU generate items:" << stats.gpu_items << endl;
    cout << "Rank " << mpi_rank << " local CPU OpenMP generate items:" << stats.cpu_threaded_items << endl;

    if (mpi_rank == 0) {
        // 0 号 rank 输出全局汇总指标，实验脚本主要解析这一组结果。
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

    // 所有 MPI 调用完成后释放 MPI 运行时资源。
    MPI_Finalize();

    return 0;
}