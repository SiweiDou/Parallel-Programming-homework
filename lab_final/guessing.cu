#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <unordered_map>
using namespace std;
using namespace chrono;

#define GPU_MAXLEN 64

// g_gpu_threshold 和 g_batch_flush_size 是命令行给出的初始实验参数；
// g_dynamic_gpu_threshold 和 g_adaptive_batch_target 会在运行过程中根据观测吞吐做小范围自适应调整。
static int g_gpu_threshold = 4096;
static int g_batch_flush_size = 131072;
static int g_dynamic_gpu_threshold = 4096;
static int g_adaptive_batch_target = 131072;
// CPU_OMP_THRESHOLD 控制 CPU 生成路径是否启用 OpenMP。过小任务并行化会被线程调度开销抵消。
static const int CPU_OMP_THRESHOLD = 2048;
// 以下全局计数器只统计当前 MPI rank 内部的生成情况，main.cpp 会再通过 MPI_Reduce 汇总。
static long long g_gpu_items = 0;
static long long g_cpu_items = 0;
static long long g_cpu_threaded_items = 0;
static long long g_cpu_serial_items = 0;
static long long g_cpu_threaded_pt_count = 0;
static long long g_cpu_serial_pt_count = 0;
static long long g_flush_count = 0;
static long long g_async_flush_count = 0;
static long long g_submitted_gpu_items = 0;
static long long g_small_pt_count = 0;
static long long g_medium_pt_count = 0;
static long long g_large_pt_count = 0;
static long long g_adaptive_flush_count = 0;
static long long g_idle_flush_count = 0;
static long long g_max_pt_flush_count = 0;
static double g_gpu_wait_time = 0;
static double g_gpu_stream_time = 0;
static double g_cpu_generate_time = 0;

// 动态调度策略的边界参数。它们不是新的命令行实验变量，而是为了避免
// 自适应策略在少量样本下过度震荡：前期沿用手动阈值，积累观测后再微调。
static const int ADAPTIVE_MIN_BATCH_ITEMS = 32768;
static const int ADAPTIVE_MAX_BATCH_PTS = 96;
static const int LARGE_PT_MULTIPLIER = 4;
static const double GPU_MIN_WORK_SHARE = 0.80;
static const double GPU_HIGH_WORK_SHARE = 0.95;

// ==== GPU Generate kernels =====================================================
// 单段 PT 的 GPU 生成：每个线程负责生成一个候选口令。
// values 使用二维扁平数组保存，每个元素占用 max_vallen 字节。
__global__ void GenerateKernel_Single(
    const char *values, const int *val_lens,
    char *out_passwords, int *out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n)
    {
        // 网格按 block 向上取整，末尾线程可能超过实际候选数，必须直接退出。
        return;
    }

    // 每个线程读取 values 中第 tid 个定长槽位，并写入 out_passwords 中第 tid 个 GPU_MAXLEN 槽位。
    const char *src = values + tid * max_vallen;
    char *dst = out_passwords + tid * GPU_MAXLEN;
    int len = val_lens[tid];

    for (int i = 0; i < len; ++i)
    {
        dst[i] = src[i];
    }

    dst[len] = '\0';
    // out_lengths 独立保存真实长度，回传 CPU 时无需依赖 C 字符串扫描长度。
    out_lengths[tid] = len;
}

// 多段 PT 的 GPU 生成：CPU 先拼好前缀，GPU 负责枚举最后一个 segment。
// 这样可以保留 CPU 端复杂控制流，同时把规则化的大规模枚举交给 GPU。
__global__ void GenerateKernel_Multi(
    const char *prefix, int prefix_len,
    const char *values, const int *val_lens,
    char *out_passwords, int *out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n)
    {
        // 同单段 kernel，过滤掉向上取整产生的空线程。
        return;
    }

    const char *val_src = values + tid * max_vallen;
    char *dst = out_passwords + tid * GPU_MAXLEN;
    int val_len = val_lens[tid];
    int total = prefix_len + val_len;

    // 先拷贝 CPU 预先拼出的前缀，再拼接当前线程负责的最后一段 value。
    for (int i = 0; i < prefix_len; ++i)
    {
        dst[i] = prefix[i];
    }

    for (int i = 0; i < val_len; ++i)
    {
        dst[prefix_len + i] = val_src[i];
    }

    dst[total] = '\0';
    out_lengths[tid] = total;
}

// ==== Segment cache ============================================================
// segment 的 ordered_values 在训练完成后保持稳定，因此可以只上传一次到显存。
// 后续 PT 复用同一个 segment 时，直接复用显存指针，减少 H2D 拷贝开销。
struct GPUSegmentCache
{
    // d_values: segment 的所有候选取值，按 max_vallen 定长槽位扁平化后放在 GPU。
    char *d_values = nullptr;
    // d_val_lens: 每个候选取值的真实长度，避免 kernel 端 strlen。
    int *d_val_lens = nullptr;
    // max_vallen: 当前 segment 中最长取值长度，也是 d_values 每个槽位的步长。
    int max_vallen = 0;
};

// key 使用 segment*，因为训练结束后 m.letters/m.digits/m.symbols 中的 segment 对象稳定存在。
static unordered_map<segment *, GPUSegmentCache> g_seg_cache;

// ==== Multi-PT batch accumulator ==============================================
// 一个 batch 中可以包含多个 PT。每个 PT 记录输出在 guesses 中的起始位置，
// GPU 生成完后再统一 D2H 回填，避免每个 PT 单独同步和拷贝。
struct PTBatch
{
    // prefix: 多段 PT 中除最后一段外已经确定的前缀；单段 PT 为空。
    string prefix;
    // a: 当前 batch 需要枚举的最后一个 segment。
    segment *a;
    // total_items: 该 PT 展开后生成的候选数量，即最后一个 segment 的候选数。
    int total_items;
    // guesses_base: 该 PT 在 guesses 向量中的写入起点，用于 GPU 回填时恢复原始顺序。
    int guesses_base;
};

static vector<PTBatch> g_pt_batch;
static int g_batch_total = 0;

struct PendingGPUBatch
{
    // entries 保存已经提交但尚未 Wait 的 PT 元数据。
    vector<PTBatch> entries;
    // d_prefixes 保存多段 PT 的 GPU 前缀指针，Wait 后统一释放。
    vector<char *> d_prefixes;
    // total 是整个 pending batch 的候选总数。
    int total = 0;
    // active 标记当前是否有 GPU stream 正在异步执行。
    bool active = false;
};

static PendingGPUBatch g_pending_batch;
static cudaStream_t g_gpu_stream = nullptr;
static char *h_out = nullptr;
static int *h_lens = nullptr;
static int host_out_cap = 0;

static GPUSegmentCache &GetOrUploadSegment(segment *a);
static void SubmitGPUBatchAsync(vector<string> &guesses);
static void WaitPendingGPUBatch(vector<string> &guesses);
static void EnsureHostOutBuf(int n);
static bool ShouldUseGPU(int pt_items);
static bool ShouldFlushBatch(int next_pt_items, bool large_pt, bool *idle_flush, bool *max_pt_flush);
static void EnqueuePTBatch(vector<string> &guesses, const string &prefix, segment *a, int n, int guesses_base);
static void GenerateOnCPU(vector<string> &guesses, int guesses_base, const string &prefix, segment *a, int n);
static void UpdateAdaptivePolicy();

void ConfigureGPUGenerate(int gpu_threshold, int batch_flush_size)
{
    if (gpu_threshold > 0)
    {
        // 只有正数参数才覆盖默认值，避免脚本误传 0 导致所有 PT 都走 GPU 或 CPU。
        g_gpu_threshold = gpu_threshold;
        g_dynamic_gpu_threshold = gpu_threshold;
    }

    if (batch_flush_size > 0)
    {
        // batch 目标同样要求为正数，并同步初始化自适应目标。
        g_batch_flush_size = batch_flush_size;
        g_adaptive_batch_target = batch_flush_size;
    }
}

GPUGenerateStats GetGPUGenerateStats()
{
    // 将 guessing.cu 内部的全局计数器复制到结构体中，避免 main.cpp 直接依赖内部变量。
    GPUGenerateStats stats;
    stats.gpu_items = g_gpu_items;
    stats.cpu_items = g_cpu_items;
    stats.cpu_threaded_items = g_cpu_threaded_items;
    stats.cpu_serial_items = g_cpu_serial_items;
    stats.cpu_threaded_pt_count = g_cpu_threaded_pt_count;
    stats.cpu_serial_pt_count = g_cpu_serial_pt_count;
    stats.flush_count = g_flush_count;
    stats.async_flush_count = g_async_flush_count;
    stats.cached_segments = static_cast<long long>(g_seg_cache.size());
    stats.small_pt_count = g_small_pt_count;
    stats.medium_pt_count = g_medium_pt_count;
    stats.large_pt_count = g_large_pt_count;
    stats.adaptive_flush_count = g_adaptive_flush_count;
    stats.idle_flush_count = g_idle_flush_count;
    stats.max_pt_flush_count = g_max_pt_flush_count;
    stats.gpu_wait_time = g_gpu_wait_time;
    stats.gpu_stream_time = g_gpu_stream_time;
    stats.cpu_generate_time = g_cpu_generate_time;
    stats.avg_batch_items = g_flush_count > 0 ? double(g_submitted_gpu_items) / double(g_flush_count) : 0;
    stats.cpu_items_per_sec = g_cpu_generate_time > 0 ? double(g_cpu_items) / g_cpu_generate_time : 0;
    stats.gpu_items_per_sec = g_gpu_stream_time > 0 ? double(g_submitted_gpu_items) / g_gpu_stream_time : 0;
    stats.overlap_enabled = 1;
    stats.gpu_threshold = g_gpu_threshold;
    stats.batch_flush_size = g_batch_flush_size;
    stats.dynamic_gpu_threshold = g_dynamic_gpu_threshold;
    stats.adaptive_batch_target = g_adaptive_batch_target;
    return stats;
}

static int ClampInt(int value, int low, int high)
{
    // 将自适应参数限制在实验初值附近，避免短时间噪声把阈值推到极端。
    return max(low, min(value, high));
}

static int DynamicThresholdLow()
{
    return max(1024, g_gpu_threshold / 2);
}

static int DynamicThresholdHigh()
{
    return min(16384, max(g_gpu_threshold, g_gpu_threshold * 2));
}

static int AdaptiveBatchLow()
{
    return max(ADAPTIVE_MIN_BATCH_ITEMS, g_batch_flush_size / 2);
}

static int AdaptiveBatchHigh()
{
    return min(131072, max(g_batch_flush_size, g_batch_flush_size * 2));
}

// 根据已观测到的 CPU/GPU 吞吐和同步等待比例，周期性微调分流阈值和 batch 目标。
// 该策略保留命令行传入的 threshold/batch 作为初值，因此仍可与第四阶段参数扫描直接对比。
static void UpdateAdaptivePolicy()
{
    if (g_flush_count < 3 || g_cpu_items < 4096 || g_gpu_stream_time <= 0 || g_cpu_generate_time <= 0)
    {
        // 冷启动阶段样本不足，CPU/GPU 吞吐估计不可靠，暂不调整策略。
        return;
    }

    double cpu_rate = double(g_cpu_items) / g_cpu_generate_time;
    double gpu_rate = double(g_submitted_gpu_items) / g_gpu_stream_time;
    double avg_wait = g_flush_count > 0 ? g_gpu_wait_time / double(g_flush_count) : 0;
    double total_items = double(g_gpu_items + g_cpu_items);
    double gpu_share = total_items > 0 ? double(g_gpu_items) / total_items : 1.0;

    // 以静态扫描得到的阈值为中心做有限微调。GPU 工作占比过低时优先降低阈值，
    // 避免自适应策略把大部分候选生成错误地回退到 CPU。
    int next_threshold = g_dynamic_gpu_threshold;
    if (gpu_share < GPU_MIN_WORK_SHARE)
    {
        // GPU 工作占比偏低，说明太多 PT 被留给 CPU，降低阈值让更多中等 PT 进入 GPU。
        next_threshold = g_dynamic_gpu_threshold * 3 / 4;
    }
    else if (gpu_rate > cpu_rate * 6.0 && avg_wait < 0.0015)
    {
        // GPU 明显快且等待不重，继续降低阈值以扩大 GPU 覆盖范围。
        next_threshold = g_dynamic_gpu_threshold * 3 / 4;
    }
    else if (gpu_share > GPU_HIGH_WORK_SHARE && (gpu_rate < cpu_rate * 2.0 || avg_wait > 0.003))
    {
        // GPU 占比很高但优势不明显或等待较重，提高阈值，把小批量任务交还 CPU。
        next_threshold = g_dynamic_gpu_threshold * 5 / 4;
    }

    g_dynamic_gpu_threshold = ClampInt(next_threshold, DynamicThresholdLow(), DynamicThresholdHigh());

    // batch 目标也围绕命令行参数小范围调整。第四阶段实验中大 batch 并不占优，
    // 因此不再允许目标膨胀到过大的 262144。
    int next_batch = g_adaptive_batch_target;
    double avg_batch = g_flush_count > 0 ? double(g_submitted_gpu_items) / double(g_flush_count) : 0;
    if (avg_wait > 0.0015)
    {
        // GPU 等待偏长时缩小 batch，减少单次回填阻塞。
        next_batch = g_adaptive_batch_target * 3 / 4;
    }
    else if (avg_batch < g_adaptive_batch_target * 0.60 && avg_wait < 0.0008)
    {
        // 实际 batch 经常偏小且等待很短时，适当放大目标以减少 flush 次数。
        next_batch = g_adaptive_batch_target * 5 / 4;
    }

    g_adaptive_batch_target = ClampInt(next_batch, AdaptiveBatchLow(), AdaptiveBatchHigh());
}

// CPU/GPU 分流的核心代价模型。冷启动阶段使用阈值规则；有统计样本后，
// 比较“CPU 串行生成估计时间”和“GPU 固定提交开销 + GPU 吞吐估计时间”。
static bool ShouldUseGPU(int pt_items)
{
    UpdateAdaptivePolicy();

    if (pt_items < g_dynamic_gpu_threshold)
    {
        // 小 PT 直接留给 CPU，避免 GPU kernel 启动和同步成本超过枚举本身。
        g_small_pt_count += 1;
        return false;
    }

    if (pt_items >= g_dynamic_gpu_threshold * LARGE_PT_MULTIPLIER)
    {
        // 大 PT 通常能摊薄 GPU 启动成本，统计为 large 方便实验分析。
        g_large_pt_count += 1;
    }
    else
    {
        // 中等 PT 需要继续用代价模型判断。
        g_medium_pt_count += 1;
    }

    // 明显大于当前阈值的 PT 直接进入 GPU，避免短期吞吐估计误差导致 GPU 利用率过低。
    if (pt_items >= g_dynamic_gpu_threshold * 2)
    {
        return true;
    }

    if (g_flush_count < 3 || g_cpu_generate_time <= 0 || g_gpu_stream_time <= 0)
    {
        // 还没有足够吞吐样本时，达到阈值的 PT 默认交给 GPU，优先保证 GPU 利用率。
        return true;
    }

    double cpu_rate = double(g_cpu_items) / g_cpu_generate_time;
    double gpu_rate = double(g_submitted_gpu_items) / g_gpu_stream_time;
    double fixed_gpu_cost = g_gpu_wait_time / double(max(1LL, g_flush_count));
    double total_items = double(g_gpu_items + g_cpu_items);
    double gpu_share = total_items > 0 ? double(g_gpu_items) / total_items : 1.0;

    if (gpu_share < GPU_MIN_WORK_SHARE)
    {
        // 如果前期 GPU 分到的工作太少，则即使模型估计不确定，也优先把当前 PT 给 GPU。
        return true;
    }

    double cpu_cost = double(pt_items) / cpu_rate;
    double gpu_cost = fixed_gpu_cost + double(pt_items) / gpu_rate;

    return gpu_cost <= cpu_cost;
}

static bool ShouldFlushBatch(int next_pt_items, bool large_pt, bool *idle_flush, bool *max_pt_flush)
{
    *idle_flush = false;
    *max_pt_flush = false;

    if (g_batch_total >= g_adaptive_batch_target)
    {
        // 累计候选数已经达到目标 batch 大小，立即提交。
        return true;
    }

    if (static_cast<int>(g_pt_batch.size()) >= ADAPTIVE_MAX_BATCH_PTS)
    {
        // 即使候选总数没达到目标，也限制单个 batch 中 PT 数，防止元数据和 prefix 过多。
        *max_pt_flush = true;
        return true;
    }

    // 大 PT 单独或少量合并提交，避免一个超大 PT 拖住许多中等 PT 的回填。
    if (large_pt && g_batch_total >= ADAPTIVE_MIN_BATCH_ITEMS)
    {
        return true;
    }

    // 如果 GPU 当前没有 pending batch，而队列中已有足够工作量，则提前提交，
    // 让 GPU 与 CPU 后续 PopNext/小 PT Generate 重叠，而不是机械等待固定 PT 数或固定总量。
    if (!g_pending_batch.active && g_batch_total >= ADAPTIVE_MIN_BATCH_ITEMS &&
        g_batch_total + next_pt_items >= g_adaptive_batch_target / 2)
    {
        *idle_flush = true;
        return true;
    }

    return false;
}

static void GenerateOnCPU(vector<string> &guesses, int guesses_base, const string &prefix, segment *a, int n)
{
    auto t_cpu = steady_clock::now();
    if (n >= CPU_OMP_THRESHOLD)
    {
        // 大于阈值时使用 OpenMP 静态划分，每个循环迭代写 guesses 的独立位置，不存在写冲突。
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            if (prefix.empty())
            {
                // 单段 PT：候选就是当前 segment 的第 i 个高频取值。
                guesses[guesses_base + i] = a->ordered_values[i];
            }
            else
            {
                // 多段 PT：前缀已经由 CPU 固定，当前循环只枚举最后一段。
                guesses[guesses_base + i] = prefix + a->ordered_values[i];
            }
        }

        g_cpu_threaded_items += n;
        g_cpu_threaded_pt_count += 1;
    }
    else
    {
        // 小 PT 串行生成，避免 OpenMP 线程调度成本。
        if (prefix.empty())
        {
            for (int i = 0; i < n; ++i)
            {
                guesses[guesses_base + i] = a->ordered_values[i];
            }
        }
        else
        {
            for (int i = 0; i < n; ++i)
            {
                guesses[guesses_base + i] = prefix + a->ordered_values[i];
            }
        }

        g_cpu_serial_items += n;
        g_cpu_serial_pt_count += 1;
    }

    g_cpu_generate_time += double(duration_cast<microseconds>(steady_clock::now() - t_cpu).count()) * 1e-6;
    g_cpu_items += n;
}

static void EnqueuePTBatch(vector<string> &guesses, const string &prefix, segment *a, int n, int guesses_base)
{
    // 只记录 PT 元数据，不立即启动 kernel；多个 PT 合并后一次提交可摊薄 GPU 固定开销。
    PTBatch e;
    e.prefix = prefix;
    e.a = a;
    e.total_items = n;
    e.guesses_base = guesses_base;

    g_pt_batch.push_back(e);
    g_batch_total += n;
    g_gpu_items += n;

    bool idle_flush = false;
    bool max_pt_flush = false;
    bool large_pt = n >= g_dynamic_gpu_threshold * LARGE_PT_MULTIPLIER;
    if (ShouldFlushBatch(n, large_pt, &idle_flush, &max_pt_flush))
    {
        // 根据 flush 原因维护统计，后续报告可判断 batch 是被容量、空闲还是 PT 数上限触发。
        g_adaptive_flush_count += 1;
        if (idle_flush)
        {
            g_idle_flush_count += 1;
        }
        if (max_pt_flush)
        {
            g_max_pt_flush_count += 1;
        }
        SubmitGPUBatchAsync(guesses);
    }
}

// batch 执行前先保证所有涉及的 segment 已经在显存中。
static void PreUploadSegments()
{
    for (auto &e : g_pt_batch)
    {
        GetOrUploadSegment(e.a);
    }
}

static GPUSegmentCache &GetOrUploadSegment(segment *a)
{
    auto it = g_seg_cache.find(a);

    if (it != g_seg_cache.end())
    {
        // 命中缓存时直接返回 GPU 指针，避免重复 H2D 拷贝。
        return it->second;
    }

    GPUSegmentCache c;
    int n = a->ordered_values.size();

    for (int i = 0; i < n; ++i)
    {
        int current_len = static_cast<int>(a->ordered_values[i].length());

        if (current_len > c.max_vallen)
        {
            // 计算定长槽位宽度，之后按 n * max_vallen 扁平化存储。
            c.max_vallen = current_len;
        }
    }

    if (c.max_vallen > 55)
    {
        // 本实验 GPU MD5 只处理单个 512-bit block，候选有效载荷最多 55 字节。
        c.max_vallen = 55;
    }

    char *h_vals = new char[n * c.max_vallen];
    int *h_lens = new int[n];

    memset(h_vals, 0, n * c.max_vallen);

    for (int i = 0; i < n; ++i)
    {
        int len = min(static_cast<int>(a->ordered_values[i].length()), c.max_vallen);
        memcpy(h_vals + i * c.max_vallen, a->ordered_values[i].c_str(), len);
        h_lens[i] = len;
    }

    // segment 内容上传到 GPU 后缓存，后续相同 segment 不再重复传输。
    cudaMalloc(&c.d_values, n * c.max_vallen);
    cudaMalloc(&c.d_val_lens, n * sizeof(int));
    cudaMemcpy(c.d_values, h_vals, n * c.max_vallen, cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_val_lens, h_lens, n * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_vals;
    delete[] h_lens;

    g_seg_cache[a] = c;
    return g_seg_cache[a];
}

// GPU 输出缓冲区跨 batch 复用，避免每次 flush 都重新分配显存。
static char *d_out = nullptr;
static int *d_out_lens = nullptr;
static int out_cap = 0;

static void EnsureOutBuf(int n)
{
    if (n <= out_cap)
    {
        // 当前设备输出缓冲区容量足够，直接复用。
        return;
    }

    if (d_out)
    {
        // 容量不足时释放旧缓冲区，再按更大的容量重新分配。
        cudaFree(d_out);
        cudaFree(d_out_lens);
    }

    out_cap = n + 65536;
    cudaMalloc(&d_out, out_cap * GPU_MAXLEN);
    cudaMalloc(&d_out_lens, out_cap * sizeof(int));
}

// 异步 GPU batch 完成后，把已经拷回 CPU 的结果按每个 PT 的原始位置回填。
static void FillGuessesFromPending(vector<string> &guesses)
{
    int total = g_pending_batch.total;

    if (total == 0)
    {
        // 没有 pending 输出时无需回填。
        return;
    }

    int gpu_offset = 0;
    for (auto &e : g_pending_batch.entries)
    {
        // 每个 PT 在 GPU 输出缓冲区中连续存放，在 guesses 中按 guesses_base 回到原始位置。
        for (int i = 0; i < e.total_items; ++i)
        {
            guesses[e.guesses_base + i] = string(
                h_out + (gpu_offset + i) * GPU_MAXLEN,
                h_lens[gpu_offset + i]);
        }

        gpu_offset += e.total_items;
    }
}

static void EnsureHostOutBuf(int n)
{
    if (n <= host_out_cap)
    {
        // pinned host 缓冲区容量足够，继续复用以提高 D2H 拷贝效率。
        return;
    }

    if (h_out)
    {
        // 重新分配 pinned memory 前必须释放旧缓冲区。
        cudaFreeHost(h_out);
        cudaFreeHost(h_lens);
    }

    host_out_cap = n + 65536;
    cudaMallocHost(&h_out, host_out_cap * GPU_MAXLEN);
    cudaMallocHost(&h_lens, host_out_cap * sizeof(int));
}

// 将累计的多个 PT 异步提交给 GPU。CPU 可继续生成后续小 PT，直到下一次显式等待。
static void SubmitGPUBatchAsync(vector<string> &guesses)
{
    if (g_pt_batch.empty())
    {
        // 没有累计 PT 时不提交空 batch。
        return;
    }

    // 当前实现只维护一个 pending batch；提交新 batch 前先等待上一批完成，
    // 仍然可以与 CPU 后续生成重叠，因为 Submit 后到下一次 Submit/Flush 之间 CPU 会继续执行。
    WaitPendingGPUBatch(guesses);

    auto t_stream = steady_clock::now();

    PreUploadSegments();
    int total = g_batch_total;
    EnsureOutBuf(total);
    EnsureHostOutBuf(total);

    if (g_gpu_stream == nullptr)
    {
        // 懒创建 stream，避免未使用 GPU 路径时产生额外初始化开销。
        cudaStreamCreate(&g_gpu_stream);
    }

    g_pending_batch.entries = g_pt_batch;
    g_pending_batch.total = total;
    g_pending_batch.active = true;
    g_pending_batch.d_prefixes.clear();
    g_pending_batch.d_prefixes.reserve(g_pending_batch.entries.size());

    int gpu_offset = 0;
    for (auto &e : g_pending_batch.entries)
    {
        GPUSegmentCache &c = g_seg_cache[e.a];
        int block = 256;
        int grid = (e.total_items + block - 1) / block;

        if (e.prefix.empty())
        {
            // 单段 PT 无需前缀，直接把 segment 的每个 value 写入输出缓冲区。
            GenerateKernel_Single<<<grid, block, 0, g_gpu_stream>>>(
                c.d_values, c.d_val_lens,
                d_out + gpu_offset * GPU_MAXLEN,
                d_out_lens + gpu_offset,
                e.total_items, c.max_vallen);
            g_pending_batch.d_prefixes.push_back(nullptr);
        }
        else
        {
            // 多段 PT 的前缀由 CPU 端生成。前缀较短，单独 H2D 开销可接受。
            int plen = static_cast<int>(e.prefix.length());
            char *d_prefix = nullptr;

            cudaMalloc(&d_prefix, plen);
            cudaMemcpyAsync(d_prefix, e.prefix.c_str(), plen, cudaMemcpyHostToDevice, g_gpu_stream);

            GenerateKernel_Multi<<<grid, block, 0, g_gpu_stream>>>(
                d_prefix, plen,
                c.d_values, c.d_val_lens,
                d_out + gpu_offset * GPU_MAXLEN,
                d_out_lens + gpu_offset,
                e.total_items, c.max_vallen);

            g_pending_batch.d_prefixes.push_back(d_prefix);
        }

        gpu_offset += e.total_items;
    }

    cudaMemcpyAsync(h_out, d_out, total * GPU_MAXLEN, cudaMemcpyDeviceToHost, g_gpu_stream);
    cudaMemcpyAsync(h_lens, d_out_lens, total * sizeof(int), cudaMemcpyDeviceToHost, g_gpu_stream);

    // 这里只记录提交阶段耗时；真正等待 GPU 完成的时间在 WaitPendingGPUBatch 中统计。
    g_async_flush_count += 1;
    g_flush_count += 1;
    g_submitted_gpu_items += total;
    g_gpu_stream_time += double(duration_cast<microseconds>(steady_clock::now() - t_stream).count()) * 1e-6;
    g_pt_batch.clear();
    g_batch_total = 0;
}

static void WaitPendingGPUBatch(vector<string> &guesses)
{
    if (!g_pending_batch.active)
    {
        // 没有异步 batch 在执行时直接返回。
        return;
    }

    auto t_wait = steady_clock::now();
    // 等待当前 stream 中 kernel 和 D2H 拷贝全部完成。
    cudaStreamSynchronize(g_gpu_stream);
    g_gpu_wait_time += double(duration_cast<microseconds>(steady_clock::now() - t_wait).count()) * 1e-6;

    FillGuessesFromPending(guesses);

    for (char *d_prefix : g_pending_batch.d_prefixes)
    {
        if (d_prefix)
        {
            // 多段 PT 的前缀是每次 batch 临时分配的，必须在 stream 完成后释放。
            cudaFree(d_prefix);
        }
    }

    g_pending_batch = PendingGPUBatch();
}

// 等待所有已提交 GPU batch 完成；主循环进入 hash 前调用该函数保证 guesses 完整。
void FlushGPUBatch(vector<string> &guesses)
{
    SubmitGPUBatchAsync(guesses);
    WaitPendingGPUBatch(guesses);
}

// ==== PriorityQueue methods ===================================================
void PriorityQueue::CalProb(PT &pt)
{
    // 概率初值是 PT 结构本身的先验概率。
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            // 字母段条件概率 = 当前字母取值频率 / 同长度字母段总频率。
            int letter_index = m.FindLetter(pt.content[index]);
            pt.prob *= m.letters[letter_index].ordered_freqs[idx];
            pt.prob /= m.letters[letter_index].total_freq;
        }
        else if (pt.content[index].type == 2)
        {
            // 数字段同理，从 digits 统计表中取条件概率。
            int digit_index = m.FindDigit(pt.content[index]);
            pt.prob *= m.digits[digit_index].ordered_freqs[idx];
            pt.prob /= m.digits[digit_index].total_freq;
        }
        else if (pt.content[index].type == 3)
        {
            // 符号段同理，从 symbols 统计表中取条件概率。
            int symbol_index = m.FindSymbol(pt.content[index]);
            pt.prob *= m.symbols[symbol_index].ordered_freqs[idx];
            pt.prob /= m.symbols[symbol_index].total_freq;
        }

        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        // 对每个 PT 模板，按段类型查出该段可枚举的候选数量，填入 max_indices。
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }

            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }

            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        // 初始化队列时每个 PT 的 curr_indices 全为 0，即每段都取最高频值。
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    PopNext(true);
}

// MPI 融合版本要求所有 rank 以完全相同的顺序推进优先队列，
// 但只有被分配到当前 PT 的 rank 真正生成候选口令。
void PriorityQueue::PopNext(bool do_generate)
{
    if (do_generate)
    {
        // 当前 rank 拥有该 PT 时才真正生成候选。
        Generate(priority.front());
    }

    // 无论是否生成候选，所有 rank 都必须生成并插入后继 PT，以保持队列状态一致。
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    // 插入到两个概率之间，维持队列按概率降序排列。
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }

            if (iter == priority.end() - 1)
            {
                // 如果扫描到队尾仍未找到更小位置，则追加到队尾。
                priority.emplace_back(pt);
                break;
            }

            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                // 新状态概率高于当前队首时插入到最前面。
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    // 队首状态已经处理完毕，最后删除它。
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        // 单段 PT 没有“前缀段”和“后继组合”的概念，展开一次即可。
        return res;
    }

    int init_pivot = pivot;

    for (int i = pivot; i < curr_indices.size() - 1; i += 1)
    {
        curr_indices[i] += 1;

        if (curr_indices[i] < max_indices[i])
        {
            // 当前段下标加一仍合法，生成一个新的相邻状态。
            pivot = i;
            res.emplace_back(*this);
        }

        // 恢复原状态，继续尝试下一个可推进位置。
        curr_indices[i] -= 1;
    }

    pivot = init_pivot;
    return res;
}

// ==== Generate: CPU (small) or batch-queue GPU (large) ========================
// 每次 Generate 只展开当前 PT 的最后一个 segment。短 PT/小规模展开留给 CPU，
// 其中超过 CPU_OMP_THRESHOLD 的部分走 OpenMP；大规模规则化枚举进入 GPU batch。
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // 单段 PT：直接枚举该段所有候选值，不需要拼接前缀。
        segment *a = nullptr;

        if (pt.content[0].type == 1)
        {
            // L 段从 letters 表中取对应长度的 segment。
            a = &m.letters[m.FindLetter(pt.content[0])];
        }

        if (pt.content[0].type == 2)
        {
            // D 段从 digits 表中取对应长度的 segment。
            a = &m.digits[m.FindDigit(pt.content[0])];
        }

        if (pt.content[0].type == 3)
        {
            // S 段从 symbols 表中取对应长度的 segment。
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int n = pt.max_indices[0];
        old_size = guesses.size();

        // 动态分流：小 PT 留在 CPU，大 PT 进入 GPU batch，中等 PT 由代价模型判断。
        if (ShouldUseGPU(n))
        {
            // GPU 路径先扩容 guesses，实际字符串稍后由异步 batch 回填。
            guesses.resize(old_size + n);
            EnqueuePTBatch(guesses, "", a, n, old_size);
        }
        else
        {
            // CPU 路径立即在 guesses 中写入完整候选。
            guesses.resize(old_size + n);
            GenerateOnCPU(guesses, old_size, "", a, n);
        }
    }
    else
    {
        // 多段 PT：先根据 curr_indices 固定除最后一段外的前缀，最后一段再批量枚举。
        string pre;
        int s = 0;

        for (int idx : pt.curr_indices)
        {
            if (pt.content[s].type == 1)
            {
                // 前缀中的字母段使用当前状态指定的 ordered_values 下标。
                pre += m.letters[m.FindLetter(pt.content[s])].ordered_values[idx];
            }

            if (pt.content[s].type == 2)
            {
                // 前缀中的数字段同理。
                pre += m.digits[m.FindDigit(pt.content[s])].ordered_values[idx];
            }

            if (pt.content[s].type == 3)
            {
                // 前缀中的符号段同理。
                pre += m.symbols[m.FindSymbol(pt.content[s])].ordered_values[idx];
            }

            s++;

            if (s == pt.content.size() - 1)
            {
                // 最后一段不加入前缀，因为它是本次 Generate 要枚举的维度。
                break;
            }
        }

        segment *a = nullptr;
        int last_index = static_cast<int>(pt.content.size()) - 1;

        if (pt.content[last_index].type == 1)
        {
            // 根据最后一段类型选择要枚举的 segment 表。
            a = &m.letters[m.FindLetter(pt.content[last_index])];
        }

        if (pt.content[last_index].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_index])];
        }

        if (pt.content[last_index].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_index])];
        }

        int n = pt.max_indices[last_index];
        old_size = guesses.size();

        if (ShouldUseGPU(n))
        {
            // 多段 GPU 路径会把 pre 作为 prefix 上传，GPU 只枚举最后一段。
            guesses.resize(old_size + n);
            EnqueuePTBatch(guesses, pre, a, n, old_size);
        }
        else
        {
            // 多段 CPU 路径直接执行 prefix + ordered_values[i] 拼接。
            guesses.resize(old_size + n);
            GenerateOnCPU(guesses, old_size, pre, a, n);
        }
    }
}