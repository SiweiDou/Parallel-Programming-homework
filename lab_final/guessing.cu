#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cstring>
#include <unordered_map>
using namespace std;
using namespace chrono;

#define GPU_MAXLEN 64

static int g_gpu_threshold = 4096;
static int g_batch_flush_size = 131072;
static long long g_gpu_items = 0;
static long long g_cpu_items = 0;
static long long g_flush_count = 0;
static long long g_async_flush_count = 0;
static double g_gpu_wait_time = 0;

// ==== GPU Generate kernels =====================================================
// 单段 PT 的 GPU 生成：每个线程负责生成一个候选口令。
// values 使用二维扁平数组保存，每个元素占用 max_vallen 字节。
__global__ void GenerateKernel_Single(
    const char* values, const int* val_lens,
    char* out_passwords, int* out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    const char* src = values + tid * max_vallen;
    char* dst = out_passwords + tid * GPU_MAXLEN;
    int len = val_lens[tid];

    for (int i = 0; i < len; ++i) {
        dst[i] = src[i];
    }

    dst[len] = '\0';
    out_lengths[tid] = len;
}

// 多段 PT 的 GPU 生成：CPU 先拼好前缀，GPU 负责枚举最后一个 segment。
// 这样可以保留 CPU 端复杂控制流，同时把规则化的大规模枚举交给 GPU。
__global__ void GenerateKernel_Multi(
    const char* prefix, int prefix_len,
    const char* values, const int* val_lens,
    char* out_passwords, int* out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) {
        return;
    }

    const char* val_src = values + tid * max_vallen;
    char* dst = out_passwords + tid * GPU_MAXLEN;
    int val_len = val_lens[tid];
    int total = prefix_len + val_len;

    for (int i = 0; i < prefix_len; ++i) {
        dst[i] = prefix[i];
    }

    for (int i = 0; i < val_len; ++i) {
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
    char* d_values = nullptr;
    int* d_val_lens = nullptr;
    int max_vallen = 0;
};

static unordered_map<segment*, GPUSegmentCache> g_seg_cache;

// ==== Multi-PT batch accumulator ==============================================
// 一个 batch 中可以包含多个 PT。每个 PT 记录输出在 guesses 中的起始位置，
// GPU 生成完后再统一 D2H 回填，避免每个 PT 单独同步和拷贝。
struct PTBatch
{
    string prefix;
    segment* a;
    int total_items;
    int guesses_base;
};

static vector<PTBatch> g_pt_batch;
static int g_batch_total = 0;

struct PendingGPUBatch
{
    vector<PTBatch> entries;
    vector<char*> d_prefixes;
    int total = 0;
    bool active = false;
};

static PendingGPUBatch g_pending_batch;
static cudaStream_t g_gpu_stream = nullptr;
static char* h_out = nullptr;
static int* h_lens = nullptr;
static int host_out_cap = 0;

static GPUSegmentCache& GetOrUploadSegment(segment* a);
static void SubmitGPUBatchAsync(vector<string>& guesses);
static void WaitPendingGPUBatch(vector<string>& guesses);
static void EnsureHostOutBuf(int n);

void ConfigureGPUGenerate(int gpu_threshold, int batch_flush_size)
{
    if (gpu_threshold > 0) {
        g_gpu_threshold = gpu_threshold;
    }

    if (batch_flush_size > 0) {
        g_batch_flush_size = batch_flush_size;
    }
}

GPUGenerateStats GetGPUGenerateStats()
{
    GPUGenerateStats stats;
    stats.gpu_items = g_gpu_items;
    stats.cpu_items = g_cpu_items;
    stats.flush_count = g_flush_count;
    stats.async_flush_count = g_async_flush_count;
    stats.cached_segments = static_cast<long long>(g_seg_cache.size());
    stats.gpu_wait_time = g_gpu_wait_time;
    stats.overlap_enabled = 1;
    stats.gpu_threshold = g_gpu_threshold;
    stats.batch_flush_size = g_batch_flush_size;
    return stats;
}

// batch 执行前先保证所有涉及的 segment 已经在显存中。
static void PreUploadSegments()
{
    for (auto& e : g_pt_batch) {
        GetOrUploadSegment(e.a);
    }
}

static GPUSegmentCache& GetOrUploadSegment(segment* a)
{
    auto it = g_seg_cache.find(a);

    if (it != g_seg_cache.end()) {
        return it->second;
    }

    GPUSegmentCache c;
    int n = a->ordered_values.size();

    for (int i = 0; i < n; ++i) {
        int current_len = static_cast<int>(a->ordered_values[i].length());

        if (current_len > c.max_vallen) {
            c.max_vallen = current_len;
        }
    }

    if (c.max_vallen > 55) {
        c.max_vallen = 55;
    }

    char* h_vals = new char[n * c.max_vallen];
    int* h_lens = new int[n];

    memset(h_vals, 0, n * c.max_vallen);

    for (int i = 0; i < n; ++i) {
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
static char* d_out = nullptr;
static int* d_out_lens = nullptr;
static int out_cap = 0;

static void EnsureOutBuf(int n)
{
    if (n <= out_cap) {
        return;
    }

    if (d_out) {
        cudaFree(d_out);
        cudaFree(d_out_lens);
    }

    out_cap = n + 65536;
    cudaMalloc(&d_out, out_cap * GPU_MAXLEN);
    cudaMalloc(&d_out_lens, out_cap * sizeof(int));
}

// 异步 GPU batch 完成后，把已经拷回 CPU 的结果按每个 PT 的原始位置回填。
static void FillGuessesFromPending(vector<string>& guesses)
{
    int total = g_pending_batch.total;

    if (total == 0) {
        return;
    }

    int gpu_offset = 0;
    for (auto& e : g_pending_batch.entries) {
        for (int i = 0; i < e.total_items; ++i) {
            guesses[e.guesses_base + i] = string(
                h_out + (gpu_offset + i) * GPU_MAXLEN,
                h_lens[gpu_offset + i]);
        }

        gpu_offset += e.total_items;
    }
}

static void EnsureHostOutBuf(int n)
{
    if (n <= host_out_cap) {
        return;
    }

    if (h_out) {
        cudaFreeHost(h_out);
        cudaFreeHost(h_lens);
    }

    host_out_cap = n + 65536;
    cudaMallocHost(&h_out, host_out_cap * GPU_MAXLEN);
    cudaMallocHost(&h_lens, host_out_cap * sizeof(int));
}

// 将累计的多个 PT 异步提交给 GPU。CPU 可继续生成后续小 PT，直到下一次显式等待。
static void SubmitGPUBatchAsync(vector<string>& guesses)
{
    if (g_pt_batch.empty()) {
        return;
    }

    WaitPendingGPUBatch(guesses);

    PreUploadSegments();
    int total = g_batch_total;
    EnsureOutBuf(total);
    EnsureHostOutBuf(total);

    if (g_gpu_stream == nullptr) {
        cudaStreamCreate(&g_gpu_stream);
    }

    g_pending_batch.entries = g_pt_batch;
    g_pending_batch.total = total;
    g_pending_batch.active = true;
    g_pending_batch.d_prefixes.clear();
    g_pending_batch.d_prefixes.reserve(g_pending_batch.entries.size());

    int gpu_offset = 0;
    for (auto& e : g_pending_batch.entries) {
        GPUSegmentCache& c = g_seg_cache[e.a];
        int block = 256;
        int grid = (e.total_items + block - 1) / block;

        if (e.prefix.empty()) {
            // 单段 PT 无需前缀，直接把 segment 的每个 value 写入输出缓冲区。
            GenerateKernel_Single<<<grid, block, 0, g_gpu_stream>>>(
                c.d_values, c.d_val_lens,
                d_out + gpu_offset * GPU_MAXLEN,
                d_out_lens + gpu_offset,
                e.total_items, c.max_vallen);
            g_pending_batch.d_prefixes.push_back(nullptr);
        } else {
            // 多段 PT 的前缀由 CPU 端生成。前缀较短，单独 H2D 开销可接受。
            int plen = static_cast<int>(e.prefix.length());
            char* d_prefix = nullptr;

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

    g_async_flush_count += 1;
    g_flush_count += 1;
    g_pt_batch.clear();
    g_batch_total = 0;
}

static void WaitPendingGPUBatch(vector<string>& guesses)
{
    if (!g_pending_batch.active) {
        return;
    }

    auto t_wait = steady_clock::now();
    cudaStreamSynchronize(g_gpu_stream);
    g_gpu_wait_time += double(duration_cast<microseconds>(steady_clock::now() - t_wait).count()) * 1e-6;

    FillGuessesFromPending(guesses);

    for (char* d_prefix : g_pending_batch.d_prefixes) {
        if (d_prefix) {
            cudaFree(d_prefix);
        }
    }

    g_pending_batch = PendingGPUBatch();
}

// 等待所有已提交 GPU batch 完成；主循环进入 hash 前调用该函数保证 guesses 完整。
void FlushGPUBatch(vector<string>& guesses)
{
    SubmitGPUBatchAsync(guesses);
    WaitPendingGPUBatch(guesses);
}

// ==== PriorityQueue methods ===================================================
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            int letter_index = m.FindLetter(pt.content[index]);
            pt.prob *= m.letters[letter_index].ordered_freqs[idx];
            pt.prob /= m.letters[letter_index].total_freq;
        } else if (pt.content[index].type == 2) {
            int digit_index = m.FindDigit(pt.content[index]);
            pt.prob *= m.digits[digit_index].ordered_freqs[idx];
            pt.prob /= m.digits[digit_index].total_freq;
        } else if (pt.content[index].type == 3) {
            int symbol_index = m.FindSymbol(pt.content[index]);
            pt.prob *= m.symbols[symbol_index].ordered_freqs[idx];
            pt.prob /= m.symbols[symbol_index].total_freq;
        }

        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }

            if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }

            if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }

            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }

            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1) {
        return res;
    }

    int init_pivot = pivot;

    for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
        curr_indices[i] += 1;

        if (curr_indices[i] < max_indices[i]) {
            pivot = i;
            res.emplace_back(*this);
        }

        curr_indices[i] -= 1;
    }

    pivot = init_pivot;
    return res;
}

// ==== Generate: CPU (small) or batch-queue GPU (large) ========================
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1) {
        segment *a = nullptr;

        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }

        if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }

        if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int n = pt.max_indices[0]; 
        old_size = guesses.size();

        // 小任务继续由 CPU 串行生成，避免 GPU kernel 启动和同步开销反超收益。
        if (n > g_gpu_threshold) {
            guesses.resize(old_size + n);
            PTBatch e;

            e.prefix = ""; 
            e.a = a; 
            e.total_items = n; 
            e.guesses_base = old_size;

            g_pt_batch.push_back(e); 
            g_batch_total += n;
            g_gpu_items += n;

            if (g_batch_total >= g_batch_flush_size) {
                SubmitGPUBatchAsync(guesses);
            }
        } else { 
            guesses.resize(old_size + n); 

            for (int i = 0; i < n; ++i) {
                guesses[old_size + i] = a->ordered_values[i]; 
            }

            g_cpu_items += n;
        }
    } else {
        string pre;
        int s = 0;

        for (int idx : pt.curr_indices) {
            if (pt.content[s].type == 1) {
                pre += m.letters[m.FindLetter(pt.content[s])].ordered_values[idx];
            }

            if (pt.content[s].type == 2) {
                pre += m.digits[m.FindDigit(pt.content[s])].ordered_values[idx];
            }

            if (pt.content[s].type == 3) {
                pre += m.symbols[m.FindSymbol(pt.content[s])].ordered_values[idx];
            }

            s++; 

            if (s == pt.content.size() - 1) {
                break;
            }
        }

        segment *a = nullptr;
        int last_index = static_cast<int>(pt.content.size()) - 1;

        if (pt.content[last_index].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[last_index])];
        }

        if (pt.content[last_index].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[last_index])];
        }

        if (pt.content[last_index].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[last_index])];
        }

        int n = pt.max_indices[last_index]; 
        old_size = guesses.size();

        if (n > g_gpu_threshold) {
            guesses.resize(old_size + n);
            PTBatch e;

            e.prefix = pre; 
            e.a = a; 
            e.total_items = n; 
            e.guesses_base = old_size;

            g_pt_batch.push_back(e); 
            g_batch_total += n;
            g_gpu_items += n;

            if (g_batch_total >= g_batch_flush_size) {
                SubmitGPUBatchAsync(guesses);
            }
        } else { 
            guesses.resize(old_size + n); 

            for (int i = 0; i < n; ++i) {
                guesses[old_size + i] = pre + a->ordered_values[i]; 
            }

            g_cpu_items += n;
        }
    }
}