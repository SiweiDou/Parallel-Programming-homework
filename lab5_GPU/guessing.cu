#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unordered_map>
using namespace std;

#define GPU_MAXLEN 64
#define GPU_THRESHOLD 4096

// ==== GPU Generate kernels ====================================================
__global__ void GenerateKernel_Single(
    const char* values, const int* val_lens,
    char* out_passwords, int* out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const char* src = values + tid * max_vallen;
    char* dst = out_passwords + tid * GPU_MAXLEN;
    int len = val_lens[tid];
    for (int i = 0; i < len; ++i) dst[i] = src[i];
    dst[len] = '\0';
    out_lengths[tid] = len;
}

__global__ void GenerateKernel_Multi(
    const char* prefix, int prefix_len,
    const char* values, const int* val_lens,
    char* out_passwords, int* out_lengths,
    int n, int max_vallen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const char* val_src = values + tid * max_vallen;
    char* dst = out_passwords + tid * GPU_MAXLEN;
    int val_len = val_lens[tid];
    int total = prefix_len + val_len;
    for (int i = 0; i < prefix_len; ++i) dst[i] = prefix[i];
    for (int i = 0; i < val_len; ++i) dst[prefix_len + i] = val_src[i];
    dst[total] = '\0';
    out_lengths[tid] = total;
}

// ==== GPU cache for segment ordered_values ====================================
struct GPUSegmentCache {
    char* d_values = nullptr;
    int*  d_val_lens = nullptr;
    int   max_vallen = 0;
};

// Key: address of segment (since segments are in model.m.letters/digits/symbols vectors, stable)
static unordered_map<segment*, GPUSegmentCache> g_seg_cache;

// Persistent GPU output buffer (reused across Generate calls)
static char* d_out_buf = nullptr;
static int*  d_out_lens_buf = nullptr;
static int   out_buf_cap = 0;

static void EnsureOutBuf(int needed) {
    if (needed <= out_buf_cap) return;
    if (d_out_buf) cudaFree(d_out_buf);
    if (d_out_lens_buf) cudaFree(d_out_lens_buf);
    int alloc = needed + 65536;
    cudaMalloc(&d_out_buf, alloc * GPU_MAXLEN);
    cudaMalloc(&d_out_lens_buf, alloc * sizeof(int));
    out_buf_cap = alloc;
}

static GPUSegmentCache& GetOrUploadSegment(segment* a) {
    auto it = g_seg_cache.find(a);
    if (it != g_seg_cache.end()) return it->second;

    // Upload ordered_values to GPU
    GPUSegmentCache c;
    int n = a->ordered_values.size();
    c.max_vallen = 0;
    for (int i = 0; i < n; ++i)
        if ((int)a->ordered_values[i].length() > c.max_vallen)
            c.max_vallen = a->ordered_values[i].length();
    if (c.max_vallen > 55) c.max_vallen = 55;

    char* h_vals = new char[n * c.max_vallen];
    int*  h_lens = new int[n];
    memset(h_vals, 0, n * c.max_vallen);
    for (int i = 0; i < n; ++i) {
        int len = min((int)a->ordered_values[i].length(), c.max_vallen);
        memcpy(h_vals + i * c.max_vallen, a->ordered_values[i].c_str(), len);
        h_lens[i] = len;
    }
    cudaMalloc(&c.d_values, n * c.max_vallen);
    cudaMalloc(&c.d_val_lens, n * sizeof(int));
    cudaMemcpy(c.d_values, h_vals, n * c.max_vallen, cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_val_lens, h_lens, n * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_vals; delete[] h_lens;

    g_seg_cache[a] = c;
    return g_seg_cache[a];
}

// ==== GPU Generate wrappers (bulk D2H) ========================================
static void Generate_GPU_Single(segment* a, int total_items,
                                 vector<string>& guesses, int old_size)
{
    GPUSegmentCache& c = GetOrUploadSegment(a);
    EnsureOutBuf(total_items);

    int block = 256, grid = (total_items + block - 1) / block;
    GenerateKernel_Single<<<grid, block>>>(c.d_values, c.d_val_lens,
        d_out_buf, d_out_lens_buf, total_items, c.max_vallen);
    cudaDeviceSynchronize();

    char* h_out = new char[total_items * GPU_MAXLEN];
    int*  h_out_lens = new int[total_items];
    cudaMemcpy(h_out, d_out_buf, total_items * GPU_MAXLEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_lens, d_out_lens_buf, total_items * sizeof(int), cudaMemcpyDeviceToHost);

    guesses.resize(old_size + total_items);
    for (int i = 0; i < total_items; ++i)
        guesses[old_size + i] = string(h_out + i * GPU_MAXLEN, h_out_lens[i]);
    delete[] h_out; delete[] h_out_lens;
}

static void Generate_GPU_Multi(const string& prefix, segment* a, int total_items,
                                vector<string>& guesses, int old_size)
{
    GPUSegmentCache& c = GetOrUploadSegment(a);
    EnsureOutBuf(total_items);

    int prefix_len = prefix.length();
    char* d_prefix; cudaMalloc(&d_prefix, prefix_len);
    cudaMemcpy(d_prefix, prefix.c_str(), prefix_len, cudaMemcpyHostToDevice);

    int block = 256, grid = (total_items + block - 1) / block;
    GenerateKernel_Multi<<<grid, block>>>(d_prefix, prefix_len,
        c.d_values, c.d_val_lens, d_out_buf, d_out_lens_buf, total_items, c.max_vallen);
    cudaDeviceSynchronize();
    cudaFree(d_prefix);

    char* h_out = new char[total_items * GPU_MAXLEN];
    int*  h_out_lens = new int[total_items];
    cudaMemcpy(h_out, d_out_buf, total_items * GPU_MAXLEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_lens, d_out_lens_buf, total_items * sizeof(int), cudaMemcpyDeviceToHost);

    guesses.resize(old_size + total_items);
    for (int i = 0; i < total_items; ++i)
        guesses[old_size + i] = string(h_out + i * GPU_MAXLEN, h_out_lens[i]);
    delete[] h_out; delete[] h_out_lens;
}

// ==== PriorityQueue methods ===================================================
void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2) pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3) pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
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
                    priority.emplace(iter + 1, pt); break;
                }
            }
            if (iter == priority.end() - 1) { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob) { priority.emplace(iter, pt); break; }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
        curr_indices[i] += 1;
        if (curr_indices[i] < max_indices[i]) { pivot = i; res.emplace_back(*this); }
        curr_indices[i] -= 1;
    }
    pivot = init_pivot;
    return res;
}

// ==== Generate: GPU-accelerated with segment cache + reusable output buffer ===
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)      a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)      a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)      a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total_items = pt.max_indices[0];
        old_size = guesses.size();

        if (total_items > GPU_THRESHOLD) {
            Generate_GPU_Single(a, total_items, guesses, old_size);
        } else {
            guesses.resize(old_size + total_items);
            for (int i = 0; i < total_items; i += 1)
                guesses[old_size + i] = a->ordered_values[i];
        }
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)      prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)      prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)      prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        int total_items = pt.max_indices[pt.content.size() - 1];
        old_size = guesses.size();

        if (total_items > GPU_THRESHOLD) {
            Generate_GPU_Multi(prefix, a, total_items, guesses, old_size);
        } else {
            guesses.resize(old_size + total_items);
            for (int i = 0; i < total_items; i += 1)
                guesses[old_size + i] = prefix + a->ordered_values[i];
        }
    }
}