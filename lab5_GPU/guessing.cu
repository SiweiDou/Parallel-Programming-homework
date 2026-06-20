#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <unordered_map>
using namespace std;

#define GPU_MAXLEN 64
#define GPU_THRESHOLD 4096
#define BATCH_FLUSH_SIZE 131072   // flush when accumulated total_items exceeds this

// ==== GPU Generate kernels (per-item, used by batch) ==========================
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

// ==== Segment cache (one-time H2D per segment) ================================
struct GPUSegmentCache { char* d_values = nullptr; int* d_val_lens = nullptr; int max_vallen = 0; };
static unordered_map<segment*, GPUSegmentCache> g_seg_cache;

// ==== Multi-PT batch accumulator =============================================
struct PTBatch {
    string prefix;          // "" for single-segment
    segment* a;             // last segment
    int total_items;
    int guesses_base;       // where in guesses vector output starts
};
static vector<PTBatch> g_pt_batch;
static int g_batch_total = 0;  // sum of total_items in batch

// Ensure all segments in batch are uploaded to GPU
static void PreUploadSegments() {
    for (auto& e : g_pt_batch) {
        GetOrUploadSegment(e.a);
    }
}

static GPUSegmentCache& GetOrUploadSegment(segment* a) {
    auto it = g_seg_cache.find(a);
    if (it != g_seg_cache.end()) return it->second;
    GPUSegmentCache c;
    int n = a->ordered_values.size();
    for (int i = 0; i < n; ++i) if ((int)a->ordered_values[i].length() > c.max_vallen) c.max_vallen = a->ordered_values[i].length();
    if (c.max_vallen > 55) c.max_vallen = 55;
    char* h_vals = new char[n * c.max_vallen]; int* h_lens = new int[n];
    memset(h_vals, 0, n * c.max_vallen);
    for (int i = 0; i < n; ++i) { int len = min((int)a->ordered_values[i].length(), c.max_vallen); memcpy(h_vals + i * c.max_vallen, a->ordered_values[i].c_str(), len); h_lens[i] = len; }
    cudaMalloc(&c.d_values, n * c.max_vallen); cudaMalloc(&c.d_val_lens, n * sizeof(int));
    cudaMemcpy(c.d_values, h_vals, n * c.max_vallen, cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_val_lens, h_lens, n * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_vals; delete[] h_lens;
    g_seg_cache[a] = c;
    return g_seg_cache[a];
}

// GPU output buffer (reused)
static char* d_out = nullptr; static int* d_out_lens = nullptr; static int out_cap = 0;
static void EnsureOutBuf(int n) {
    if (n <= out_cap) return;
    if (d_out) { cudaFree(d_out); cudaFree(d_out_lens); }
    out_cap = n + 65536;
    cudaMalloc(&d_out, out_cap * GPU_MAXLEN);
    cudaMalloc(&d_out_lens, out_cap * sizeof(int));
}

// Bulk D2H: copy GPU output → CPU guesses at specified offsets
static void BulkD2H_MultiPT(vector<string>& guesses) {
    int total = g_batch_total;
    if (total == 0) return;
    char* h_out = new char[total * GPU_MAXLEN];
    int*  h_lens = new int[total];
    cudaMemcpy(h_out, d_out, total * GPU_MAXLEN, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lens, d_out_lens, total * sizeof(int), cudaMemcpyDeviceToHost);

    int gpu_offset = 0;
    for (auto& e : g_pt_batch) {
        for (int i = 0; i < e.total_items; ++i) {
            guesses[e.guesses_base + i] = string(h_out + (gpu_offset + i) * GPU_MAXLEN, h_lens[gpu_offset + i]);
        }
        gpu_offset += e.total_items;
    }
    delete[] h_out; delete[] h_lens;
}

// Flush accumulated PT batch to GPU
void FlushGPUBatch(vector<string>& guesses) {
    if (g_pt_batch.empty()) return;

    PreUploadSegments();
    int total = g_batch_total;
    EnsureOutBuf(total);

    int gpu_offset = 0;
    for (auto& e : g_pt_batch) {
        GPUSegmentCache& c = g_seg_cache[e.a];
        int block = 256, grid = (e.total_items + block - 1) / block;

        if (e.prefix.empty()) {
            // Single-segment PT
            GenerateKernel_Single<<<grid, block>>>(
                c.d_values, c.d_val_lens,
                d_out + gpu_offset * GPU_MAXLEN,
                d_out_lens + gpu_offset,
                e.total_items, c.max_vallen);
        } else {
            // Multi-segment PT
            int plen = e.prefix.length();
            char* d_prefix; cudaMalloc(&d_prefix, plen);
            cudaMemcpy(d_prefix, e.prefix.c_str(), plen, cudaMemcpyHostToDevice);
            GenerateKernel_Multi<<<grid, block>>>(
                d_prefix, plen,
                c.d_values, c.d_val_lens,
                d_out + gpu_offset * GPU_MAXLEN,
                d_out_lens + gpu_offset,
                e.total_items, c.max_vallen);
            cudaFree(d_prefix);
        }
        gpu_offset += e.total_items;
    }
    cudaDeviceSynchronize();  // single sync for entire batch

    // Single bulk D2H
    BulkD2H_MultiPT(guesses);

    g_pt_batch.clear();
    g_batch_total = 0;
}

// Expose for main.cpp
extern "C" void FlushGPUBatch_C() {
    // Will be called with PriorityQueue's guesses
    // declared in PCGF.h
}

// ==== PriorityQueue methods ===================================================
void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob; int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) { pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx]; pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq; }
        else if (pt.content[index].type == 2) { pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx]; pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq; }
        else if (pt.content[index].type == 3) { pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx]; pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq; }
        index += 1;
    }
}
void PriorityQueue::init() {
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2) pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3) pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm; CalProb(pt); priority.emplace_back(pt);
    }
}
void PriorityQueue::PopNext() {
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) { if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) { priority.emplace(iter + 1, pt); break; } }
            if (iter == priority.end() - 1) { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob) { priority.emplace(iter, pt); break; }
        }
    }
    priority.erase(priority.begin());
}
vector<PT> PT::NewPTs() {
    vector<PT> res; if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < curr_indices.size() - 1; i += 1) { curr_indices[i] += 1; if (curr_indices[i] < max_indices[i]) { pivot = i; res.emplace_back(*this); } curr_indices[i] -= 1; }
    pivot = init_pivot; return res;
}

// ==== Generate: CPU (small) or batch-queue GPU (large) ========================
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        int n = pt.max_indices[0]; old_size = guesses.size();
        if (n > GPU_THRESHOLD) {
            guesses.resize(old_size + n);
            PTBatch e; e.prefix = ""; e.a = a; e.total_items = n; e.guesses_base = old_size;
            g_pt_batch.push_back(e); g_batch_total += n;
            if (g_batch_total >= BATCH_FLUSH_SIZE) FlushGPUBatch(guesses);
        } else { guesses.resize(old_size + n); for (int i = 0; i < n; ++i) guesses[old_size + i] = a->ordered_values[i]; }
    } else {
        string pre; int s = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[s].type == 1) pre += m.letters[m.FindLetter(pt.content[s])].ordered_values[idx];
            if (pt.content[s].type == 2) pre += m.digits[m.FindDigit(pt.content[s])].ordered_values[idx];
            if (pt.content[s].type == 3) pre += m.symbols[m.FindSymbol(pt.content[s])].ordered_values[idx];
            s++; if (s == pt.content.size() - 1) break;
        }
        segment *a;
        if (pt.content[pt.content.size()-1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size()-1])];
        if (pt.content[pt.content.size()-1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size()-1])];
        if (pt.content[pt.content.size()-1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size()-1])];
        int n = pt.max_indices[pt.content.size()-1]; old_size = guesses.size();
        if (n > GPU_THRESHOLD) {
            guesses.resize(old_size + n);
            PTBatch e; e.prefix = pre; e.a = a; e.total_items = n; e.guesses_base = old_size;
            g_pt_batch.push_back(e); g_batch_total += n;
            if (g_batch_total >= BATCH_FLUSH_SIZE) FlushGPUBatch(guesses);
        } else { guesses.resize(old_size + n); for (int i = 0; i < n; ++i) guesses[old_size + i] = pre + a->ordered_values[i]; }
    }
}