#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cstdint>
#include <string>

using namespace std;
typedef unsigned int bit32;

// MD5 每一轮使用固定的循环左移位数。S11-S44 对应 RFC 1321 中四轮 64 步的移位表。
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

// MD5 的四个非线性函数和循环左移。宏形式可减少 device 端函数调用开销。
#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// 四轮变换步骤宏：把非线性函数、消息字 x、轮常量 ac 和左移 s 合并到 a 上。
// 每个宏调用对应 MD5 标准算法中的一步。
#define MD5_FF(a, b, c, d, x, s, ac) \
    do { (a) += MD5_F((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_GG(a, b, c, d, x, s, ac) \
    do { (a) += MD5_G((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_HH(a, b, c, d, x, s, ac) \
    do { (a) += MD5_H((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_II(a, b, c, d, x, s, ac) \
    do { (a) += MD5_I((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)

// 输入已经在 CPU 端预填充为 n 个 64 字节 block 时使用的 kernel。
// 每个线程处理一个口令的单个 MD5 block，并输出 4 个 32 位状态字。
__global__ void MD5Kernel(const unsigned char* padded_data, bit32* results, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // grid 按 blockDim 向上取整，超过 n 的线程不参与计算。
    if (tid >= n) return;
    // 每个输入 block 固定 64 字节，即 MD5 单块压缩函数的消息长度。
    const unsigned char* block = padded_data + tid * 64;
    bit32 x[16];
    for (int i = 0; i < 16; ++i) {
        // MD5 以小端方式把 64 字节消息解释为 16 个 32 位字。
        x[i] = (block[4 * i]) | (block[4 * i + 1] << 8) | (block[4 * i + 2] << 16) | (block[4 * i + 3] << 24);
    }
    // MD5 初始 IV。这里的实现面向单 block 短口令，因此不需要多 block 链式累加。
    bit32 a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    // 第一轮 FF：处理 x[0..15]，强调按标准常量和移位顺序展开，避免循环带来的索引开销。
    MD5_FF(a,b,c,d,x[0],S11,0xd76aa478);MD5_FF(d,a,b,c,x[1],S12,0xe8c7b756);MD5_FF(c,d,a,b,x[2],S13,0x242070db);MD5_FF(b,c,d,a,x[3],S14,0xc1bdceee);
    MD5_FF(a,b,c,d,x[4],S11,0xf57c0faf);MD5_FF(d,a,b,c,x[5],S12,0x4787c62a);MD5_FF(c,d,a,b,x[6],S13,0xa8304613);MD5_FF(b,c,d,a,x[7],S14,0xfd469501);
    MD5_FF(a,b,c,d,x[8],S11,0x698098d8);MD5_FF(d,a,b,c,x[9],S12,0x8b44f7af);MD5_FF(c,d,a,b,x[10],S13,0xffff5bb1);MD5_FF(b,c,d,a,x[11],S14,0x895cd7be);
    MD5_FF(a,b,c,d,x[12],S11,0x6b901122);MD5_FF(d,a,b,c,x[13],S12,0xfd987193);MD5_FF(c,d,a,b,x[14],S13,0xa679438e);MD5_FF(b,c,d,a,x[15],S14,0x49b40821);
    // 第二轮 GG、第三轮 HH、第四轮 II 与标准 MD5 的 64 步完全对应。
    MD5_GG(a,b,c,d,x[1],S21,0xf61e2562);MD5_GG(d,a,b,c,x[6],S22,0xc040b340);MD5_GG(c,d,a,b,x[11],S23,0x265e5a51);MD5_GG(b,c,d,a,x[0],S24,0xe9b6c7aa);
    MD5_GG(a,b,c,d,x[5],S21,0xd62f105d);MD5_GG(d,a,b,c,x[10],S22,0x2441453);MD5_GG(c,d,a,b,x[15],S23,0xd8a1e681);MD5_GG(b,c,d,a,x[4],S24,0xe7d3fbc8);
    MD5_GG(a,b,c,d,x[9],S21,0x21e1cde6);MD5_GG(d,a,b,c,x[14],S22,0xc33707d6);MD5_GG(c,d,a,b,x[3],S23,0xf4d50d87);MD5_GG(b,c,d,a,x[8],S24,0x455a14ed);
    MD5_GG(a,b,c,d,x[13],S21,0xa9e3e905);MD5_GG(d,a,b,c,x[2],S22,0xfcefa3f8);MD5_GG(c,d,a,b,x[7],S23,0x676f02d9);MD5_GG(b,c,d,a,x[12],S24,0x8d2a4c8a);
    MD5_HH(a,b,c,d,x[5],S31,0xfffa3942);MD5_HH(d,a,b,c,x[8],S32,0x8771f681);MD5_HH(c,d,a,b,x[11],S33,0x6d9d6122);MD5_HH(b,c,d,a,x[14],S34,0xfde5380c);
    MD5_HH(a,b,c,d,x[1],S31,0xa4beea44);MD5_HH(d,a,b,c,x[4],S32,0x4bdecfa9);MD5_HH(c,d,a,b,x[7],S33,0xf6bb4b60);MD5_HH(b,c,d,a,x[10],S34,0xbebfbc70);
    MD5_HH(a,b,c,d,x[13],S31,0x289b7ec6);MD5_HH(d,a,b,c,x[0],S32,0xeaa127fa);MD5_HH(c,d,a,b,x[3],S33,0xd4ef3085);MD5_HH(b,c,d,a,x[6],S34,0x4881d05);
    MD5_HH(a,b,c,d,x[9],S31,0xd9d4d039);MD5_HH(d,a,b,c,x[12],S32,0xe6db99e5);MD5_HH(c,d,a,b,x[15],S33,0x1fa27cf8);MD5_HH(b,c,d,a,x[2],S34,0xc4ac5665);
    MD5_II(a,b,c,d,x[0],S41,0xf4292244);MD5_II(d,a,b,c,x[7],S42,0x432aff97);MD5_II(c,d,a,b,x[14],S43,0xab9423a7);MD5_II(b,c,d,a,x[5],S44,0xfc93a039);
    MD5_II(a,b,c,d,x[12],S41,0x655b59c3);MD5_II(d,a,b,c,x[3],S42,0x8f0ccc92);MD5_II(c,d,a,b,x[10],S43,0xffeff47d);MD5_II(b,c,d,a,x[1],S44,0x85845dd1);
    MD5_II(a,b,c,d,x[8],S41,0x6fa87e4f);MD5_II(d,a,b,c,x[15],S42,0xfe2ce6e0);MD5_II(c,d,a,b,x[6],S43,0xa3014314);MD5_II(b,c,d,a,x[13],S44,0x4e0811a1);
    MD5_II(a,b,c,d,x[4],S41,0xf7537e82);MD5_II(d,a,b,c,x[11],S42,0xbd3af235);MD5_II(c,d,a,b,x[2],S43,0x2ad7d2bb);MD5_II(b,c,d,a,x[9],S44,0xeb86d391);
    bit32* out = results + tid * 4;
    // 输出时做字节序转换，使 main.cpp 中的十六进制拼接结果与常见 MD5 字符串一致。
    out[0]=((a&0xff)<<24)|((a&0xff00)<<8)|((a&0xff0000)>>8)|((a&0xff000000)>>24);
    out[1]=((b&0xff)<<24)|((b&0xff00)<<8)|((b&0xff0000)>>8)|((b&0xff000000)>>24);
    out[2]=((c&0xff)<<24)|((c&0xff00)<<8)|((c&0xff0000)>>8)|((c&0xff000000)>>24);
    out[3]=((d&0xff)<<24)|((d&0xff00)<<8)|((d&0xff0000)>>8)|((d&0xff000000)>>24);
}

// Raw kernel 直接接收未 padding 的口令字符数组和长度数组。
// raw_chars 按 n * max_len 扁平化存储，每个线程在寄存器/局部数组中完成 padding。
__global__ void MD5Kernel_Raw(const char* raw_chars, const int* lengths, bit32* results, int n, int max_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 过滤向上取整产生的无效线程。
    if (tid >= n) return;
    // 第 tid 条口令位于 raw_chars 的第 tid 个定长槽位。
    const char* my_str = raw_chars + tid * max_len;
    int len = lengths[tid];
    unsigned char padded[64] = {0};
    // 当前 GPU 实现只处理长度不超过 55 字节的单 block MD5，有效载荷拷贝到 padded 前部。
    for (int i = 0; i < len && i < 55; ++i) padded[i] = my_str[i];
    // MD5 padding: 消息后追加 0x80，再补 0，最后 8 字节写入原始 bit 长度。
    if (len < 55) padded[len] = 0x80;
    unsigned long long bit_len = ((unsigned long long)len) * 8ULL;
    for (int k = 0; k < 8; ++k) padded[56 + k] = (bit_len >> (k * 8)) & 0xFF;
    bit32 x[16];
    // 将 padding 后的 64 字节小端展开为 16 个 32 位消息字。
    for (int i = 0; i < 16; ++i) x[i] = (padded[4*i])|(padded[4*i+1]<<8)|(padded[4*i+2]<<16)|(padded[4*i+3]<<24);
    bit32 a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
    // 下面 64 次宏展开是标准 MD5 压缩函数四轮计算；为了性能没有写成循环。
    MD5_FF(a,b,c,d,x[0],S11,0xd76aa478);MD5_FF(d,a,b,c,x[1],S12,0xe8c7b756);MD5_FF(c,d,a,b,x[2],S13,0x242070db);MD5_FF(b,c,d,a,x[3],S14,0xc1bdceee);
    MD5_FF(a,b,c,d,x[4],S11,0xf57c0faf);MD5_FF(d,a,b,c,x[5],S12,0x4787c62a);MD5_FF(c,d,a,b,x[6],S13,0xa8304613);MD5_FF(b,c,d,a,x[7],S14,0xfd469501);
    MD5_FF(a,b,c,d,x[8],S11,0x698098d8);MD5_FF(d,a,b,c,x[9],S12,0x8b44f7af);MD5_FF(c,d,a,b,x[10],S13,0xffff5bb1);MD5_FF(b,c,d,a,x[11],S14,0x895cd7be);
    MD5_FF(a,b,c,d,x[12],S11,0x6b901122);MD5_FF(d,a,b,c,x[13],S12,0xfd987193);MD5_FF(c,d,a,b,x[14],S13,0xa679438e);MD5_FF(b,c,d,a,x[15],S14,0x49b40821);
    MD5_GG(a,b,c,d,x[1],S21,0xf61e2562);MD5_GG(d,a,b,c,x[6],S22,0xc040b340);MD5_GG(c,d,a,b,x[11],S23,0x265e5a51);MD5_GG(b,c,d,a,x[0],S24,0xe9b6c7aa);
    MD5_GG(a,b,c,d,x[5],S21,0xd62f105d);MD5_GG(d,a,b,c,x[10],S22,0x2441453);MD5_GG(c,d,a,b,x[15],S23,0xd8a1e681);MD5_GG(b,c,d,a,x[4],S24,0xe7d3fbc8);
    MD5_GG(a,b,c,d,x[9],S21,0x21e1cde6);MD5_GG(d,a,b,c,x[14],S22,0xc33707d6);MD5_GG(c,d,a,b,x[3],S23,0xf4d50d87);MD5_GG(b,c,d,a,x[8],S24,0x455a14ed);
    MD5_GG(a,b,c,d,x[13],S21,0xa9e3e905);MD5_GG(d,a,b,c,x[2],S22,0xfcefa3f8);MD5_GG(c,d,a,b,x[7],S23,0x676f02d9);MD5_GG(b,c,d,a,x[12],S24,0x8d2a4c8a);
    MD5_HH(a,b,c,d,x[5],S31,0xfffa3942);MD5_HH(d,a,b,c,x[8],S32,0x8771f681);MD5_HH(c,d,a,b,x[11],S33,0x6d9d6122);MD5_HH(b,c,d,a,x[14],S34,0xfde5380c);
    MD5_HH(a,b,c,d,x[1],S31,0xa4beea44);MD5_HH(d,a,b,c,x[4],S32,0x4bdecfa9);MD5_HH(c,d,a,b,x[7],S33,0xf6bb4b60);MD5_HH(b,c,d,a,x[10],S34,0xbebfbc70);
    MD5_HH(a,b,c,d,x[13],S31,0x289b7ec6);MD5_HH(d,a,b,c,x[0],S32,0xeaa127fa);MD5_HH(c,d,a,b,x[3],S33,0xd4ef3085);MD5_HH(b,c,d,a,x[6],S34,0x4881d05);
    MD5_HH(a,b,c,d,x[9],S31,0xd9d4d039);MD5_HH(d,a,b,c,x[12],S32,0xe6db99e5);MD5_HH(c,d,a,b,x[15],S33,0x1fa27cf8);MD5_HH(b,c,d,a,x[2],S34,0xc4ac5665);
    MD5_II(a,b,c,d,x[0],S41,0xf4292244);MD5_II(d,a,b,c,x[7],S42,0x432aff97);MD5_II(c,d,a,b,x[14],S43,0xab9423a7);MD5_II(b,c,d,a,x[5],S44,0xfc93a039);
    MD5_II(a,b,c,d,x[12],S41,0x655b59c3);MD5_II(d,a,b,c,x[3],S42,0x8f0ccc92);MD5_II(c,d,a,b,x[10],S43,0xffeff47d);MD5_II(b,c,d,a,x[1],S44,0x85845dd1);
    MD5_II(a,b,c,d,x[8],S41,0x6fa87e4f);MD5_II(d,a,b,c,x[15],S42,0xfe2ce6e0);MD5_II(c,d,a,b,x[6],S43,0xa3014314);MD5_II(b,c,d,a,x[13],S44,0x4e0811a1);
    MD5_II(a,b,c,d,x[4],S41,0xf7537e82);MD5_II(d,a,b,c,x[11],S42,0xbd3af235);MD5_II(c,d,a,b,x[2],S43,0x2ad7d2bb);MD5_II(b,c,d,a,x[9],S44,0xeb86d391);
    bit32* out = results + tid * 4;
    // 与 CPU MD5Hash 的输出格式保持一致，便于直接比较摘要字符串。
    out[0]=((a&0xff)<<24)|((a&0xff00)<<8)|((a&0xff0000)>>8)|((a&0xff000000)>>24);
    out[1]=((b&0xff)<<24)|((b&0xff00)<<8)|((b&0xff0000)>>8)|((b&0xff000000)>>24);
    out[2]=((c&0xff)<<24)|((c&0xff00)<<8)|((c&0xff0000)>>8)|((c&0xff000000)>>24);
    out[3]=((d&0xff)<<24)|((d&0xff00)<<8)|((d&0xff0000)>>8)|((d&0xff000000)>>24);
}

// CPU 端批量接口：接收 string 数组，打包成定长二维字符数组后调用 Raw kernel。
// results 由调用者分配，大小必须至少为 n * 4 个 bit32。
int MD5HashBatch_GPU_Raw(const string* inputs, int n, bit32* results, float* kernel_ms = nullptr)
{
    // 空 batch 直接返回，避免 cudaMalloc(0) 在不同运行时上的行为差异。
    if (n == 0) return 0;
    int max_len = 0;
    // 找到当前 batch 最长口令，作为扁平化存储的槽位宽度。
    for (int i = 0; i < n; ++i) { int l = inputs[i].length(); if (l > max_len) max_len = l; }
    char* h_raw; int* h_lengths;
    // 使用 pinned host memory 提升 H2D/D2H 拷贝效率。
    cudaHostAlloc(&h_raw, n * max_len, cudaHostAllocDefault);
    cudaHostAlloc(&h_lengths, n * sizeof(int), cudaHostAllocDefault);
    memset(h_raw, 0, n * max_len);
    // 将 string 数组拷贝到定长槽位中，同时保存真实长度。
    for (int i = 0; i < n; ++i) { h_lengths[i] = inputs[i].length(); memcpy(h_raw + i * max_len, inputs[i].c_str(), h_lengths[i]); }
    char* d_raw; int* d_lengths; bit32* d_results;
    // 分配设备端输入、长度和输出缓冲区。
    cudaMalloc(&d_raw, n * max_len); cudaMalloc(&d_lengths, n * sizeof(int)); cudaMalloc(&d_results, n * 4 * sizeof(bit32));
    cudaMemcpy(d_raw, h_raw, n * max_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaFreeHost(h_raw); cudaFreeHost(h_lengths);
    int bs = 256, gs = (n + bs - 1) / bs;
    // 每个 CUDA 线程计算一个口令的 MD5。
    MD5Kernel_Raw<<<gs, bs>>>(d_raw, d_lengths, d_results, n, max_len);
    // 本接口是同步接口，返回前保证 results 已可用。
    cudaDeviceSynchronize();
    cudaFree(d_raw); cudaFree(d_lengths);
    bit32* h_results;
    cudaHostAlloc(&h_results, n * 4 * sizeof(bit32), cudaHostAllocDefault);
    // 将 GPU 结果拷回 pinned host memory，再写入调用者提供的普通数组。
    cudaMemcpy(h_results, d_results, n * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    for (int i = 0; i < n; ++i) {
        results[i*4+0]=h_results[i*4+0]; results[i*4+1]=h_results[i*4+1];
        results[i*4+2]=h_results[i*4+2]; results[i*4+3]=h_results[i*4+3];
    }
    cudaFreeHost(h_results);
    return n;
}

// 兼容旧接口的包装函数，当前统一转到 Raw 版本实现。
int MD5HashBatch_GPU(const string* inputs, int n, bit32* results, float* kernel_ms = nullptr) {
    return MD5HashBatch_GPU_Raw(inputs, n, results, kernel_ms);
}

// GPU Direct 路径：候选口令已经由 guessing.cu 在 GPU 上生成时使用，避免再次 H2D 输入拷贝。
// d_passwords/d_lengths 均为设备指针；results 仍然是 CPU 端输出数组。
int MD5HashBatch_GPU_Direct(const char* d_passwords, const int* d_lengths, int n, bit32* results) {
    // 空 batch 直接返回。
    if (n == 0) return 0;
    bit32* d_results;
    cudaMalloc(&d_results, n * 4 * sizeof(bit32));
    int bs = 256, gs = (n + bs - 1) / bs;
    // guessing.cu 的候选输出槽位固定为 64 字节，因此 max_len 传 64。
    MD5Kernel_Raw<<<gs, bs>>>(d_passwords, d_lengths, d_results, n, 64);
    cudaDeviceSynchronize();
    bit32* h_results;
    cudaHostAlloc(&h_results, n * 4 * sizeof(bit32), cudaHostAllocDefault);
    // Direct 只省掉输入 H2D，摘要结果仍需 D2H 供 CPU 端后续比较/统计。
    cudaMemcpy(h_results, d_results, n * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    for (int i = 0; i < n; ++i) {
        results[i*4+0]=h_results[i*4+0]; results[i*4+1]=h_results[i*4+1];
        results[i*4+2]=h_results[i*4+2]; results[i*4+3]=h_results[i*4+3];
    }
    cudaFreeHost(h_results);
    return n;
}