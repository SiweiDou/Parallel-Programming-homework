#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cstdint>
#include <string>

using namespace std;
typedef unsigned int bit32;

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

#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define MD5_FF(a, b, c, d, x, s, ac) \
    do { (a) += MD5_F((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_GG(a, b, c, d, x, s, ac) \
    do { (a) += MD5_G((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_HH(a, b, c, d, x, s, ac) \
    do { (a) += MD5_H((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)
#define MD5_II(a, b, c, d, x, s, ac) \
    do { (a) += MD5_I((b), (c), (d)) + (x) + (unsigned int)(ac); (a) = ROTATE_LEFT((a), (s)); (a) += (b); } while(0)

__global__ void MD5Kernel(const unsigned char* padded_data, bit32* results, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const unsigned char* block = padded_data + tid * 64;
    bit32 x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = (block[4 * i]) | (block[4 * i + 1] << 8) | (block[4 * i + 2] << 16) | (block[4 * i + 3] << 24);
    }
    bit32 a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
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
    out[0]=((a&0xff)<<24)|((a&0xff00)<<8)|((a&0xff0000)>>8)|((a&0xff000000)>>24);
    out[1]=((b&0xff)<<24)|((b&0xff00)<<8)|((b&0xff0000)>>8)|((b&0xff000000)>>24);
    out[2]=((c&0xff)<<24)|((c&0xff00)<<8)|((c&0xff0000)>>8)|((c&0xff000000)>>24);
    out[3]=((d&0xff)<<24)|((d&0xff00)<<8)|((d&0xff0000)>>8)|((d&0xff000000)>>24);
}

__global__ void MD5Kernel_Raw(const char* raw_chars, const int* lengths, bit32* results, int n, int max_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const char* my_str = raw_chars + tid * max_len;
    int len = lengths[tid];
    unsigned char padded[64] = {0};
    for (int i = 0; i < len && i < 55; ++i) padded[i] = my_str[i];
    if (len < 55) padded[len] = 0x80;
    unsigned long long bit_len = ((unsigned long long)len) * 8ULL;
    for (int k = 0; k < 8; ++k) padded[56 + k] = (bit_len >> (k * 8)) & 0xFF;
    bit32 x[16];
    for (int i = 0; i < 16; ++i) x[i] = (padded[4*i])|(padded[4*i+1]<<8)|(padded[4*i+2]<<16)|(padded[4*i+3]<<24);
    bit32 a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
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
    out[0]=((a&0xff)<<24)|((a&0xff00)<<8)|((a&0xff0000)>>8)|((a&0xff000000)>>24);
    out[1]=((b&0xff)<<24)|((b&0xff00)<<8)|((b&0xff0000)>>8)|((b&0xff000000)>>24);
    out[2]=((c&0xff)<<24)|((c&0xff00)<<8)|((c&0xff0000)>>8)|((c&0xff000000)>>24);
    out[3]=((d&0xff)<<24)|((d&0xff00)<<8)|((d&0xff0000)>>8)|((d&0xff000000)>>24);
}

int MD5HashBatch_GPU_Raw(const string* inputs, int n, bit32* results, float* kernel_ms = nullptr)
{
    if (n == 0) return 0;
    int max_len = 0;
    for (int i = 0; i < n; ++i) { int l = inputs[i].length(); if (l > max_len) max_len = l; }
    char* h_raw; int* h_lengths;
    cudaHostAlloc(&h_raw, n * max_len, cudaHostAllocDefault);
    cudaHostAlloc(&h_lengths, n * sizeof(int), cudaHostAllocDefault);
    memset(h_raw, 0, n * max_len);
    for (int i = 0; i < n; ++i) { h_lengths[i] = inputs[i].length(); memcpy(h_raw + i * max_len, inputs[i].c_str(), h_lengths[i]); }
    char* d_raw; int* d_lengths; bit32* d_results;
    cudaMalloc(&d_raw, n * max_len); cudaMalloc(&d_lengths, n * sizeof(int)); cudaMalloc(&d_results, n * 4 * sizeof(bit32));
    cudaMemcpy(d_raw, h_raw, n * max_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaFreeHost(h_raw); cudaFreeHost(h_lengths);
    int bs = 256, gs = (n + bs - 1) / bs;
    MD5Kernel_Raw<<<gs, bs>>>(d_raw, d_lengths, d_results, n, max_len);
    cudaDeviceSynchronize();
    cudaFree(d_raw); cudaFree(d_lengths);
    bit32* h_results;
    cudaHostAlloc(&h_results, n * 4 * sizeof(bit32), cudaHostAllocDefault);
    cudaMemcpy(h_results, d_results, n * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    for (int i = 0; i < n; ++i) {
        results[i*4+0]=h_results[i*4+0]; results[i*4+1]=h_results[i*4+1];
        results[i*4+2]=h_results[i*4+2]; results[i*4+3]=h_results[i*4+3];
    }
    cudaFreeHost(h_results);
    return n;
}

int MD5HashBatch_GPU(const string* inputs, int n, bit32* results, float* kernel_ms = nullptr) {
    return MD5HashBatch_GPU_Raw(inputs, n, results, kernel_ms);
}

// GPU v4: hash passwords already on GPU (0 H2D!)
int MD5HashBatch_GPU_Direct(const char* d_passwords, const int* d_lengths, int n, bit32* results) {
    if (n == 0) return 0;
    bit32* d_results;
    cudaMalloc(&d_results, n * 4 * sizeof(bit32));
    int bs = 256, gs = (n + bs - 1) / bs;
    MD5Kernel_Raw<<<gs, bs>>>(d_passwords, d_lengths, d_results, n, 64);
    cudaDeviceSynchronize();
    bit32* h_results;
    cudaHostAlloc(&h_results, n * 4 * sizeof(bit32), cudaHostAllocDefault);
    cudaMemcpy(h_results, d_results, n * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    for (int i = 0; i < n; ++i) {
        results[i*4+0]=h_results[i*4+0]; results[i*4+1]=h_results[i*4+1];
        results[i*4+2]=h_results[i*4+2]; results[i*4+3]=h_results[i*4+3];
    }
    cudaFreeHost(h_results);
    return n;
}