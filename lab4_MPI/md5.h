#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>   

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化
// 将原版的宏改为了内联函数，并实现了SIMD重载版本
inline bit32 F(bit32 x, bit32 y, bit32 z) {
    return (x & y) | (~x & z);
}
inline bit32 G(bit32 x, bit32 y, bit32 z) {
    return (x & z) | (y & ~z);
}
inline bit32 H(bit32 x, bit32 y, bit32 z) {
    return x ^ y ^ z;
}
inline bit32 I(bit32 x, bit32 y, bit32 z) {
    return y ^ (x | ~z);
}

inline uint32x4_t F(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(
        vandq_u32(x, y),
        vbicq_u32(z, x)   // ~x & z
    );
}
inline uint32x4_t G(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return vorrq_u32(
        vandq_u32(x, z),
        vbicq_u32(y, z)   // ~z & y
    );
}
inline uint32x4_t H(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    return veorq_u32(veorq_u32(x, y), z);
}
inline uint32x4_t I(uint32x4_t x, uint32x4_t y, uint32x4_t z) {
    uint32x4_t not_z = veorq_u32(z, vdupq_n_u32(0xFFFFFFFF));
    return veorq_u32(y, vorrq_u32(x, not_z));
}

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
inline bit32 ROTATELEFT(bit32 num, int n) {
    return (num << n) | (num >> (32 - n));
}

inline uint32x4_t ROTATELEFT(uint32x4_t num, int n) {
    return vsliq_n_u32(vshrq_n_u32(num, 32 - n), num, n);
}

inline void FF(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += F(b, c, d) + x + ac;
    a = ROTATELEFT(a, s);
    a += b;
}
inline void GG(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += G(b, c, d) + x + ac;
    a = ROTATELEFT(a, s);
    a += b;
}
inline void HH(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += H(b, c, d) + x + ac;
    a = ROTATELEFT(a, s);
    a += b;
}
inline void II(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += I(b, c, d) + x + ac;
    a = ROTATELEFT(a, s);
    a += b;
}

inline void FF(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32x4_t ac) {
    // a += F(b,c,d) + x + ac
    a = vaddq_u32(a, F(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, ac);
    // a = ROTATELEFT(a, s)
    a = ROTATELEFT(a, s);
    // a += b
    a = vaddq_u32(a, b);
}
inline void GG(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32x4_t ac) {
    a = vaddq_u32(a, G(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, ac);
    a = ROTATELEFT(a, s);
    a = vaddq_u32(a, b);
}
inline void HH(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32x4_t ac) {
    a = vaddq_u32(a, H(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, ac);
    a = ROTATELEFT(a, s);
    a = vaddq_u32(a, b);
}
inline void II(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32x4_t ac) {
    a = vaddq_u32(a, I(b, c, d));
    a = vaddq_u32(a, x);
    a = vaddq_u32(a, ac);
    a = ROTATELEFT(a, s);
    a = vaddq_u32(a, b);
}

void MD5Hash(string input, bit32 *state);
void MD5HashBatch(const string input_arry[4], bit32 state_batch[4][4]);

#endif