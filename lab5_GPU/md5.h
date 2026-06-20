#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>

using namespace std;

typedef unsigned char Byte;
typedef unsigned int bit32;

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

inline bit32 F(bit32 x, bit32 y, bit32 z) { return (x & y) | (~x & z); }
inline bit32 G(bit32 x, bit32 y, bit32 z) { return (x & z) | (y & ~z); }
inline bit32 H(bit32 x, bit32 y, bit32 z) { return x ^ y ^ z; }
inline bit32 I(bit32 x, bit32 y, bit32 z) { return y ^ (x | ~z); }
inline bit32 ROTATELEFT(bit32 num, int n) { return (num << n) | (num >> (32 - n)); }

inline void FF(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += F(b, c, d) + x + ac; a = ROTATELEFT(a, s); a += b;
}
inline void GG(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += G(b, c, d) + x + ac; a = ROTATELEFT(a, s); a += b;
}
inline void HH(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += H(b, c, d) + x + ac; a = ROTATELEFT(a, s); a += b;
}
inline void II(bit32 &a, bit32 b, bit32 c, bit32 d, bit32 x, int s, bit32 ac) {
    a += I(b, c, d) + x + ac; a = ROTATELEFT(a, s); a += b;
}

void MD5Hash(string input, bit32 *state);
void MD5HashBatch(const string input_arry[4], bit32 state_batch[4][4]);

// GPU-accelerated batch MD5 hash
// Returns number of passwords successfully hashed on GPU (<=55 chars each)
// Passwords >55 chars must be handled by caller separately with CPU fallback
int MD5HashBatch_GPU(const string* inputs, int n, bit32* results);

#endif