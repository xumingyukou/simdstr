#include <benchmark/benchmark.h>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>

#define as_m128p(v) ((__m128i *)(v))
#define as_m128v(v) (*(const __m128i *)(v))
#define as_m128c(v) ((const __m128i *)(v))
#define as_uint64v(p) (*(uint64_t *)(p))
#define sonic_align(s) __attribute__((aligned(s)))

static const char kVec16xAsc0[16] sonic_align(16) = {
    '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0',
};

static const uint16_t kVec8x10[8] sonic_align(16) = {
    10, 10, 10, 10, 10, 10, 10, 10,
};

static const uint32_t kVec4x10k[4] sonic_align(16) = {
    10000,
    10000,
    10000,
    10000,
};

static const uint32_t kVec4xDiv10k[4] sonic_align(16) = {
    0xd1b71759,
    0xd1b71759,
    0xd1b71759,
    0xd1b71759,
};

static const uint16_t kVecDivPowers[8] sonic_align(16) = {
    0x20c5, 0x147b, 0x3334, 0x8000, 0x20c5, 0x147b, 0x3334, 0x8000,
};

static const uint16_t kVecShiftPowers[8] sonic_align(16) = {
    0x0080, 0x0800, 0x2000, 0x8000, 0x0080, 0x0800, 0x2000, 0x8000,
};

// static void print_m128i(__m128i v) {
//   uint8_t *p = (uint8_t *)&v;
//   for (int i = 0; i < 16; i++) {
//     printf("%02x ", p[i]);
//   }
//   printf("  ");
// }

// Convert num's each digit as packed 16-bit in a vector.
// num's digits as abcdefgh (high bits is 0 if not enough)
// If the num is abcdefgh, the converted vector is { a, b, c, d, e, f, g, h }
static inline __m128i Digits8toaSSE(uint32_t num) {
  // v00 = v128{0, 0, 0, 0, 0, abcdefgh}
  __m128i v00 = _mm_cvtsi32_si128(num);
  // 利用乘法+位移运算来逼近除法，epu32 * kVec4xDiv10k 和 右移45位的结果等价于除以10000
  // v02 = vector{abcd, 0, 0, 0, 0, 0}
  __m128i v01 = _mm_mul_epu32(v00, as_m128v(kVec4xDiv10k));
  __m128i v02 = _mm_srli_epi64(v01, 45);
  // v04 = vector{efgh, 0, 0, 0, 0, 0}
  __m128i v03 = _mm_mul_epu32(v02, as_m128v(kVec4x10k));
  __m128i v04 = _mm_sub_epi32(v00, v03);

  // v05 = vector{abcd, efgh, 0, 0, 0, 0, 0, 0}
  __m128i v05 = _mm_unpacklo_epi16(v02, v04);

  // v06 = vector{abcd * 4, efgh * 4, 0, 0, 0, 0, 0, 0}
  __m128i v06 = _mm_slli_epi64(v05, 2);
  // v07 = vector{abcd * 4, abcd * 4, efgh * 4, efgh * 4, 0, 0, 0, 0}
  __m128i v07 = _mm_unpacklo_epi16(v06, v06);
  // v08 = vector{abcd * 4, abcd * 4, abcd * 4, abcd * 4, efgh * 4, efgh * 4,
  // efgh * 4, efgh * 4}
  __m128i v08 = _mm_unpacklo_epi32(v07, v07);

  // v10 = v08 div 10^3, 10^2, 10^1, 10^0 = { a, ab, abc, abcd, e, ef, efg, efgh }
  __m128i v09 = _mm_mulhi_epu16(v08, as_m128v(kVecDivPowers));
  __m128i v10 = _mm_mulhi_epu16(v09, as_m128v(kVecShiftPowers));

  // v12 = { 0, a0, ab0, abc0, 0, e0, ef0, efg0 }
  __m128i v11 = _mm_mullo_epi16(v10, as_m128v(kVec8x10));
  __m128i v12 = _mm_slli_epi64(v11, 16);

  // v13 = { a, b, c, d, e, f, g, h }
  __m128i v13 = _mm_sub_epi16(v10, v12);
  return v13;
}


char *Utoa_16(uint64_t val, char *out) {
  /* remaining digits */
  __m128i v0 = Digits8toaSSE((uint32_t)(val / 100000000));
  __m128i v1 = Digits8toaSSE((uint32_t)(val % 100000000));
  __m128i v2 = _mm_packus_epi16(v0, v1);
  __m128i v3 = _mm_add_epi8(v2, as_m128v(kVec16xAsc0));

  /* convert to bytes, add '0' */
  _mm_storeu_si128(as_m128p(out), v3);
  return out + 16;
}

char *Utoa_Naive(uint64_t val, char *out) {
    uint32_t v = (uint32_t)(val / 100000000);
    uint32_t r = (uint32_t)(val % 100000000);
    out[0] = (v / 10000000) + '0';
    out[1] = (v / 1000000) % 10 + '0';
    out[2] = (v / 100000) % 10 + '0';
    out[3] = (v / 10000) % 10 + '0';
    out[4] = (v / 1000) % 10 + '0';
    out[5] = (v / 100) % 10 + '0';
    out[6] = (v / 10) % 10 + '0';
    out[7] = v % 10 + '0';
    out[8] = (r / 10000000) + '0';
    out[9] = (r / 1000000) % 10 + '0';
    out[10] = (r / 100000) % 10 + '0';
    out[11] = (r / 10000) % 10 + '0';
    out[12] = (r / 1000) % 10 + '0';
    out[13] = (r / 100) % 10 + '0';
    out[14] = (r / 10) % 10 + '0';
    out[15] = r % 10 + '0';
    return out + 16;
}

static void BM_Utoa_SIMD(benchmark::State &state) {
  char buf[16] = {0};
  uint64_t num = 1000000000000000 + (std::rand() % 1000000000000000);
  for (auto _ : state) {
    Utoa_16(num, buf);
    benchmark::DoNotOptimize(buf);
  }

  char buf2[16] = {0};
    Utoa_Naive(num, buf2);
    if (memcmp(buf, buf2, 16) != 0) {
        std::cout << "error" << std::endl;
    }
}
BENCHMARK(BM_Utoa_SIMD);


static void BM_Utoa_Naive(benchmark::State &state) {
  char buf[16];
  uint64_t num = 1000000000000000 + (std::rand() % 1000000000000000);
  for (auto _ : state) {
    Utoa_Naive(num, buf);
    benchmark::DoNotOptimize(buf);
  }
}
BENCHMARK(BM_Utoa_Naive);

BENCHMARK_MAIN();

