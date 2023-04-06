#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#include "naivestr.h"
#include "simdstr.h"

bool memcmpeq_autovec(const char *s1, const char *s2, size_t len) {
    // Cannot auto vectorization.
    // while (len > 0 && *s1++ == *s2++) len--;
    // return len == 0;

    // auto vectorization.
    bool ret = true;
    for (size_t i = 0; i < len; i++) {
        ret &= (s1[i] == s2[i]);
    }
    return ret;
}

bool memcmpeq_sse(const char *s1, const char *s2, size_t len) {
    // use SSE for 16-byte loop
    while (len >= 16) {
        __m128i  v1   = _mm_loadu_si128((const void *)s1);
        __m128i  v2   = _mm_loadu_si128((const void *)s2);
        __m128i  eq   = _mm_cmpeq_epi8(v1, v2);
        uint16_t mask = (uint16_t)(_mm_movemask_epi8(eq));
        if (mask != 0xFFFFu) {
            return false;
        }
        s1  += 16;
        s2  += 16;
        len -= 16;
    };
    // deal with trailing bytes
    while (len > 0 && *s1++ == *s2++) len--;
    return len == 0;
}

bool memcmpeq_avx2(const char *s1, const char *s2, size_t len) {
    // use AVX2 for 32-byte loop
    while (len >= 32) {
        __m256i  v1 = _mm256_loadu_si256((const void *)s1);
        __m256i  v2 = _mm256_loadu_si256((const void *)s2);
        __m256i  eq = _mm256_cmpeq_epi8(v1, v2);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);
        if (mask != 0xFFFFFFFFu) {
            return false;
        }
        s1  += 32;
        s2  += 32;
        len -= 32;
    };
    // deal with trailing bytes
    return memcmpeq_sse(s1, s2,len);
}

#if __AVX512F__ &&  __AVX512BW__
bool memcmpeq_avx512(const char *s1, const char *s2, size_t len) {
    // use AVX512 for 64-byte loop
    while (len >= 64) {
        __m512i   v1 = _mm512_loadu_si512((const void *)s1);
        __m512i   v2 = _mm512_loadu_si512((const void *)s2);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if (~mask) {
            return false;
        }
        s1  += 64;
        s2  += 64;
        len -= 64;
    };
    // deal with trailing bytes
    return memcmpeq_avx2(s1, s2,len);
}
#endif

// TODO: implememt follow functions
// char* tolower_simd(char *dst, const char *src, size_t len);
// int   compact_simd(char *dst, const char *src, size_t len);
// int   qstrlen_simd(const char *src, size_t len);
// char* strstr_simd(const char *str, size_t n, const char *substr, size_t sn);