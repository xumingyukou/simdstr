#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#include "naivestr.h"
#include "simdstr.h"

float sum_simd(const float *arr, size_t len) {
    float ret = 0.0;
    __m256i sum =  _mm256_setzero_si256();
    const float *ap = arr;
    while (len >= 8) {
        __m256i block = _mm256_loadu_si256((__m256i *)(ap));
        sum = _mm256_add_ps(sum, block);
        ap  += 8;
        len -= 8;
    }

    // add the reduced float vector
    float temp[8];
    _mm256_storeu_si256((__m256i *)temp, sum);
    for (int j = 0; j < 8; j++) {
        ret += temp[j];
    }

    // add the trail floats of array
    while (len > 0) {
        ret += *ap;
        ap++, len--;
    }
    return ret;
}

float sum_simd_fast(const float *arr, size_t len) {
    float ret = 0.0;
    __m256i sum1 =  _mm256_setzero_si256();
    __m256i sum2 =  _mm256_setzero_si256();
    const float *ap = arr;
    while (len >= 16) {
        __m256i block1 = _mm256_loadu_si256((__m256i *)(ap));
        __m256i block2 = _mm256_loadu_si256((__m256i *)(ap + 8));
        sum1 = _mm256_add_ps(sum1, block1);
        sum2 = _mm256_add_ps(sum2, block2);
        ap  += 16;
        len -= 16;
    }

    // add the reduced float vector
    float temp[8];
    sum1 = _mm256_add_ps(sum1, sum2);
    _mm256_storeu_si256((__m256i *)temp, sum1);
    for (int j = 0; j < 8; j++) {
        ret += temp[j];
    }

    // add the trail floats of array
    while (len > 0) {
        ret += *ap;
        ap++, len--;
    }
    return ret;
}

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

// memcmpeq use SSE
bool memcmpeq_sse(const char *s1, const char *s2, size_t len) {
    // use SSE for 16-byte loop
    while (len >= 16) {
        __m128i  v1  = _mm_loadu_si128((__m128i *)s1);
        __m128i  v2  = _mm_loadu_si128((__m128i *)s2);
        __m128i  eq  = _mm_cmpeq_epi8(v1, v2);
        int16_t mask = _mm_movemask_epi8(eq);
        if (mask != -1) {
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

// memcmpeq use SSE 4.2, reference:
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ssetechs=SSE4_2
// https://en.wikipedia.org/wiki/SSE4
bool memcmpeq_sse4_2(const char *s1, const char *s2, size_t len) {
    // use SSE4.2 for 16-byte loop
    while (len >= 16) {
        __m128i  v1   = _mm_loadu_si128((__m128i *)s1);
        __m128i  v2   = _mm_loadu_si128((__m128i *)s2);
        int result = _mm_cmpestri(v1, 16, v2, 16, _SIDD_CMP_EQUAL_ORDERED);
        if (result != 0) {
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

bool memcmpeq_sse4_2_fast(const char *s1, const char *s2, size_t len) {
    // use SSE4.2 for 16-byte loop
    while (len >= 16) {
        __m128i  v1   = _mm_loadu_si128((__m128i *)s1);
        __m128i  v2   = _mm_loadu_si128((__m128i *)s2);
        int result = _mm_cmpistri(v1, v2,_SIDD_CMP_EQUAL_ORDERED);
        if (result != 0) {
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

// memcmpeq use AVX2
bool memcmpeq_avx2(const char *s1, const char *s2, size_t len) {
    // use AVX2 for 32-byte loop
    while (len >= 32) {
        __m256i  v1 = _mm256_loadu_si256((__m256i *)s1);
        __m256i  v2 = _mm256_loadu_si256((__m256i *)s2);
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
    return memcmpeq_sse(s1, s2, len);
}

#if __AVX512F__ &&  __AVX512BW__
// memcmpeq use AVX512
bool memcmpeq_avx512(const char *s1, const char *s2, size_t len) {
    // use AVX512 for 64-byte loop
    while (len >= 64) {
        __m512i   v1 = _mm512_loadu_si512((__m512i *)s1);
        __m512i   v2 = _mm512_loadu_si512((__m512i *)s2);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(v1, v2);
        if (~mask) {
            return false;
        }
        s1  += 64;
        s2  += 64;
        len -= 64;
    };
    // deal with trailing bytes
    return memcmpeq_avx2(s1, s2, len);
}
#endif

// TODO: implememt follow functions
// char* tolower_simd(char *dst, const char *src, size_t len);
// int   compact_simd(char *dst, const char *src, size_t len);
// int   qstrlen_simd(const char *src, size_t len);
// char* strstr_simd(const char *str, size_t n, const char *substr, size_t sn);