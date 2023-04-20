#include <benchmark/benchmark.h>
#include <cstring>
#include <immintrin.h>
#include <vector>

#define ALIGNMENT 64

void memcpy_simd(char *dest, const char *src, size_t n) {
    while (n >= 32) {
        __m256i vec = _mm256_loadu_si256((__m256i *)src);
        _mm256_storeu_si256((__m256i *)dest, vec);
        src  += 32;
        dest += 32;
        n -= 32;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_stream(char *dest, const char *src, size_t n) {
    while (n >= 32) {
        __m256i vec = _mm256_stream_load_si256((__m256i *)src);
        _mm256_stream_si256((__m256i *)dest, vec);
        src  += 32;
        dest += 32;
        n -= 32;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_aligned(char *dest, const char *src, size_t n) {
    while (n >= 32) {
        __m256i vec = _mm256_load_si256((__m256i *)src);
        _mm256_store_si256((__m256i *)dest, vec);
        src  += 32;
        dest += 32;
        n -= 32;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_unroll2(char *dest, const char *src, size_t n) {
    while (n >= 64) {
        __m256i vec0 = _mm256_load_si256((__m256i *)src);
        __m256i vec1 = _mm256_load_si256((__m256i *)(src + 32));
        _mm256_store_si256((__m256i *)dest, vec0);
        _mm256_store_si256((__m256i *)(dest + 32), vec1);
        src  += 64;
        dest += 64;
        n -= 64;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_unroll4(char *dest, const char *src, size_t n) {
    while (n >= 128) {
        __m256i vec0 = _mm256_load_si256((__m256i *)src);
        __m256i vec1 = _mm256_load_si256((__m256i *)(src + 32));
        __m256i vec2 = _mm256_load_si256((__m256i *)(src + 64));
        __m256i vec3 = _mm256_load_si256((__m256i *)(src + 96));
        _mm256_store_si256((__m256i *)dest, vec0);
        _mm256_store_si256((__m256i *)(dest + 32), vec1);
        _mm256_store_si256((__m256i *)(dest + 64), vec2);
        _mm256_store_si256((__m256i *)(dest + 96), vec3);
        src  += 128;
        dest += 128;
        n -= 128;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_unroll8(char *dest, const char *src, size_t n) {
    while (n >= 32 * 8) {
#define LOAD(i) \
        __m256i vec##i = _mm256_load_si256((__m256i *)(src + 32 * i));
#define STORE(i) \
        _mm256_store_si256((__m256i *)(dest + 32 * i), vec##i);
        LOAD(0); LOAD(1); LOAD(2); LOAD(3); LOAD(4); LOAD(5); LOAD(6); LOAD(7);
        STORE(0); STORE(1); STORE(2); STORE(3); STORE(4); STORE(5); STORE(6); STORE(7);
#undef LOAD
#undef STORE
        src  += 32 * 8;
        dest += 32 * 8;
        n -= 32 * 8;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}

void memcpy_simd_unroll16(char *dest, const char *src, size_t n) {
    while (n >= 32 * 16) {
#define LOAD(i) \
        __m256i vec##i = _mm256_load_si256((__m256i *)(src + 32 * i));
#define STORE(i) \
        _mm256_store_si256((__m256i *)(dest + 32 * i), vec##i);
        LOAD(0); LOAD(1); LOAD(2); LOAD(3); LOAD(4); LOAD(5); LOAD(6); LOAD(7);
        LOAD(8); LOAD(9); LOAD(10); LOAD(11); LOAD(12); LOAD(13); LOAD(14); LOAD(15);
        STORE(0); STORE(1); STORE(2); STORE(3); STORE(4); STORE(5); STORE(6); STORE(7);
        STORE(8); STORE(9); STORE(10); STORE(11); STORE(12); STORE(13); STORE(14); STORE(15);
#undef LOAD
#undef STORE
        src  += 32 * 16;
        dest += 32 * 16;
        n -= 32 * 16;
    }
    while (n > 0) {
        *dest++ = *src++;
        --n;
    }
}


// 地址未对齐
void BM_Memcpy_Unaligned_SIMD(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    char *src_unaligned  = (char*)(src)  + 1;
    char *dest_unaligned = (char*)(dest) + 1;
    for (auto _ : state) {
        memcpy_simd(src_unaligned, dest_unaligned, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Unaligned_SIMD)->Arg(1<<20);


// 地址对齐
void BM_Memcpy_Aligned_SIMD(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD)->Arg(1<<20);

// 地址对齐
void BM_Memcpy_Aligned_SIMD_Stream(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_stream((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Stream)->Arg(1<<20);

// 地址对齐，使用需对齐的 SIMD 指令
void BM_Memcpy_Aligned_SIMD_Aligned(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_aligned((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Aligned)->Arg(1<<20);

void BM_Memcpy_Aligned_SIMD_Unroll_2(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_unroll2((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Unroll_2)->Arg(1<<20);

void BM_Memcpy_Aligned_SIMD_Unroll_4(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_unroll4((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Unroll_4)->Arg(1<<20);


void BM_Memcpy_Aligned_SIMD_Unroll_8(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_unroll8((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Unroll_8)->Arg(1<<20);

void BM_Memcpy_Aligned_SIMD_Unroll_16(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       memcpy_simd_unroll16((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Aligned_SIMD_Unroll_16)->Arg(1<<20);

void BM_Memcpy_Glibc(benchmark::State& state) {
    size_t size = state.range(0);
    void *src   = aligned_alloc(ALIGNMENT,  size + 1);
    void *dest  = aligned_alloc(ALIGNMENT,  size + 1);
    for (auto _ : state) {
       std::memcpy((char*)src, (char*)dest, size);
    }
    free(src);
    free(dest);
}
BENCHMARK(BM_Memcpy_Glibc)->Arg(1<<20);

BENCHMARK_MAIN();