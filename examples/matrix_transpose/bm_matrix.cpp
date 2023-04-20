#include <benchmark/benchmark.h>
#include <new>
#include <immintrin.h>
#include <vector>

constexpr int kMatrixSize = 512;


bool IsTransposeCorrect(const std::vector<float>& src, const std::vector<float>& dst, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (std::abs(src[i * size + j] - dst[j * size + i]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}

void RandomFillMatrix(std::vector<float>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = rand() % 100;
        }
    }
}

void TransposeNaive(float* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            dst[j * size + i] = src[i * size + j];
        }
    }
}

void TransposeBlocked_8x8(float* src, float* dst, int size) {
    constexpr int kBlockSize = 8;
    for (int i = 0; i < size; i += kBlockSize) {
        for (int j = 0; j < size; j += kBlockSize) {
            for (int jj = j; jj < j + kBlockSize; ++jj) {
                for (int ii = i; ii < i + kBlockSize; ++ii) {
                    dst[jj * size + ii] = src[ii * size + jj];
                }
            }
        }
    }
}

#define src(i, j) src[(i) * col_size + (j)]
#define dst(i, j) dst[(i) * row_size + (j)]

void TransposeBlocked_8x8_SIMD(float* src, float* dst, int size) {
    constexpr int kBlockSize = 8;
    int row_size = size;
    int col_size = size;
    for (int i = 0; i < row_size; i += kBlockSize) {
        for (int j = 0; j < col_size; j += kBlockSize) {
            // Load 8x8 block
            __m256 r0 = _mm256_loadu_ps( &src(i, j));
            __m256 r1 = _mm256_loadu_ps( &src(i + 1, j));
            __m256 r2 = _mm256_loadu_ps( &src(i + 2, j));
            __m256 r3 = _mm256_loadu_ps( &src(i + 3, j));
            __m256 r4 = _mm256_loadu_ps( &src(i + 4, j));
            __m256 r5 = _mm256_loadu_ps( &src(i + 5, j));
            __m256 r6 = _mm256_loadu_ps( &src(i + 6, j));
            __m256 r7 = _mm256_loadu_ps( &src(i + 7, j));

            // Transpose 8x8 block
            // t0 = src(i, j) src(i+1, j) src(i, j+1) src(i+1, j+1) 
            //      src(i, j+4) src(i+1, j+4) src(i, j+5) src(i+1, j+5)
            // t2 = src(i+2, j) src(i+3, j) src(i+2, j+1) src(i+3, j+1) 
            //      src(i+2, j+4) src(i+3, j+4) src(i+2, j+5) src(i+3, j+5)
            // t4 = src(i+4, j) src(i+5, j) src(i+4, j+1) src(i+5, j+1) 
            //      src(i+4, j+4) src(i+5, j+4) src(i+4, j+5) src(i+5, j+5)
            // t6 = src(i+6, j) src(i+7, j) src(i+6, j+1) src(i+7, j+1) 
            //      src(i+6, j+4) src(i+7, j+4) src(i+6, j+5) src(i+7, j+5)
            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5);
            __m256 t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7);
            __m256 t7 = _mm256_unpackhi_ps(r6, r7);

            // _MM_SHUFFLE 从右到左看，
            // 第一个0,1 是从 t0 中选，第二个 0,1 是从 t2 中选
            // r0 = src(i, j) src(i+1, j) src(i+2, j) src(i+3, j)
            //      src(i, j+4) src(i+1, j+4) src(i+2, j+4) src(i+3, j+4)
            // r4 = src(i+4, j) src(i+5, j) src(i+6, j) src(i+7, j)
            //      src(i+4, j+4) src(i+5, j+4) src(i+6, j+4) src(i+7, j+4)
            r0 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(1,0,1,0));
            r1 = _mm256_shuffle_ps(t0,t2,_MM_SHUFFLE(3,2,3,2));
            r2 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(1,0,1,0));
            r3 = _mm256_shuffle_ps(t1,t3,_MM_SHUFFLE(3,2,3,2));
            r4 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(1,0,1,0));
            r5 = _mm256_shuffle_ps(t4,t6,_MM_SHUFFLE(3,2,3,2));
            r6 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(1,0,1,0));
            r7 = _mm256_shuffle_ps(t5,t7,_MM_SHUFFLE(3,2,3,2));

            // t0 = src(i, j) src(i+1, j) src(i+2, j) src(i+3, j)
            //      src(i+4, j) src(i+5, j) src(i+6, j) src(i+7, j)
            // t4 = src(i, j+4) src(i+1, j+4) src(i+2, j+4) src(i+3, j+4)
            //      src(i+4, j+4) src(i+5, j+4) src(i+6, j+4) src(i+7, j+4)
            t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
            t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
            t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
            t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
            t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
            t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
            t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
            t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

            // Store 4x4 block, 因为是行存储，store的时候仍是逐行进行
            _mm256_storeu_ps(&dst(j, i), t0);
            _mm256_storeu_ps(&dst(j + 1, i), t1);
            _mm256_storeu_ps(&dst(j + 2, i), t2);
            _mm256_storeu_ps(&dst(j + 3, i), t3);
            _mm256_storeu_ps(&dst(j + 4, i), t4);
            _mm256_storeu_ps(&dst(j + 5, i), t5);
            _mm256_storeu_ps(&dst(j + 6, i), t6);
            _mm256_storeu_ps(&dst(j + 7, i), t7);
        }
    }
}

#undef src
#undef dst

static void BM_TransposeNaive(benchmark::State& state) {
    std::vector<float> src(kMatrixSize * kMatrixSize);
    std::vector<float> dst(kMatrixSize * kMatrixSize);

    RandomFillMatrix(src, kMatrixSize);

    for (auto _ : state) {
        TransposeNaive(src.data(), dst.data(), kMatrixSize);
    }

    if (!IsTransposeCorrect(src, dst, kMatrixSize)) {
        state.SkipWithError("Transpose is incorrect");
    }
}

static void BM_TransposeBlocked_8x8(benchmark::State& state) {
    std::vector<float> src(kMatrixSize * kMatrixSize);
    std::vector<float> dst(kMatrixSize * kMatrixSize);

    RandomFillMatrix(src, kMatrixSize);

    for (auto _ : state) {
        TransposeBlocked_8x8(src.data(), dst.data(), kMatrixSize);
    }

    if (!IsTransposeCorrect(src, dst, kMatrixSize)) {
        state.SkipWithError("Transpose is incorrect");
    }
}

static void BM_TransposeBlocked_8x8_SIMD(benchmark::State& state) {
    std::vector<float> src(kMatrixSize * kMatrixSize);
    std::vector<float> dst(kMatrixSize * kMatrixSize);

    RandomFillMatrix(src, kMatrixSize);

    for (auto _ : state) {
        TransposeBlocked_8x8_SIMD(src.data(), dst.data(), kMatrixSize);
    }

    if (!IsTransposeCorrect(src, dst, kMatrixSize)) {
        state.SkipWithError("Transpose is incorrect");
    }
}


BENCHMARK(BM_TransposeNaive);
BENCHMARK(BM_TransposeBlocked_8x8);
BENCHMARK(BM_TransposeBlocked_8x8_SIMD);


BENCHMARK_MAIN();
