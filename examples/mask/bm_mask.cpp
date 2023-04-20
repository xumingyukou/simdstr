#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>
#include <limits>

constexpr int kMaxElement = 4096;
constexpr int kElementsPerIteration = 16;

void Branch_Naive(const std::vector<int16_t>& A, const std::vector<int16_t>& B,
                  const std::vector<int16_t>& D, const std::vector<int16_t>& E,
                  std::vector<int16_t>& C) {
    for (int i = 0; i < kMaxElement; i++) {
        if (A[i] > B[i]) {
            C[i] = D[i];
        } else {
            C[i] = E[i];
        }
    }
}

void Branchless_AVX2(const std::vector<int16_t>& A, const std::vector<int16_t>& B,
                     const std::vector<int16_t>& D, const std::vector<int16_t>& E,
                     std::vector<int16_t>& C) {
    for (int i = 0; i < kMaxElement; i += kElementsPerIteration) {
        __m256i a = _mm256_loadu_si256((__m256i*)&A[i]);
        __m256i b = _mm256_loadu_si256((__m256i*)&B[i]);
        __m256i d = _mm256_loadu_si256((__m256i*)&D[i]);
        __m256i e = _mm256_loadu_si256((__m256i*)&E[i]);

        __m256i greater_than = _mm256_cmpgt_epi16(a, b);
        __m256i result = _mm256_blendv_epi8(e, d, greater_than);

        _mm256_storeu_si256((__m256i*)&C[i], result);
    }
}

bool VectorsEqual(const std::vector<int16_t>& a, const std::vector<int16_t>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static void BM_Branch_Naive(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> dist(-1000, 1000);
    std::vector<int16_t> A(kMaxElement), B(kMaxElement), D(kMaxElement), E(kMaxElement), C(kMaxElement);

    for (auto& elem : A) elem = dist(gen);
    for (auto& elem : B) elem = dist(gen);
    for (auto& elem : D) elem = dist(gen);
    for (auto& elem : E) elem = dist(gen);

    for (auto _ : state) {
        Branch_Naive(A, B, D, E, C);
        benchmark::DoNotOptimize(C);
    }

    std::vector<int16_t> C2(kMaxElement);
    Branch_Naive(A, B, D, E, C2);
    if (!VectorsEqual(C, C2)) {
        state.SkipWithError("Branch_Naive test failed");
    }
}
BENCHMARK(BM_Branch_Naive);

static void BM_Branchless_AVX2(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> dist(-1000, 1000);
    std::vector<int16_t> A(kMaxElement), B(kMaxElement), D(kMaxElement), E(kMaxElement), C(kMaxElement);

    for (auto& elem : A) elem = dist(gen);
    for (auto& elem : B) elem = dist(gen);
    for (auto& elem : D) elem = dist(gen);
    for (auto& elem : E) elem = dist(gen);

    for (auto _ : state) {
        Branchless_AVX2(A, B, D, E, C);
        benchmark::DoNotOptimize(C);
    }

    std::vector<int16_t> C2(kMaxElement);
    Branch_Naive(A, B, D, E, C2);
    if (!VectorsEqual(C, C2)) {
        state.SkipWithError("Branchless_AVX2 test failed");
    }
}
BENCHMARK(BM_Branchless_AVX2);

BENCHMARK_MAIN();