#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <cmath>

constexpr int kNum = 1 << 20;

typedef struct _VERTEX {
    float x, y, z, nx, ny, nz, u, v;
} Vertex_rec;

void Transform(Vertex_rec &v) {
    v.z *= 2.0f;
}

void Lighting(Vertex_rec &v) {
    v.nx = v.x * v.y;
}

void ProcessVertices_SeparateLoops(std::vector<Vertex_rec> &vertices) {
    for (int i = 0; i < kNum; i++) {
        Transform(vertices[i]);
    }
    for (int i = 0; i < kNum; i++) {
        Lighting(vertices[i]);
    }
}

void ProcessVertices_CombinedLoop(std::vector<Vertex_rec> &vertices) {
    for (int i = 0; i < kNum; i++) {
        Transform(vertices[i]);
        Lighting(vertices[i]);
    }
}

static void BM_ProcessVertices_SeparateLoops(benchmark::State& state) {
    std::vector<Vertex_rec> vertices(kNum);

    for (auto _ : state) {
        ProcessVertices_SeparateLoops(vertices);
        benchmark::DoNotOptimize(vertices);
    }
}
BENCHMARK(BM_ProcessVertices_SeparateLoops);

static void BM_ProcessVertices_CombinedLoop(benchmark::State& state) {
    std::vector<Vertex_rec> vertices(kNum);

    for (auto _ : state) {
        ProcessVertices_CombinedLoop(vertices);
        benchmark::DoNotOptimize(vertices);
    }
}
BENCHMARK(BM_ProcessVertices_CombinedLoop);

BENCHMARK_MAIN();