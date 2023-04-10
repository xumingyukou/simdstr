#include <random>
#include <cstring>
// #include <limits>
#include <iostream>
#include <benchmark/benchmark.h>

extern "C" {
    #include  "naivestr.h"
    #include  "simdstr.h"
}

std::string gen_ascii(size_t len) {
  std::string temp(len, '\0');
  for (size_t i = 0; i < len; i++) {
    temp[i] = (char)(std::rand() % 128);
  }
  return temp;
}

void fill_random(float *arr, int size, float range) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-range, range);
    for (int i = 0; i < size; i++) {
        arr[i] = dis(gen);
    }
}

std::string quote(const std::string& s) {
  std::string temp;
  temp += '"';
  for (size_t i = 0; i < s.size(); i++) {
    if (s[i] == '"') {
      temp += "\\\"";
    } else if (s[i] == '\\') {
      temp += "\\\\";
    } else {
      temp += s[i];
    }
  }
  temp += '"';
  return temp;
}

using sum_t      = float (*)(const float *arr, size_t len);
using memcmpeq_t = bool  (*)(const char *s1, const char *s2, size_t len);
using tolower_t  = char* (*)(char *dst, const char *src, size_t len);
using compact_t  = int   (*)(char *dst, const char *src, size_t len);
using qstrlen_t  = int   (*)(const char *src, size_t len);
using strstr_t   = char* (*)(const char *str, size_t n, const char *substr, size_t sn);

static void test_memcmpeq(benchmark::State& state, memcmpeq_t memcmpeq, const char *s1, const char *s2, size_t len) {
  if (memcmpeq(s1, s2, len) != memcmpeq_naive(s1, s2, len)) {
    state.SkipWithError("memcmpeq test failed");
  }
}

static void test_tolower(benchmark::State& state, tolower_t tolower, const char *s, size_t len) {
  char *buf1 = new char[len];
  char *buf2 = new char[len];

  char *out1 = tolower(buf1, s, len);
  char *out2 = tolower_naive(buf2, s, len);
  if (std::memcmp(out1, out2, len) != 0) {
    state.SkipWithError("tolower test failed");
  }

  delete[] buf1;
  delete[] buf2;
}

static void test_compact(benchmark::State& state, compact_t compact, const char *s, size_t len) {
  char *buf1 = new char[len];
  char *buf2 = new char[len];

  int out1 = compact(buf1, s, len);
  int out2 = compact_naive(buf2, s, len);
  if (out1 != out2 || std::memcmp(buf1, buf2, out1) != 0) {
    state.SkipWithError("compact test failed");
  }

  delete[] buf1;
  delete[] buf2;
}

static void test_qstrlen(benchmark::State& state, qstrlen_t qstrlen, const char *s, size_t len) {
  if (qstrlen(s, len) != qstrlen_naive(s, len)) {
    state.SkipWithError("unquote test failed");
  }
}

static void test_strstr(benchmark::State& state, strstr_t strstr, 
  const char *str, size_t n, const char *substr, size_t sn) {
  char *got = strstr(str, n, substr, sn);
  char *exp = strstr_naive(str, n, substr, sn);
  if (got != exp) {
    state.SkipWithError("strstr test failed");
  }
}

static void bm_sum(benchmark::State& state, sum_t fsum) {
  size_t len = 5120;
  float *arr = new float[len];
  fill_random(arr, len, 1.0);
  float sum = 0.0;
  for (auto _ : state) {
    sum = fsum(arr, len);
  }

  float diff = sum - sum_naive(arr, len);
  if (diff >= 1e-6) {
    // state.SkipWithError("sum test failed");
  }
  delete[] arr;
}

static void bm_memcmpeq(benchmark::State& state, memcmpeq_t memcmpeq) {
  std::string data1 = gen_ascii(10000);
  std::string data2 = data1;

  const char *s1 = data1.c_str();
  const char *s2 = data2.c_str();
  size_t len = data1.size();
  test_memcmpeq(state, memcmpeq, s1, s2, len);

  for (auto _ : state) {
    memcmpeq(s1, s2, len);
  }
}

static void bm_tolower(benchmark::State& state, tolower_t tolower) {
  std::string data = gen_ascii(10000);
  const char *s = data.c_str();
  size_t len = data.size();

  test_tolower(state, tolower, s, len);

  char *buf = new char[len];
  for (auto _ : state) {
    tolower(buf, s, len);
  }

  delete[] buf;
}

static void bm_compact(benchmark::State& state, compact_t compact) {
  std::string data = gen_ascii(10000);
  const char *s = data.c_str();
  size_t len = data.size();

  test_compact(state, compact, s, len);

  char *buf = new char[len];
  for (auto _ : state) {
    compact(buf, s, len);
  }

  delete[] buf;
}

static void bm_qstrlen(benchmark::State& state, qstrlen_t qstrlen) {
  std::string data = quote(gen_ascii(10000));
  const char *s = data.c_str();
  size_t len = data.size();

  test_qstrlen(state, qstrlen, s, len);

  for (auto _ : state) {
    qstrlen(s, len);
  }
}

static void bm_strstr(benchmark::State& state, strstr_t strstr) {
  size_t len = 10000;
  std::string substr = "hello";
  std::string data = gen_ascii(len) + substr;

  test_strstr(state, strstr, data.c_str(), data.size(), substr.c_str(), substr.size());

  char *buf = new char[len];
  for (auto _ : state) {
    strstr(data.c_str(), data.size(), substr.c_str(), substr.size());
  }

  delete[] buf;
}

int main(int argc, char **argv) {
  using benchmark::RegisterBenchmark;
  benchmark::Initialize(&argc, argv);

#define ADD_BM(func, arch)  do {                \
  benchmark::RegisterBenchmark(                 \
    (std::string(#func) + "_" + #arch).c_str(), \
    bm_##func, func##_##arch);                  \
  } while(0)

  ADD_BM(sum, naive);
  ADD_BM(sum, simd);
  ADD_BM(sum, simd_fast);
  ADD_BM(memcmpeq, naive);
  ADD_BM(memcmpeq, sse);
  ADD_BM(memcmpeq, sse4_2);
  ADD_BM(memcmpeq, sse4_2_fast);
  ADD_BM(memcmpeq, avx2);
#if __AVX512F__ &&  __AVX512BW__
  ADD_BM(memcmpeq, avx512);
#endif
  ADD_BM(memcmpeq, autovec);

  ADD_BM(tolower, naive);
  ADD_BM(compact, naive);
  ADD_BM(qstrlen, naive);
  ADD_BM(strstr, naive);

  // TODO: add more benchmarks
  
#undef ADD_BM
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
