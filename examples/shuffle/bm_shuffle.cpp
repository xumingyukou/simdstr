#include <random>
#include <cstring>
#include <tuple>
#include <benchmark/benchmark.h>

#include "shuffle.h"

std::string gen_ascii(size_t len) {
  std::string temp(len, '\0');
  for (size_t i = 0; i < len; i++) {
    temp[i] = (char)(std::rand() % 128);
  }
  return temp;
}

std::string gen_ascii(const char* chars, size_t len) {
  std::string temp(len, '\0');
  int nums = std::strlen(chars);
  for (size_t i = 0; i < len; i++) {
    temp[i] = chars[(std::rand() % nums)];
  }
  return temp;
}

// set the size of bench data < L1 cache (32 KB)
#define BENCH_SIZE(typ) (16 * 1024 / sizeof(typ))

template <typename Func>
class BM_SkipSpace {
public:
  BM_SkipSpace(Func&& func) : func_(func) {}

  void Fill() {
    this->json_ = gen_ascii(" \r\t\n", BENCH_SIZE(char));
    this->json_ += 'x';
  }

  void Test(benchmark::State& state) {
    auto got = func_(this->json_.data(), this->json_.size());
    auto expect = skipspace_naive(this->json_.data(), this->json_.size());
    if (got != expect) {
      state.SkipWithError("skipspace test failed");
    }
  }

  void Run() {
    func_(this->json_.data(), this->json_.size());
  }

private:
  std::string json_;
  Func& func_;
};

#undef BENCH_SIZE

template <typename BM, typename Func>
static void Bench(benchmark::State& state, Func&& func) {
  BM bm(func);
  bm.Fill();
  bm.Test(state);
  for (auto _ : state) {
    bm.Run();
  }
}

int main(int argc, char **argv) {
  using benchmark::RegisterBenchmark;
  benchmark::Initialize(&argc, argv);

#define ADD_BM(func, class)  do {     \
  benchmark::RegisterBenchmark(#func, \
  Bench<BM_##class<decltype(func)>,  \
        decltype(func)>, func);      \
  } while(0)

  ADD_BM(skipspace_naive, SkipSpace);
  ADD_BM(skipspace_use_cmpeq, SkipSpace);
  ADD_BM(skipspace_use_shuffle, SkipSpace);
  
#undef ADD_BM
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}