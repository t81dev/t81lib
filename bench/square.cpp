#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Square(benchmark::State& state) {
    auto x = bench::random_limb();
    for (auto _ : state) {
        auto [lo, hi] = T81Limb::mul_wide(x, x);
        benchmark::DoNotOptimize(lo);
        benchmark::DoNotOptimize(hi);
    }
}
BENCHMARK(BM_T81_Square);
