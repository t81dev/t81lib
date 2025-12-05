#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Subtract(benchmark::State& state) {
    T81Limb a = bench::random_limb();
    T81Limb b = bench::random_limb();
    for (auto _ : state) {
        benchmark::DoNotOptimize(a - b);
    }
}
BENCHMARK(BM_T81_Subtract);
