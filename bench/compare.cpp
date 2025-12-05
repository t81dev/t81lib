#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Compare(benchmark::State& state) {
    T81Limb lhs = bench::random_limb();
    T81Limb rhs = bench::random_limb();
    for (auto _ : state) {
        benchmark::DoNotOptimize(lhs.compare(rhs));
    }
}
BENCHMARK(BM_T81_Compare);
