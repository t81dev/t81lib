#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Shift_Left_Trytes(benchmark::State& state) {
    auto value = bench::random_limb();
    for (auto _ : state) {
        auto shifted = bench::shift_left_trytes(value, 1);
        benchmark::DoNotOptimize(shifted);
    }
}
BENCHMARK(BM_T81_Shift_Left_Trytes);
