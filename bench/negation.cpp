#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Negate(benchmark::State& state) {
    T81Limb x;
    for (int i = 0; i < T81Limb::TRYTES; ++i) x.set_tryte(i, static_cast<int8_t>((i % 27) - 13));
    for (auto _ : state) {
        benchmark::DoNotOptimize(-x);
    }
}
BENCHMARK(BM_T81_Negate);
