#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_To_From_Trits(benchmark::State& state) {
    for (auto _ : state) {
        auto digits = bench::random_trits();
        auto limb = T81Limb::from_trits(digits);
        auto roundtrip = limb.to_trits();
        benchmark::DoNotOptimize(roundtrip);
    }
}
BENCHMARK(BM_T81_To_From_Trits);
