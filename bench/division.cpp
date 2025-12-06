#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

namespace {

static void BM_T81_Division(benchmark::State& state) {
    T81Limb a = bench::random_limb();
    T81Limb b;
    do {
        b = bench::random_limb();
    } while (b.compare(T81Limb()) == 0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(a / b);
    }
}

BENCHMARK(BM_T81_Division)->Unit(benchmark::kNanosecond);

} // namespace
