#include <benchmark/benchmark.h>
#include <random>

#include "t81/core/T81Limb.hpp"

namespace {

std::mt19937 rng(42);
std::uniform_int_distribution<int> trit_dist(-13, 13);

void fill_random_t81(t81::core::T81Limb& value) {
    for (int i = 0; i < t81::core::T81Limb::TRYTES; ++i) {
        value.set_tryte(i, trit_dist(rng));
    }
}

} // namespace

static void BM_T81_Add(benchmark::State& state) {
    t81::core::T81Limb a, b;
    fill_random_t81(a);
    fill_random_t81(b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
    }
}
BENCHMARK(BM_T81_Add)->Unit(benchmark::kNanosecond);
