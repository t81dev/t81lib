#include <array>
#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Mul_16_Limbs(benchmark::State& state) {
    auto a = bench::random_limbs<std::array<T81Limb, 16>>();
    auto b = bench::random_limbs<std::array<T81Limb, 16>>();
    for (auto _ : state) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                auto [lo, hi] = T81Limb::mul_wide(a[i], b[j]);
                benchmark::DoNotOptimize(lo);
                benchmark::DoNotOptimize(hi);
            }
        }
    }
}
BENCHMARK(BM_T81_Mul_16_Limbs);
