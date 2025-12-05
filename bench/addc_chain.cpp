#include <array>
#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Addc_Chain(benchmark::State& state) {
    std::array<T81Limb, 8> limbs = bench::random_limbs<std::array<T81Limb, 8>>();
    for (auto _ : state) {
        T81Limb acc = limbs[0];
        for (size_t idx = 1; idx < limbs.size(); ++idx) {
            auto [sum, carry] = acc.addc(limbs[idx]);
            benchmark::DoNotOptimize(carry);
            acc = sum;
        }
        benchmark::DoNotOptimize(acc);
    }
}
BENCHMARK(BM_T81_Addc_Chain);
