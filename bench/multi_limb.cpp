#include <array>
#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

static void BM_T81_Mul_4_Limbs(benchmark::State& state) {
    auto a = bench::random_limbs<std::array<T81Limb, 4>>();
    auto b = bench::random_limbs<std::array<T81Limb, 4>>();
    for (auto _ : state) {
        std::array<T81Limb, 8> result{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                auto [lo, hi] = T81Limb::mul_wide(a[i], b[j]);
                result[i + j] = result[i + j] + lo;
                result[i + j + 1] = result[i + j + 1] + hi;
            }
        }
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_T81_Mul_4_Limbs);
