// bench/bench_limb_add.cpp — Benchmark for limb addition routines.

#include <random>

#include <benchmark/benchmark.h>

#include <t81/core/limb.hpp>
#include <t81/util/random.hpp>

namespace {

    inline t81::core::limb random_limb(std::mt19937_64 &rng) { return t81::util::random_limb(rng); }

} // namespace

static void bench_limb_add(benchmark::State &state) {
    std::mt19937_64 rng(0x5eedc0de + static_cast<int>(state.thread_index()));
    while (state.KeepRunning()) {
        const auto lhs = random_limb(rng);
        const auto rhs = random_limb(rng);
        const auto result = lhs + rhs;
        benchmark::DoNotOptimize(result);
    }
}

static void bench_limb_mul(benchmark::State &state) {
    std::mt19937_64 rng(0x5eedc0de + static_cast<int>(state.thread_index()) + 0x10);
    while (state.KeepRunning()) {
        const auto lhs = random_limb(rng);
        const auto rhs = random_limb(rng);
        const auto product = t81::core::limb::mul_wide(lhs, rhs);
        benchmark::DoNotOptimize(product.first);
    }
}

BENCHMARK(bench_limb_add);
BENCHMARK(bench_limb_mul);

BENCHMARK_MAIN();
