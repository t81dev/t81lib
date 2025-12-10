// bench/bench_bigint_montgomery.cpp — Benchmark for Montgomery reduction performance.

#include <random>

#include <benchmark/benchmark.h>

#include <t81/t81lib.hpp>
#include <t81/util/random.hpp>

namespace {

inline t81::core::bigint random_bigint(std::mt19937_64& rng, std::size_t limbs) {
    return t81::util::random_bigint(rng, limbs, true);
}

inline t81::core::bigint random_positive_bigint(std::mt19937_64& rng,
                                                 std::size_t limbs) {
    auto value = random_bigint(rng, limbs);
    if (value.is_negative()) {
        value = -value;
    }
    if (value.is_zero()) {
        value = t81::core::bigint::one();
    }
    return value;
}

} // namespace

static void bench_bigint_div_mod(benchmark::State& state) {
    std::mt19937_64 rng(0x5eedf00d + static_cast<int>(state.thread_index()));
    while (state.KeepRunning()) {
        const auto dividend = random_bigint(rng, 3);
        auto divisor = random_bigint(rng, 2);
        if (divisor.is_zero()) {
            divisor = t81::core::bigint::one();
        }
        const auto result = t81::core::bigint::div_mod(dividend, divisor);
        benchmark::DoNotOptimize(result);
    }
}

static void bench_montgomery_mul(benchmark::State& state) {
    std::mt19937_64 rng(0x5eedf00d + static_cast<int>(state.thread_index()) + 0x10);
    static const t81::core::bigint modulus = t81::core::bigint(197);
    static const t81::core::MontgomeryContext<t81::core::bigint> context(modulus);
    while (state.KeepRunning()) {
        const auto a = random_positive_bigint(rng, 2);
        const auto b = random_positive_bigint(rng, 2);
        const auto ma = context.to_montgomery(a);
        const auto mb = context.to_montgomery(b);
        const auto result = context.mul(ma, mb);
        benchmark::DoNotOptimize(result);
    }
}

static void bench_montgomery_pow(benchmark::State& state) {
    std::mt19937_64 rng(0x5eedf00d + static_cast<int>(state.thread_index()) + 0x20);
    static const t81::core::bigint modulus = t81::core::bigint(197);
    static const t81::core::MontgomeryContext<t81::core::bigint> context(modulus);
    const t81::core::bigint exponent = t81::core::bigint(5);
    while (state.KeepRunning()) {
        const auto base = random_positive_bigint(rng, 2);
        const auto mb = context.to_montgomery(base);
        const auto result = context.pow(mb, exponent);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(bench_bigint_div_mod);
BENCHMARK(bench_montgomery_mul);
BENCHMARK(bench_montgomery_pow);

BENCHMARK_MAIN();
