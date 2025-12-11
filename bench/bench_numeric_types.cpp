// bench/bench_numeric_types.cpp — Benchmarks for the high-level numeric helpers in t81.

#include <benchmark/benchmark.h>

#include <t81/t81lib.hpp>

namespace {

namespace core = t81::core;

static core::limb make_limb(int value) {
    return core::limb::from_value(value);
}

static void BM_FloatMultiply(benchmark::State& state) {
    const t81::Float lhs(make_limb(27), 5);
    const t81::Float rhs(make_limb(9), -2);
    for (auto _ : state) {
        auto product = lhs * rhs;
        benchmark::DoNotOptimize(product.mantissa());
        benchmark::DoNotOptimize(product.exponent());
    }
}
BENCHMARK(BM_FloatMultiply);

static void BM_RatioArithmetic(benchmark::State& state) {
    const t81::Ratio lhs(make_limb(63));
    const t81::Ratio rhs(make_limb(14));
    for (auto _ : state) {
        auto sum = lhs + rhs;
        auto ordering = lhs <=> rhs;
        benchmark::DoNotOptimize(sum.numerator());
        benchmark::DoNotOptimize(ordering);
    }
}
BENCHMARK(BM_RatioArithmetic);

static void BM_MontgomeryIntAdd(benchmark::State& state) {
    const t81::Modulus modulus(make_limb(101));
    const t81::MontgomeryInt left(modulus, make_limb(33));
    const t81::MontgomeryInt right(modulus, make_limb(58));
    for (auto _ : state) {
        auto result = left;
        result += right;
        benchmark::DoNotOptimize(result.to_limb());
    }
}
BENCHMARK(BM_MontgomeryIntAdd);

static void BM_MontgomeryIntMultiply(benchmark::State& state) {
    const t81::Modulus modulus(make_limb(101));
    const t81::MontgomeryInt left(modulus, make_limb(7));
    const t81::MontgomeryInt right(modulus, make_limb(13));
    for (auto _ : state) {
        auto result = left;
        result *= right;
        benchmark::DoNotOptimize(result.to_limb());
    }
}
BENCHMARK(BM_MontgomeryIntMultiply);

} // namespace

BENCHMARK_MAIN();
