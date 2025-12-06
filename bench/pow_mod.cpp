#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

namespace {

static T81Limb legacy_pow_mod(T81Limb base, T81Limb exp, const T81Limb& mod) {
    T81Limb result = T81Limb::one();
    base = base % mod;
    auto two = T81Limb::from_int(2);
    while (!exp.is_zero()) {
        auto [quot, rem] = exp.div_mod(two);
        if (!rem.is_zero()) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp = quot;
    }
    return result;
}

static T81Limb random_nonzero_mod() {
    T81Limb value;
    do {
        value = bench::random_limb();
    } while (value.compare(T81Limb()) == 0);
    return value;
}

static void BM_PowMod_Legacy(benchmark::State& state) {
    T81Limb modulus = random_nonzero_mod();
    if (modulus.compare(T81Limb()) < 0) modulus = -modulus;
    T81Limb base = bench::random_limb() % modulus;
    T81Limb exponent = bench::random_limb();
    if (exponent.compare(T81Limb()) < 0) exponent = -exponent;

    for (auto _ : state) {
        benchmark::DoNotOptimize(legacy_pow_mod(base, exponent, modulus));
    }
}

static void BM_PowMod_Montgomery(benchmark::State& state) {
    T81Limb modulus = random_nonzero_mod();
    if (modulus.compare(T81Limb()) < 0) modulus = -modulus;
    T81Limb base = bench::random_limb() % modulus;
    T81Limb exponent = bench::random_limb();
    if (exponent.compare(T81Limb()) < 0) exponent = -exponent;

    for (auto _ : state) {
        benchmark::DoNotOptimize(T81Limb::pow_mod(base, exponent, modulus));
    }
}

BENCHMARK(BM_PowMod_Legacy)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_PowMod_Montgomery)->Unit(benchmark::kNanosecond);

} // namespace
