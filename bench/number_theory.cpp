#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

namespace {

static T81Limb random_nonzero() {
    T81Limb value;
    do {
        value = bench::random_limb();
    } while (value.compare(T81Limb()) == 0);
    return value;
}

static T81Limb random_positive() {
    T81Limb value = random_nonzero();
    return value.compare(T81Limb()) < 0 ? -value : value;
}

static void BM_T81_Gcd(benchmark::State& state) {
    T81Limb a = bench::random_limb();
    T81Limb b = bench::random_limb();
    for (auto _ : state) {
        benchmark::DoNotOptimize(a.gcd(b));
    }
}

static void BM_T81_InvMod(benchmark::State& state) {
    T81Limb mod = random_positive();
    T81Limb value;
    do {
        value = bench::random_limb();
    } while (value.compare(T81Limb()) == 0 || value.gcd(mod).compare(T81Limb::one()) != 0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(value.inv_mod(mod));
    }
}

static void BM_T81_PowMod(benchmark::State& state) {
    T81Limb base = bench::random_limb();
    T81Limb modulus = random_positive();
    T81Limb exponent = bench::random_limb();
    if (exponent.compare(T81Limb()) < 0) exponent = -exponent;

    for (auto _ : state) {
        benchmark::DoNotOptimize(T81Limb::pow_mod(base, exponent, modulus));
    }
}

BENCHMARK(BM_T81_Gcd)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_T81_InvMod)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_T81_PowMod)->Unit(benchmark::kNanosecond);

} // namespace
