#include <benchmark/benchmark.h>

#include "bench_utils.hpp"

using namespace t81::core;

namespace {

T81Limb abs_limb(const T81Limb& limb) {
    if (limb.compare(T81Limb()) < 0) return -limb;
    return limb;
}

T81Limb naive_division(T81Limb numerator, T81Limb denominator) {
    if (denominator.compare(T81Limb()) == 0) return T81Limb();
    bool negative = numerator.compare(T81Limb()) < 0;
    if (denominator.compare(T81Limb()) < 0) negative = !negative;

    numerator = abs_limb(numerator);
    denominator = abs_limb(denominator);

    T81Limb quotient;
    while (numerator.compare(denominator) >= 0) {
        numerator = numerator - denominator;
        quotient = quotient + T81Limb::one();
    }
    return negative ? -quotient : quotient;
}

static void BM_T81_Division(benchmark::State& state) {
    T81Limb a = bench::random_limb();
    T81Limb b;
    do {
        b = bench::random_limb();
    } while (b.compare(T81Limb()) == 0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(naive_division(a, b));
    }
}

BENCHMARK(BM_T81_Division)->Unit(benchmark::kNanosecond);

} // namespace
