#include <array>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>

#if __has_include(<gmpxx.h>)
#include <gmpxx.h>
#define T81_HAVE_GMP 1
#else
#define T81_HAVE_GMP 0
#endif

#include "t81/core/T81Limb.hpp"

namespace {

std::mt19937 rng(1337);
std::uniform_int_distribution<int> trit_dist(-1, 1);
std::uniform_int_distribution<int> tryte_dist(-13, 13);

template <std::size_t N>
std::array<int8_t, N> random_trits() {
    std::array<int8_t, N> arr{};
    for (std::size_t i = 0; i < N; ++i) {
        arr[i] = static_cast<int8_t>(trit_dist(rng));
    }
    return arr;
}

void fill_random_trytes(t81::core::T81Limb& limb) {
    for (int i = 0; i < t81::core::T81Limb::TRYTES; ++i) {
        limb.set_tryte(i, static_cast<int8_t>(tryte_dist(rng)));
    }
}

static void BM_T81_Mul(benchmark::State& state) {
    using namespace t81::core;
    T81Limb a, b;
    fill_random_trytes(a);
    fill_random_trytes(b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(T81Limb::booth_mul(a, b));
    }
}
BENCHMARK(BM_T81_Mul);

static void BM_T81_Mul_Karatsuba(benchmark::State& state) {
    using namespace t81::core;
    T81Limb a, b;
    fill_random_trytes(a);
    fill_random_trytes(b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(T81Limb::mul_wide(a, b));
    }
}
BENCHMARK(BM_T81_Mul_Karatsuba);

#if T81_HAVE_GMP
template <std::size_t N>
mpz_class mpz_from_trits(const std::array<int8_t, N>& digits) {
    mpz_class value = 0;
    for (int idx = static_cast<int>(N) - 1; idx >= 0; --idx) {
        value *= 3;
        value += digits[idx];
    }
    return value;
}

static void BM_GMP_Mul_192bit(benchmark::State& state) {
    auto digits_a = random_trits<192>();
    auto digits_b = random_trits<192>();
    auto mpz_a = mpz_from_trits(digits_a);
    auto mpz_b = mpz_from_trits(digits_b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(mpz_a * mpz_b);
    }
}
BENCHMARK(BM_GMP_Mul_192bit);
#endif

static void BM_Schoolbook_Ternary(benchmark::State& state) {
    auto digits_a = random_trits<192>();
    auto digits_b = random_trits<192>();
    std::array<int, 384> accum{};
    for (auto _ : state) {
        accum.fill(0);
        for (std::size_t i = 0; i < digits_a.size(); ++i) {
            for (std::size_t j = 0; j < digits_b.size(); ++j) {
                accum[i + j] += digits_a[i] * digits_b[j];
            }
        }
        int carry = 0;
        for (std::size_t idx = 0; idx < accum.size(); ++idx) {
            int value = accum[idx] + carry;
            int q = (value >= 0) ? (value + 1) / 3 : (value - 1) / 3;
            int digit = value - q * 3;
            if (digit > 1) { digit -= 3; q += 1; }
            if (digit < -1) { digit += 3; q -= 1; }
            accum[idx] = digit;
            carry = q;
        }
        benchmark::DoNotOptimize(accum);
    }
}
BENCHMARK(BM_Schoolbook_Ternary);

} // namespace
