#include <benchmark/benchmark.h>
#include <random>

#if defined(T81BENCH_GMP_ENABLED)
#include <gmp.h>
#include <gmpxx.h>
#endif

#if defined(T81BENCH_BOOST_ENABLED)
#include <boost/multiprecision/cpp_int.hpp>
#endif

#if defined(T81BENCH_TTMATH_ENABLED)
#include <ttmath.h>
#endif

namespace {

std::mt19937_64& bench_rng() {
    static std::mt19937_64 rng(0xC0FFEE);
    return rng;
}

#if defined(T81BENCH_GMP_ENABLED)
void gmp_multiply_bits(benchmark::State& state, unsigned bits) {
    mpz_t a, b, c;
    mpz_inits(a, b, c, nullptr);
    gmp_randstate_t st;
    gmp_randinit_default(st);
    gmp_randseed_ui(st, 42);
    mpz_urandomb(a, st, bits);
    mpz_urandomb(b, st, bits);
    for (auto _ : state) {
        mpz_mul(c, a, b);
        benchmark::DoNotOptimize(c);
    }
    mpz_clears(a, b, c, nullptr);
    gmp_randclear(st);
}

static void BM_GMP_64bit(benchmark::State& state) { gmp_multiply_bits(state, 64); }
BENCHMARK(BM_GMP_64bit);

static void BM_GMP_128bit(benchmark::State& state) { gmp_multiply_bits(state, 128); }
BENCHMARK(BM_GMP_128bit);

static void BM_GMP_256bit_Mul(benchmark::State& state) { gmp_multiply_bits(state, 256); }
BENCHMARK(BM_GMP_256bit_Mul);
#endif

#if defined(T81BENCH_BOOST_ENABLED)
boost::multiprecision::cpp_int random_cpp_int(std::size_t bits) {
    boost::multiprecision::cpp_int value = 0;
    std::mt19937_64& rng = bench_rng();
    std::uniform_int_distribution<std::uint64_t> dist;
    std::size_t blocks = (bits + 63) / 64;
    for (std::size_t i = 0; i < blocks; ++i) {
        value <<= 64;
        value += dist(rng);
    }
    return value;
}

template <std::size_t Bits>
void boost_mul(benchmark::State& state) {
    auto a = random_cpp_int(Bits);
    auto b = random_cpp_int(Bits);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}

static void BM_Boost_128bit(benchmark::State& state) { boost_mul<128>(state); }
BENCHMARK(BM_Boost_128bit);

static void BM_Boost_256bit(benchmark::State& state) { boost_mul<256>(state); }
BENCHMARK(BM_Boost_256bit);
#endif

#if defined(T81BENCH_TTMATH_ENABLED)
using tt_uint128 = ttmath::UInt<4>;
using tt_uint256 = ttmath::UInt<8>;

static void BM_TTMath_128bit(benchmark::State& state) {
    tt_uint128 a = 0;
    tt_uint128 b = 0;
    a += 0x1234567890ABCDEFULL;
    b += 0xFEDCBA0987654321ULL;
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}
BENCHMARK(BM_TTMath_128bit);

static void BM_TTMath_256bit(benchmark::State& state) {
    tt_uint256 a = 0;
    tt_uint256 b = 0;
    a += tt_uint256(0x1111222233334444ULL);
    b += tt_uint256(0x5555666677778888ULL);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a * b);
    }
}
BENCHMARK(BM_TTMath_256bit);
#endif

} // namespace
