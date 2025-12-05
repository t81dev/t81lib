#include <benchmark/benchmark.h>

#if __has_include(<gmp.h>) && __has_include(<gmpxx.h>)
#include <gmp.h>
#include <gmpxx.h>
#define T81_HAVE_GMP256 1
#else
#define T81_HAVE_GMP256 0
#endif

#if T81_HAVE_GMP256
static void BM_GMP_256bit_Mul(benchmark::State& state) {
    mpz_t a, b, c;
    mpz_inits(a, b, c, nullptr);
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 42);
    mpz_urandomb(a, rs, 256);
    mpz_urandomb(b, rs, 256);
    for (auto _ : state) {
        mpz_mul(c, a, b);
        benchmark::DoNotOptimize(c);
    }
    mpz_clears(a, b, c, nullptr);
    gmp_randclear(rs);
}
BENCHMARK(BM_GMP_256bit_Mul);
#endif
