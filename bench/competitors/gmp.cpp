#if __has_include(<gmpxx.h>)
#include <gmpxx.h>
#define T81_COMPETITOR_HAVE_GMP 1
#else
#define T81_COMPETITOR_HAVE_GMP 0
#endif

namespace bench::competitors {

void gmp_placeholder() {
#if T81_COMPETITOR_HAVE_GMP
    mpz_class sample = 0;
    sample += 1;
#else
    (void)0;
#endif
}

} // namespace bench::competitors
