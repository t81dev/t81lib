#if __has_include(<ttmath/ttmath.h>)
#include <ttmath/ttmath.h>
#define T81_COMPETITOR_HAVE_TTMATH 1
#else
#define T81_COMPETITOR_HAVE_TTMATH 0
#endif

namespace bench::competitors {

void ttmath_placeholder() {
#if T81_COMPETITOR_HAVE_TTMATH
    ttmath::Big<1> value;
    value = 1;
#else
    (void)0;
#endif
}

} // namespace bench::competitors
