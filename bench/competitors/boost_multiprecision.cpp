#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#include <boost/multiprecision/cpp_int.hpp>
#define T81_COMPETITOR_HAVE_BOOST 1
#else
#define T81_COMPETITOR_HAVE_BOOST 0
#endif

namespace bench::competitors {

void boost_placeholder() {
#if T81_COMPETITOR_HAVE_BOOST
    boost::multiprecision::cpp_int value = 0;
    value += 1;
#else
    (void)0;
#endif
}

} // namespace bench::competitors
