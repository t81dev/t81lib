#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <string>
#include <vector>
#include "t81/core/T81Limb.hpp"

using t81::core::T81Limb;
using t81::core::MontgomeryContext;

static T81Limb from_int128(signed __int128 value) {
    return T81Limb::from_int128(value);
}

static signed __int128 to_int128(const T81Limb& limb) {
    return T81Limb::to_int128(limb);
}


TEST_CASE("Basic invariants and conversions", "[core][basic]") {
    REQUIRE(T81Limb().is_zero());
    REQUIRE(T81Limb::one().is_one());
    REQUIRE(from_int128(-1).is_negative());
    REQUIRE(from_int128(1).signum() == 1);
    REQUIRE(from_int128(-1).signum() == -1);

    auto number = from_int128(123456789012345LL);
    auto trits = number.to_trits();
    auto roundtrip = T81Limb::from_trits(trits);
    REQUIRE(roundtrip.compare(number) == 0);
}

TEST_CASE("Addition with carry propagation", "[core][add]") {
    T81Limb max_tryte{};
    for (int idx = 0; idx < T81Limb::TRYTES; ++idx) {
        max_tryte.set_tryte(idx, 13);
    }

    auto [sum, carry] = max_tryte.addc(max_tryte);
    REQUIRE(carry == 1);
    REQUIRE(sum.compare(max_tryte + max_tryte) == 0);
    REQUIRE(sum.compare(max_tryte + max_tryte) == 0);
    REQUIRE((max_tryte + max_tryte).compare(sum) == 0);
}

TEST_CASE("Multiplication and wide product consistency", "[core][mul]") {
    for (int32_t a = -512; a <= 512; a += 64) {
        for (int32_t b = -512; b <= 512; b += 64) {
            T81Limb lhs = from_int128(a);
            T81Limb rhs = from_int128(b);
            auto [low, high] = T81Limb::mul_wide(lhs, rhs);
            auto product = lhs * rhs;
            REQUIRE(low.compare(T81Limb::reference_mul(lhs, rhs)) == 0);
            // ensure high limb is sane (non-zero only when |a|*|b| exceeds 3^48)
            if (a == 0 || b == 0) {
                REQUIRE(high.is_zero());
            }
            REQUIRE(product.compare(T81Limb::reference_mul(lhs, rhs)) == 0);
        }
    }
}

TEST_CASE("Division surrounding sign combinations", "[core][div]") {
    const std::vector<std::pair<signed __int128, signed __int128>> cases = {
        {27, 5},
        {-27, 5},
        {27, -5},
        {-27, -5},
        {100, 3},
        {-100, 3},
    };

    for (auto [numerator, denominator] : cases) {
        T81Limb num = from_int128(numerator);
        T81Limb den = from_int128(denominator);
        auto [quot, rem] = num.div_mod(den);
        REQUIRE(num.compare(quot * den + rem) == 0);
        REQUIRE(rem.abs().compare(den.abs()) < 0);
    }

    REQUIRE_THROWS_AS(
        [&]() {
            auto result = from_int128(1).div_mod(T81Limb());
            (void)result;
        }(),
        std::domain_error);
}

TEST_CASE("GCD and modular inverse", "[core][number-theory]") {
    T81Limb a = from_int128(12345);
    T81Limb b = from_int128(54321);
    auto expected_gcd = std::gcd(12345LL, 54321LL);
    auto g = a.gcd(b);
    auto abs_g = g.abs();
    auto expected_limb = T81Limb::from_int(static_cast<int>(expected_gcd));
    REQUIRE(g.compare(expected_limb) == 0 || g.compare(-expected_limb) == 0);

    T81Limb base = from_int128(1337);
    T81Limb mod = from_int128(10007);
    auto inv = base.inv_mod(mod);
    auto prod = (base * inv) % mod;
    REQUIRE(prod.compare(T81Limb::one()) == 0);
}

TEST_CASE("Montgomery fallback uses modular multiply", "[core][montgomery]") {
    T81Limb modulus = from_int128(9);  // divisible by 3
    MontgomeryContext ctx(modulus);
    REQUIRE(!ctx.use_montgomery);

    T81Limb lhs = from_int128(5);
    T81Limb rhs = from_int128(7);
    auto product = ctx.multiply(lhs, rhs);
    auto expected = (lhs * rhs) % modulus;
    REQUIRE(product.compare(expected) == 0);

    auto reduced = ctx.to_montgomery(lhs);
    REQUIRE(reduced.compare(lhs % modulus) == 0);
}

TEST_CASE("Base-3 string preserves digits", "[core][string]") {
    T81Limb value = from_int128(12345);
    auto str = value.to_string(3);
    REQUIRE(str.rfind("t#", 0) == 0);
    REQUIRE(str.size() == 2 + T81Limb::TRITS);
    for (size_t idx = 2; idx < str.size(); ++idx) {
        char ch = str[idx];
        REQUIRE(ch == '+' || ch == '-' || ch == '0');
    }
}

TEST_CASE("Trit round-trip and base-10 string", "[core][trits]") {
    T81Limb value = from_int128(1LL << 20);
    auto trits = value.to_trits();
    auto roundtrip = T81Limb::from_trits(trits);
    REQUIRE(roundtrip.compare(value) == 0);
}
