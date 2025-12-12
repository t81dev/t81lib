// tests/unit/test_montgomery.cpp — Unit tests validating Montgomery reductions and helpers.

#include <array>
#include <random>

#include <t81/core/montgomery.hpp>
#include <t81/core/montgomery_helpers.hpp>
#include <t81/t81lib.hpp>

#include <iostream>
#include <stdexcept>

namespace {

    using t81::core::bigint;
    using t81::core::limb;
    using t81::core::MontgomeryContext;

    t81::core::bigint small_positive_bigint(std::mt19937_64 &rng) {
        static std::uniform_int_distribution<std::int64_t> dist(0, 1'000'000);
        return t81::core::bigint(dist(rng));
    }

    bool check_limb_modular(const MontgomeryContext<limb> &ctx) {
        using namespace t81::core::montgomery;
        using limb_int128 = t81::core::detail::limb_int128;

        const auto modulus_value = ctx.modulus().to_value();
        const std::array<long long, 5> values = {0, 1, 7, 15, -11};
        const std::array<int, 4> exponents = {0, 1, 2, 5};

        auto reduce_value = [&](limb_int128 input) {
            limb_int128 reduced = input % modulus_value;
            if (reduced < 0) {
                reduced += modulus_value;
            }
            return reduced;
        };

        auto pow_mod = [&](long long base, int exponent) {
            limb_int128 result = 1;
            limb_int128 current = reduce_value(base);
            limb_int128 remaining = exponent;
            while (remaining > 0) {
                if ((remaining & 1) != 0) {
                    result = (result * current) % modulus_value;
                }
                current = (current * current) % modulus_value;
                remaining >>= 1;
            }
            return result;
        };

        MontgomeryConstTimeGuard<limb> guard(ctx, 6);

        for (long long a_value : values) {
            const auto a = limb::from_value(a_value);
            const auto recovered_a = ctx.from_montgomery(ctx.to_montgomery(a));
            const auto expected_a = limb::from_value(reduce_value(a_value));
            if (recovered_a != expected_a) {
                std::cerr << "limb conversion mismatch\n";
                return false;
            }
            for (long long b_value : values) {
                const auto b = limb::from_value(b_value);
                const auto product = modular_multiply(ctx, a, b);
                limb_int128 expected =
                    (reduce_value(a_value) * reduce_value(b_value)) % modulus_value;
                if (expected < 0) {
                    expected += modulus_value;
                }
                const auto expected_limb = limb::from_value(expected);
                if (product != expected_limb) {
                    std::cerr << "limb montgomery mul mismatch\n";
                    std::cerr << "a=" << t81::io::to_string(a) << " b=" << t81::io::to_string(b)
                              << "\n";
                    std::cerr << "expected=" << t81::io::to_string(expected_limb)
                              << " got=" << t81::io::to_string(product) << "\n";
                    return false;
                }
            }
            for (int exponent : exponents) {
                const auto pow_direct = modular_pow(ctx, a, limb::from_value(exponent));
                const auto pow_result = guard.pow(a, limb::from_value(exponent));
                const auto expected_pow = limb::from_value(pow_mod(a_value, exponent));
                if (pow_direct != expected_pow || pow_result != expected_pow) {
                    std::cerr << "limb montgomery pow mismatch\n";
                    return false;
                }
            }
        }

        const auto small_exponent = limb::from_value(3);
        if (!guard.allows(small_exponent)) {
            std::cerr << "const-time guard rejected legal exponent\n";
            return false;
        }
        const auto large_exponent = limb::from_value(static_cast<long long>(1) << guard.max_bits());
        bool threw = false;
        try {
            guard.require(large_exponent);
        } catch (const std::domain_error &) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "const-time guard failed to reject overflow exponent\n";
            return false;
        }
        return true;
    }

    bool check_bigint_modular(const MontgomeryContext<bigint> &ctx) {
        using namespace t81::core::montgomery;

        auto rng = std::mt19937_64(0x1337c0de);
        const std::array<int, 4> exponents = {0, 1, 3, 5};
        MontgomeryConstTimeGuard<bigint> guard(ctx, 16);

        auto reduce = [&](const bigint &value) {
            auto remainder = bigint::div_mod(value, ctx.modulus()).second;
            if (remainder.is_negative()) {
                remainder += ctx.modulus();
            }
            return remainder;
        };

        for (int iteration = 0; iteration < 8; ++iteration) {
            const auto a = small_positive_bigint(rng);
            const auto b = small_positive_bigint(rng);
            const auto product = modular_multiply(ctx, a, b);
            const auto expected = reduce(a * b);
            if (product != expected) {
                std::cerr << "bigint montgomery mul mismatch\n"
                          << "a=" << t81::io::to_string(a) << " b=" << t81::io::to_string(b) << "\n"
                          << "expected=" << t81::io::to_string(expected)
                          << " got=" << t81::io::to_string(product) << "\n";
                return false;
            }
            for (int exponent : exponents) {
                const auto pow_result = guard.pow(a, bigint(exponent));
                bigint pow_acc = bigint::one();
                for (int step = 0; step < exponent; ++step) {
                    pow_acc *= a;
                }
                const auto expected_pow = reduce(pow_acc);
                if (pow_result != expected_pow) {
                    std::cerr << "bigint montgomery pow mismatch\n"
                              << "a=" << t81::io::to_string(a) << "\n"
                              << "expected_pow=" << t81::io::to_string(expected_pow)
                              << " got=" << t81::io::to_string(pow_result) << "\n";
                    return false;
                }
            }
        }

        const auto small_exponent = bigint(5);
        if (!guard.allows(small_exponent)) {
            std::cerr << "bigint const-time guard rejected legal exponent\n";
            return false;
        }

        bigint overflow = bigint::one();
        for (std::size_t i = 0; i <= guard.max_bits(); ++i) {
            overflow *= bigint(2);
        }
        bool threw = false;
        try {
            guard.require(overflow);
        } catch (const std::domain_error &) {
            threw = true;
        }
        if (!threw) {
            std::cerr << "bigint const-time guard failed to reject overflow exponent\n";
            return false;
        }
        return true;
    }

} // namespace

int main() {
    const limb modulus = limb::from_value(17);
    const MontgomeryContext<limb> limb_ctx(modulus);
    if (!check_limb_modular(limb_ctx)) {
        return 1;
    }
    const bigint big_mod = bigint(197);
    const MontgomeryContext<bigint> bigint_ctx(big_mod);
    if (!check_bigint_modular(bigint_ctx)) {
        return 1;
    }
    std::cout << "montgomery context tests passed\n";
    return 0;
}
