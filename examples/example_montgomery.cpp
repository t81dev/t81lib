// examples/example_montgomery.cpp — Illustrates Montgomery multiplication helpers in practice.

#include <iostream>
#include <stdexcept>

#include <t81/core/montgomery_helpers.hpp>
#include <t81/t81lib.hpp>

int
main() {
    using t81::core::bigint;
    using t81::core::limb;
    using namespace t81::core::montgomery;

    const limb modulus = limb::from_value(17);
    const auto ctx = make_limb_context(modulus);

    const limb a = limb::from_value(3);
    const limb b = limb::from_value(5);
    const auto product = modular_multiply(ctx, a, b);
    std::cout << "limb modular multiply 3·5 mod 17 = " << t81::io::to_string(product) << "\n";

    const auto power = modular_pow(ctx, a, limb::from_value(4));
    std::cout << "limb modular exponentiation 3^4 mod 17 = " << t81::io::to_string(power) << "\n";

    MontgomeryConstTimeGuard<limb> guard(ctx, 6);
    const auto guarded_pow = guard.pow(a, limb::from_value(4));
    std::cout << "guarded pow = " << t81::io::to_string(guarded_pow) << "\n";
    try {
        guard.require(limb::from_value(1) << static_cast<int>(guard.max_bits()));
    } catch (const std::domain_error &err) {
        std::cout << "guard blocked large exponent: " << err.what() << "\n";
    }

    const bigint big_mod = bigint(197);
    const auto big_ctx = make_bigint_context(big_mod);
    const bigint big_base = bigint(42);
    const bigint big_result = modular_pow(big_ctx, big_base, bigint(3));
    std::cout << "bigint 42^3 mod 197 = " << t81::io::to_string(big_result) << "\n";

    MontgomeryConstTimeGuard<bigint> big_guard(big_ctx, 16);
    if (big_guard.allows(bigint(3))) {
        const auto big_guarded = big_guard.pow(big_base, bigint(3));
        std::cout << "guarded big pow = " << t81::io::to_string(big_guarded) << "\n";
    }

    return 0;
}
