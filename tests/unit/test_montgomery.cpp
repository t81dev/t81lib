#include <random>

#include <t81/t81lib.hpp>
#include <t81/core/montgomery.hpp>

#include <iostream>

namespace {

t81::core::bigint small_positive_bigint(std::mt19937_64& rng) {
    static std::uniform_int_distribution<std::int64_t> dist(0, 1'000'000);
    return t81::core::bigint(dist(rng));
}

} // namespace

bool check_limb_modular(const t81::core::MontgomeryContext<t81::core::limb>& ctx,
                        const t81::core::limb& a,
                        const t81::core::limb& b) {
    const auto a_bar = ctx.to_montgomery(a);
    const auto b_bar = ctx.to_montgomery(b);
    const auto product = ctx.mul(a_bar, b_bar);
    const auto recovered = ctx.from_montgomery(product);
    const auto mod_value = ctx.modulus().to_value();
    const auto expected = t81::core::limb::from_value(
        (a.to_value() * b.to_value()) % mod_value);
    if (recovered != expected) {
        std::cerr << "limb montgomery mul mismatch\n";
        std::cerr << "a=" << t81::io::to_string(a) << " b=" << t81::io::to_string(b) << "\n";
        std::cerr << "expected=" << t81::io::to_string(expected) << " got=" << t81::io::to_string(recovered) << "\n";
        return false;
    }
    const auto pow_bar = ctx.pow(a_bar, t81::core::limb::from_value(3));
    const auto pow_plain = t81::core::limb::from_value(
        (a.to_value() * a.to_value() * a.to_value()) % mod_value);
    const auto pow_result = ctx.from_montgomery(pow_bar);
    if (pow_result != pow_plain) {
        std::cerr << "limb montgomery pow mismatch\n";
        return false;
    }
    return true;
}

bool check_bigint_modular(const t81::core::MontgomeryContext<t81::core::bigint>& ctx) {
    auto rng = std::mt19937_64(0x1337c0de);
    for (int i = 1; i <= 4; ++i) {
        const auto a = small_positive_bigint(rng);
        const auto b = small_positive_bigint(rng);
        const auto a_bar = ctx.to_montgomery(a);
        const auto b_bar = ctx.to_montgomery(b);
        const auto c_bar = ctx.mul(a_bar, b_bar);
        const auto result = ctx.from_montgomery(c_bar);
        const auto expected = t81::core::bigint::div_mod(a * b, ctx.modulus()).second;
        if (result != expected) {
            std::cerr << "bigint montgomery mul mismatch\n";
            std::cerr << "a=" << t81::io::to_string(a) << " b=" << t81::io::to_string(b) << "\n";
            std::cerr << "expected=" << t81::io::to_string(expected) << " got=" << t81::io::to_string(result) << "\n";
            return false;
        }
        const auto pow_bar = ctx.pow(a_bar, t81::core::bigint(3));
        const auto pow_result = ctx.from_montgomery(pow_bar);
        const auto expected_pow = t81::core::bigint::div_mod(a * a * a, ctx.modulus()).second;
        if (pow_result != expected_pow) {
            std::cerr << "bigint montgomery pow mismatch\n";
            std::cerr << "a=" << t81::io::to_string(a) << "\n";
            std::cerr << "expected_pow=" << t81::io::to_string(expected_pow)
                      << " got=" << t81::io::to_string(pow_result) << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    using t81::core::limb;
    using t81::core::bigint;
    using t81::core::MontgomeryContext;

    const limb modulus = limb::from_value(17);
    const MontgomeryContext<limb> limb_ctx(modulus);
    if (!check_limb_modular(limb_ctx, limb::from_value(3), limb::from_value(5))) {
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
