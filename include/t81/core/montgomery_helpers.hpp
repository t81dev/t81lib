// include/t81/core/montgomery_helpers.hpp — Helpers that back Montgomery math.

#pragma once

#include <cstddef>
#include <stdexcept>

#include <t81/core/bigint.hpp>
#include <t81/core/montgomery.hpp>

namespace t81::core::montgomery {

inline MontgomeryContext<limb> make_limb_context(const limb& modulus) {
    return MontgomeryContext<limb>(modulus);
}

inline MontgomeryContext<bigint> make_bigint_context(const bigint& modulus) {
    return MontgomeryContext<bigint>(modulus);
}

inline limb modular_multiply(const MontgomeryContext<limb>& ctx,
                             const limb& lhs,
                             const limb& rhs) {
    const auto lhs_bar = ctx.to_montgomery(lhs);
    const auto rhs_bar = ctx.to_montgomery(rhs);
    const auto product = ctx.mul(lhs_bar, rhs_bar);
    return ctx.from_montgomery(product);
}

inline bigint modular_multiply(const MontgomeryContext<bigint>& ctx,
                               const bigint& lhs,
                               const bigint& rhs) {
    const auto lhs_bar = ctx.to_montgomery(lhs);
    const auto rhs_bar = ctx.to_montgomery(rhs);
    const auto product = ctx.mul(lhs_bar, rhs_bar);
    return ctx.from_montgomery(product);
}

inline limb modular_pow(const MontgomeryContext<limb>& ctx,
                        const limb& base,
                        const limb& exponent) {
    const auto base_bar = ctx.to_montgomery(base);
    const auto power = ctx.pow(base_bar, exponent);
    return ctx.from_montgomery(power);
}

inline bigint modular_pow(const MontgomeryContext<bigint>& ctx,
                          const bigint& base,
                          const bigint& exponent) {
    const auto base_bar = ctx.to_montgomery(base);
    const auto power = ctx.pow(base_bar, exponent);
    return ctx.from_montgomery(power);
}

template <typename Int>
class MontgomeryConstTimeGuard;

template <>
class MontgomeryConstTimeGuard<limb> {
public:
    using context_type = MontgomeryContext<limb>;

    MontgomeryConstTimeGuard(const context_type& ctx, std::size_t max_bits)
        : ctx_(ctx), max_bits_(max_bits) {}

    std::size_t max_bits() const noexcept {
        return max_bits_;
    }

    bool allows(const limb& exponent) const {
        if (exponent.is_negative()) {
            return false;
        }
        return bit_width(exponent) <= max_bits_;
    }

    void require(const limb& exponent) const {
        if (!allows(exponent)) {
            throw std::domain_error("MontgomeryConstTimeGuard<limb>: exponent exceeds guarded bit width");
        }
    }

    limb pow(const limb& base, const limb& exponent) const {
        require(exponent);
        return modular_pow(ctx_, base, exponent);
    }

private:
    static std::size_t bit_width(const limb& value) {
        auto numeric = value.to_value();
        if (numeric < 0) {
            numeric = -numeric;
        }
        std::size_t bits = 0;
        while (numeric > 0) {
            numeric >>= 1;
            ++bits;
        }
        return bits;
    }

    const context_type& ctx_;
    std::size_t max_bits_;
};

template <>
class MontgomeryConstTimeGuard<bigint> {
public:
    using context_type = MontgomeryContext<bigint>;

    MontgomeryConstTimeGuard(const context_type& ctx, std::size_t max_bits)
        : ctx_(ctx), max_bits_(max_bits) {}

    std::size_t max_bits() const noexcept {
        return max_bits_;
    }

    bool allows(const bigint& exponent) const {
        if (exponent.is_negative()) {
            return false;
        }
        return bit_width(exponent) <= max_bits_;
    }

    void require(const bigint& exponent) const {
        if (!allows(exponent)) {
            throw std::domain_error("MontgomeryConstTimeGuard<bigint>: exponent exceeds guarded bit width");
        }
    }

    bigint pow(const bigint& base, const bigint& exponent) const {
        require(exponent);
        return modular_pow(ctx_, base, exponent);
    }

private:
    static std::size_t bit_width(bigint value) {
        if (value.is_negative()) {
            value = value.abs();
        }
        if (value.is_zero()) {
            return 0;
        }
        std::size_t bits = 0;
        const bigint two(2);
        while (!value.is_zero()) {
            const auto division = bigint::div_mod(value, two);
            value = division.first;
            ++bits;
        }
        return bits;
    }

    const context_type& ctx_;
    std::size_t max_bits_;
};

} // namespace t81::core::montgomery
