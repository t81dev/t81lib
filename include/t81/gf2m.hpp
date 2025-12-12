#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <t81/core/bigint.hpp>

namespace t81 {

    class F2m {
      public:
        explicit F2m(core::bigint modulus)
            : modulus_(normalize_modulus(std::move(modulus))),
              modulus_degree_(bit_degree(modulus_)) {
            if (modulus_degree_ < 0) {
                throw std::invalid_argument("F2m modulus must be non-zero");
            }
            pow2_cache_.push_back(core::bigint::one());
        }

        const core::bigint &modulus() const noexcept {
            return modulus_;
        }

        core::bigint add(const core::bigint &lhs, const core::bigint &rhs) const {
            return xor_bits(lhs, rhs);
        }

        core::bigint multiply(core::bigint lhs, core::bigint rhs) const {
            lhs = reduce(std::move(lhs));
            rhs = reduce(std::move(rhs));
            auto product = poly_multiply(lhs, rhs);
            return reduce(std::move(product));
        }

        core::bigint pow(core::bigint base, std::size_t exponent) const {
            auto value = reduce(std::move(base));
            core::bigint result = core::bigint::one();
            while (exponent > 0) {
                if ((exponent & 1) != 0) {
                    result = multiply(result, value);
                }
                value = multiply(value, value);
                exponent >>= 1;
            }
            return result;
        }

        core::bigint reduce(core::bigint value) const {
            auto deg = bit_degree(value);
            while (deg >= modulus_degree_) {
                std::size_t shift = static_cast<std::size_t>(deg - modulus_degree_);
                value = xor_bits(value, shift_left(modulus_, shift));
                deg = bit_degree(value);
            }
            return value;
        }

      private:
        static core::bigint xor_bits(core::bigint lhs, core::bigint rhs) {
            const core::bigint two(2);
            core::bigint result = core::bigint::zero();
            core::bigint place = core::bigint::one();
            while (!lhs.is_zero() || !rhs.is_zero()) {
                auto [lhs_quotient, lhs_remainder] = core::bigint::div_mod(lhs, two);
                auto [rhs_quotient, rhs_remainder] = core::bigint::div_mod(rhs, two);
                const bool lhs_bit = !lhs_remainder.is_zero();
                const bool rhs_bit = !rhs_remainder.is_zero();
                if (lhs_bit ^ rhs_bit) {
                    result += place;
                }
                place *= two;
                lhs = lhs_quotient;
                rhs = rhs_quotient;
            }
            return result;
        }

        static core::bigint normalize_modulus(core::bigint modulus) {
            if (modulus.is_zero()) {
                throw std::invalid_argument("modulus cannot be zero");
            }
            return modulus;
        }

        static int bit_degree(core::bigint value) {
            if (value.is_zero()) {
                return -1;
            }
            value = value.abs();
            int degree = -1;
            const core::bigint two(2);
            while (!value.is_zero()) {
                const auto [quotient, remainder] = core::bigint::div_mod(value, two);
                value = quotient;
                ++degree;
            }
            return degree;
        }

        core::bigint shift_left(const core::bigint &value, std::size_t bits) const {
            return value * pow2(bits);
        }

        core::bigint pow2(std::size_t bits) const {
            if (bits >= pow2_cache_.size()) {
                for (std::size_t next = pow2_cache_.size(); next <= bits; ++next) {
                    pow2_cache_.push_back(pow2_cache_.back() * core::bigint(2));
                }
            }
            return pow2_cache_[bits];
        }

        static core::bigint poly_multiply(core::bigint lhs, core::bigint rhs) {
            core::bigint result = core::bigint::zero();
            const core::bigint two(2);
            while (!rhs.is_zero()) {
                const auto [quotient, remainder] = core::bigint::div_mod(rhs, two);
                if (!remainder.is_zero()) {
                    result = xor_bits(result, lhs);
                }
                lhs *= two;
                rhs = quotient;
            }
            return result;
        }

        core::bigint modulus_;
        int modulus_degree_;
        mutable std::vector<core::bigint> pow2_cache_;
    };

} // namespace t81
