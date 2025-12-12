// include/t81/core/montgomery.hpp — Montgomery arithmetic interface declarations.

#pragma once

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>

namespace t81::core {

    template <class Int> class MontgomeryContext;

    namespace detail {

        inline limb normalized_limb(detail::limb_int128 value) {
            while (value > MAX_VALUE) {
                value -= RADIX;
            }
            while (value < MIN_VALUE) {
                value += RADIX;
            }
            return limb::from_value(value);
        }

        inline std::pair<limb, detail::limb_int128> radix_digit(detail::limb_int128 value) {
            detail::limb_int128 remainder = value % RADIX;
            detail::limb_int128 carry = (value - remainder) / RADIX;
            if (remainder > MAX_VALUE) {
                remainder -= RADIX;
                ++carry;
            } else if (remainder < MIN_VALUE) {
                remainder += RADIX;
                --carry;
            }
            return {limb::from_value(remainder), carry};
        }

        inline detail::limb_int128 mod_inverse(detail::limb_int128 value,
                                               detail::limb_int128 modulus) {
            if (modulus <= 0) {
                throw std::invalid_argument("modulus must be positive");
            }
            detail::limb_int128 r0 = modulus;
            detail::limb_int128 r1 = value % modulus;
            if (r1 < 0) {
                r1 += modulus;
            }
            detail::limb_int128 t0 = 0;
            detail::limb_int128 t1 = 1;
            while (r1 != 0) {
                detail::limb_int128 q = r0 / r1;
                detail::limb_int128 r2 = r0 - q * r1;
                r0 = r1;
                r1 = r2;
                detail::limb_int128 t2 = t0 - q * t1;
                t0 = t1;
                t1 = t2;
            }
            if (r0 != 1) {
                throw std::invalid_argument("modular inverse does not exist");
            }
            if (t0 < 0) {
                t0 += modulus;
            }
            return t0;
        }

        inline limb reduce_mod(const limb &value, const limb &modulus) {
            const auto dividend = value.to_value();
            const auto divisor = modulus.to_value();
            auto remainder = dividend % divisor;
            if (remainder < 0) {
                remainder += divisor;
            }
            return limb::from_value(remainder);
        }

    } // namespace detail

    template <> class MontgomeryContext<limb> {
      public:
        using int_type = limb;

        explicit MontgomeryContext(const limb &modulus) : modulus_(modulus) {
            const auto mod_value = modulus.to_value();
            if (mod_value <= 0) {
                throw std::invalid_argument("modulus must be positive");
            }
            if (mod_value >= detail::RADIX) {
                throw std::invalid_argument("modulus must be smaller than radix");
            }
            if (mod_value % 3 == 0) {
                throw std::invalid_argument("modulus must not be divisible by 3");
            }
            R_mod_m_ = compute_R_mod_m();
            R2_mod_m_ = compute_R2_mod_m();
            m_prime_ = compute_m_prime();
        }

        const limb &modulus() const noexcept {
            return modulus_;
        }

        limb to_montgomery(const limb &value) const {
            const limb reduced = detail::reduce_mod(value, modulus_);
            const auto [low, high] = limb::mul_wide(reduced, R2_mod_m_);
            auto result = redc(low, high);
            return result;
        }

        limb from_montgomery(const limb &value) const {
            return redc(value, limb::zero());
        }

        limb mul(const limb &lhs, const limb &rhs) const {
            const auto [low, high] = limb::mul_wide(lhs, rhs);
            return redc(low, high);
        }

        limb square(const limb &value) const {
            return mul(value, value);
        }

        limb pow(const limb &base_bar, const limb &exponent) const {
            auto exp_value = exponent.to_value();
            if (exp_value < 0) {
                throw std::domain_error("negative exponent");
            }
            limb result = to_montgomery(limb::one());
            limb base = base_bar;
            while (exp_value > 0) {
                if ((exp_value & 1) != 0) {
                    result = mul(result, base);
                }
                base = square(base);
                exp_value >>= 1;
            }
            return result;
        }

      private:
        limb compute_R_mod_m() const {
            const auto remainder = detail::RADIX % modulus_.to_value();
            return limb::from_value(remainder);
        }

        limb compute_R2_mod_m() const {
            const auto r = R_mod_m_.to_value();
            const auto mod_value = modulus_.to_value();
            const auto sq = (r * r) % mod_value;
            return limb::from_value(sq);
        }

        limb compute_m_prime() const {
            const auto m_value = modulus_.to_value();
            const auto inv = detail::mod_inverse(m_value, detail::RADIX);
            detail::limb_int128 prime = (-inv) % detail::RADIX;
            if (prime < 0) {
                prime += detail::RADIX;
            }
            return detail::normalized_limb(prime);
        }

        limb redc(const limb &low, const limb &high) const {
            const auto [u_low, u_high] = limb::mul_wide(low, m_prime_);
            const auto [um_low, um_high] = limb::mul_wide(u_low, modulus_);
            const detail::limb_int128 low_sum = low.to_value() + um_low.to_value();
            auto [normalized_low, carry] = detail::radix_digit(low_sum);
            detail::limb_int128 high_sum = high.to_value() + um_high.to_value() + carry;
            auto [result, high_carry] = detail::radix_digit(high_sum);
            while (high_carry != 0) {
                detail::limb_int128 adjusted = result.to_value() + high_carry * detail::RADIX;
                auto next = detail::radix_digit(adjusted);
                result = next.first;
                high_carry = next.second;
            }
            return detail::reduce_mod(result, modulus_);
        }

        limb modulus_;
        limb R_mod_m_;
        limb R2_mod_m_;
        limb m_prime_;
    };

    template <> class MontgomeryContext<bigint> {
      public:
        using int_type = bigint;

        explicit MontgomeryContext(const bigint &modulus)
            : modulus_(modulus), radix_power_(modulus_.limb_count()) {
            if (modulus_.is_zero() || modulus_.is_negative()) {
                throw std::invalid_argument("modulus must be positive");
            }
            if (radix_power_ == 0) {
                radix_power_ = 1;
            }
            if (modulus_mod_3(modulus_) == 0) {
                throw std::invalid_argument("modulus must not be divisible by 3");
            }
            R_ = bigint::one().shift_limbs(radix_power_);
            R2_ = R_.shift_limbs(radix_power_);
            R_mod_m_ = bigint::div_mod(R_, modulus_).second;
            R2_mod_m_ = bigint::div_mod(R2_, modulus_).second;
            m_prime_ = compute_m_prime();
        }

        const bigint &modulus() const noexcept {
            return modulus_;
        }

        bigint to_montgomery(const bigint &value) const {
            const auto reduced = reduce(value);
            return redc(reduced * R2_mod_m_);
        }

        bigint from_montgomery(const bigint &value) const {
            return redc(value);
        }

        bigint mul(const bigint &lhs, const bigint &rhs) const {
            return redc(lhs * rhs);
        }

        bigint square(const bigint &value) const {
            return mul(value, value);
        }

        bigint pow(const bigint &base_bar, const bigint &exponent) const {
            if (exponent.is_negative()) {
                throw std::domain_error("negative exponent");
            }
            auto exp = exponent;
            bigint result = to_montgomery(bigint::one());
            bigint base = base_bar;
            const auto two = bigint(2);
            while (!exp.is_zero()) {
                const auto [quotient, remainder] = bigint::div_mod(exp, two);
                if (!remainder.is_zero()) {
                    result = mul(result, base);
                }
                base = square(base);
                exp = quotient;
            }
            return result;
        }

      private:
        bigint reduce(const bigint &value) const {
            const auto remainder = bigint::div_mod(value, modulus_).second;
            return remainder.is_negative() ? remainder + modulus_ : remainder;
        }

        static int modulus_mod_3(const bigint &value) {
            if (value.is_zero()) {
                return 0;
            }
            const auto first = value.limb_at(0).to_value();
            const int rem = static_cast<int>(first % 3);
            return rem < 0 ? rem + 3 : rem;
        }

        bigint compute_m_prime() const {
            const auto inverse = mod_inverse(modulus_, R_);
            auto prime = R_ - inverse;
            if (prime.is_negative()) {
                prime += R_;
            }
            const auto reduced = bigint::div_mod(prime, R_).second;
            return reduced;
        }

        bigint redc(const bigint &T) const {
            const auto remainder = bigint::div_mod(T, R_).second;
            const auto u = bigint::div_mod(remainder * m_prime_, R_).second;
            const auto T_prime = T + u * modulus_;
            auto result = bigint::div_mod(T_prime, R_).first;
            while (result.is_negative()) {
                result += modulus_;
            }
            while (!(result < modulus_)) {
                result -= modulus_;
            }
            return result;
        }

        static bigint mod_inverse(bigint value, const bigint &modulus) {
            auto a = modulus;
            auto b = bigint::div_mod(value, modulus).second;
            if (b.is_negative()) {
                b += modulus;
            }
            auto x0 = bigint::zero();
            auto x1 = bigint::one();
            while (!b.is_zero()) {
                const auto [q, r] = bigint::div_mod(a, b);
                a = b;
                b = r;
                const auto temp = x0 - q * x1;
                x0 = x1;
                x1 = temp;
            }
            if (!(a == bigint::one())) {
                throw std::invalid_argument("modulus not invertible");
            }
            if (x0.is_negative()) {
                x0 += modulus;
            }
            return x0;
        }

        bigint modulus_;
        std::size_t radix_power_;
        bigint R_;
        bigint R2_;
        bigint R_mod_m_;
        bigint R2_mod_m_;
        bigint m_prime_;
    };

} // namespace t81::core
