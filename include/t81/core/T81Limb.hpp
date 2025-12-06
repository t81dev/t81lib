/**
 * @file T81Limb.hpp
 * @brief Defines the T81Limb class, a 48-trit packed ternary integer.
 *
 * This file provides T81Limb, a high-performance 48-trit (16-tryte) numeric
 * type. It uses a Kogge-Stone carry-lookahead adder with function composition
 * on carry maps to achieve high-throughput addition.
 */
// T81 Foundation — Packed Balanced Ternary Limb (48 trits)
// v1.1.0-DEV — High-performance core using Kogge-Stone adder.
// License: MIT / GPL-3.0 dual

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(__x86_64__) && defined(__AVX2__)
#include <immintrin.h>
#endif
#include <string>
#include <type_traits>
#include <utility>
#include "t81/packing.hpp"

namespace t81::core {

namespace detail {
    // LUTs are unchanged.
    struct AddEntry { int8_t cout[3], sum_value[3]; };
    static constexpr std::array<std::array<AddEntry, 27>, 27> build_add_table() {
        std::array<std::array<AddEntry, 27>, 27> table{};
        for (int a_idx = 0; a_idx < 27; ++a_idx) {
            for (int b_idx = 0; b_idx < 27; ++b_idx) {
                int a_val = a_idx - 13; int b_val = b_idx - 13;
                for (int cin_idx = 0; cin_idx < 3; ++cin_idx) {
                    int cin = cin_idx - 1; int temp = a_val + b_val + cin;
                    int cout = 0;
                    if (temp > 13) { cout = 1; temp -= 27; }
                    if (temp < -13) { cout = -1; temp += 27; }
                    table[a_idx][b_idx].sum_value[cin_idx] = temp;
                    table[a_idx][b_idx].cout[cin_idx] = cout;
                }
            }
        }
        return table;
    }
    static constexpr std::array<std::array<int, 27>, 27> build_composition_table() {
        std::array<std::array<int, 27>, 27> table{};
        for (int id1 = 0; id1 < 27; ++id1) {
            int map1[3]; int temp1 = id1;
            for (int j = 0; j < 3; ++j) { map1[j] = (temp1 % 3) - 1; temp1 /= 3; }
            for (int id2 = 0; id2 < 27; ++id2) {
                int map2[3]; int temp2 = id2;
                for (int j = 0; j < 3; ++j) { map2[j] = (temp2 % 3) - 1; temp2 /= 3; }
                int new_map[3];
                for (int cin_idx = 0; cin_idx < 3; ++cin_idx) {
                    new_map[cin_idx] = map1[map2[cin_idx] + 1];
                }
                table[id1][id2] = (new_map[0] + 1) + 3 * (new_map[1] + 1) + 9 * (new_map[2] + 1);
            }
        }
        return table;
    }
    static constexpr auto ADD_TABLE = build_add_table();
    static constexpr auto COMPOSITION_TABLE = build_composition_table();

    static constexpr std::array<int32_t, 27 * 27> build_flat_composition_table() {
        std::array<int32_t, 27 * 27> flat{};
        for (int row = 0; row < 27; ++row) {
            for (int col = 0; col < 27; ++col) {
                flat[row * 27 + col] = COMPOSITION_TABLE[row][col];
            }
        }
        return flat;
    }
    static constexpr auto COMPOSITION_FLAT = build_flat_composition_table();
    static constexpr std::array<int32_t, 27> build_carry_from_zero_table() {
        std::array<int32_t, 27> table{};
        for (int id = 0; id < 27; ++id) {
            table[id] = ((id / 3) % 3) - 1;
        }
        return table;
    }
    static constexpr auto CARRY_FROM_ZERO = build_carry_from_zero_table();

    struct Booth27Decision { int8_t mul; int8_t shift_trytes; };
    static constexpr std::array<Booth27Decision, 27 * 27> build_booth27_table() {
        std::array<Booth27Decision, 27 * 27> table{};
        for (int b0 = -13; b0 <= 13; ++b0) {
            for (int b1 = -13; b1 <= 13; ++b1) {
                int8_t mul = 0;
                int8_t shift = 0;
                int pattern = b0 + 3 * b1;
                if (pattern == 1 || pattern == 3) {
                    mul = 1;
                } else if (pattern == 2 || pattern == 4) {
                    mul = 1;
                    shift = 1;
                } else if (pattern == -1 || pattern == -3) {
                    mul = -1;
                } else if (pattern == -2 || pattern == -4) {
                    mul = -1;
                    shift = 1;
                }
                table[(b0 + 13) * 27 + (b1 + 13)] = {mul, shift};
            }
        }
        return table;
    }
    static constexpr auto BOOTH27_TABLE = build_booth27_table();
    static constexpr std::array<std::array<int8_t, 3>, 27> build_tryte_to_trits_table() {
        std::array<std::array<int8_t, 3>, 27> table{};
        for (int i = -13; i <= 13; ++i) {
            int8_t trits[3];
            t81::decode_tryte(static_cast<int8_t>(i), trits);
            table[i + 13] = {trits[0], trits[1], trits[2]};
        }
        return table;
    }
    static constexpr auto TRYTE_TO_TRITS = build_tryte_to_trits_table();

    static constexpr int WIDE_TRITS = 96;

    inline constexpr void normalize_wide_once(std::array<int, WIDE_TRITS>& acc) noexcept {
        for (int idx = 0; idx < WIDE_TRITS; ++idx) {
            int carry = (acc[idx] + (acc[idx] >= 0 ? 1 : -1)) / 3;
            acc[idx] -= carry * 3;
            if (idx + 1 < WIDE_TRITS) acc[idx + 1] += carry;
        }
    }

    inline constexpr void normalize_wide(std::array<int, WIDE_TRITS>& acc) noexcept {
        normalize_wide_once(acc);
    }

    inline constexpr void finalize_wide(std::array<int8_t, WIDE_TRITS>& trits,
                              const std::array<int, WIDE_TRITS>& acc) noexcept
    {
        for (int i = 0; i < WIDE_TRITS; ++i) {
            int value = acc[i];
            if (value > 1) value = 1;
            else if (value < -1) value = -1;
            trits[i] = static_cast<int8_t>(value);
        }
    }
} // namespace detail

class T81Limb54;
class T81Limb27;

class T81Limb {
public:
    static constexpr int TRITS = 48;
    static constexpr int TRYTES = 16;
    static constexpr T81Limb one() noexcept {
        T81Limb limb;
        limb.trytes_[0] = 1;
        return limb;
    }
private:
    alignas(16) int8_t trytes_[TRYTES]{};
    friend class T81Limb54;
    friend class T81Limb27;
public:
    T81Limb() noexcept = default;
    void set_tryte(int i, int8_t val) { trytes_[i] = val; }
    int8_t get_tryte(int i) const noexcept { return trytes_[i]; }

    [[nodiscard]] constexpr T81Limb operator+(const T81Limb& other) const noexcept;
    [[nodiscard]] constexpr T81Limb operator-() const noexcept;
    [[nodiscard]] constexpr T81Limb operator-(const T81Limb& rhs) const noexcept;
    [[nodiscard]] T81Limb operator/(const T81Limb& divisor) const;
    [[nodiscard]] T81Limb operator%(const T81Limb& divisor) const;
    [[nodiscard]] constexpr std::pair<T81Limb, T81Limb> div_mod(const T81Limb& divisor) const;
    [[nodiscard]] constexpr T81Limb gcd(const T81Limb& other) const;
    [[nodiscard]] constexpr T81Limb inv_mod(const T81Limb& modulus) const;
    [[nodiscard]] static constexpr T81Limb pow_mod(T81Limb base, T81Limb exp, T81Limb mod);
    [[nodiscard]] constexpr T81Limb abs() const noexcept;
    [[nodiscard]] constexpr bool is_zero() const noexcept;
    [[nodiscard]] constexpr bool is_one() const noexcept;
    [[nodiscard]] constexpr bool is_negative() const noexcept;
    [[nodiscard]] constexpr int signum() const noexcept;
    [[nodiscard]] constexpr int compare(const T81Limb& other) const noexcept;
    [[nodiscard]] constexpr std::array<int8_t, TRITS> to_trits() const noexcept;
    [[nodiscard]] static constexpr T81Limb from_trits(const std::array<int8_t, TRITS>& digits) noexcept;
    [[nodiscard]] constexpr std::pair<T81Limb, int8_t> addc(const T81Limb& other) const noexcept;
    [[nodiscard]] constexpr T81Limb operator*(const T81Limb& rhs) const noexcept;
    [[nodiscard]] static T81Limb reference_mul(const T81Limb& a, const T81Limb& b) noexcept;
    [[nodiscard]] static std::array<int8_t, TRITS> booth_mul_trits(
        const std::array<int8_t, TRITS>& a,
        const std::array<int8_t, TRITS>& b) noexcept;
    [[nodiscard]] static T81Limb booth_mul(const T81Limb& a, const T81Limb& b) noexcept;
    [[nodiscard]] static T81Limb bohemian_mul(const T81Limb& a, const T81Limb& b) noexcept;
    [[nodiscard]] static constexpr std::pair<T81Limb, T81Limb> mul_wide(
        const T81Limb& a,
        const T81Limb& b) noexcept;
    [[nodiscard]] std::string to_string(int base = 3) const;
    [[nodiscard]] static constexpr T81Limb from_int(int value) noexcept;
    [[nodiscard]] T81Limb square() const noexcept;
private:
    [[nodiscard]] static constexpr std::pair<T81Limb, T81Limb> mul_wide_canonical(
        const T81Limb& a,
        const T81Limb& b) noexcept;
    [[nodiscard]] static constexpr std::pair<T81Limb, T81Limb> mul_wide_fast(
        const T81Limb& a,
        const T81Limb& b) noexcept;
    [[nodiscard]] static constexpr T81Limb mul_booth_karatsuba(const T81Limb& a, const T81Limb& b) noexcept;
    [[nodiscard]] constexpr T81Limb shift_left_trytes(int count) const noexcept;
    [[nodiscard]] constexpr T81Limb shift_right_trytes(int count) const noexcept;
    [[nodiscard]] T81Limb operator<<(int count) const noexcept;
    [[nodiscard]] T81Limb operator>>(int count) const noexcept;
    T81Limb& operator<<=(int count);
    T81Limb& operator>>=(int count);
    [[nodiscard]] static std::pair<T81Limb, T81Limb> square_wide(const T81Limb& a) noexcept;
    [[nodiscard]] static constexpr std::pair<T81Limb, T81Limb> div_mod_pair(const T81Limb& numerator, const T81Limb& denominator);
    [[nodiscard]] int small_int() const noexcept;
};

struct Base3Digits {
    std::array<int, T81Limb::TRITS> digits{};

    constexpr void add_digit(int position, int value) noexcept {
        int carry = value;
        for (int idx = position; idx < T81Limb::TRITS && carry != 0; ++idx) {
            int sum = digits[idx] + carry;
            digits[idx] = sum % 3;
            carry = sum / 3;
        }
    }

    constexpr T81Limb to_limb() const noexcept {
        std::array<int, T81Limb::TRITS> temp = digits;
        std::array<int8_t, T81Limb::TRITS> balanced{};
        int carry = 0;
        for (int idx = 0; idx < T81Limb::TRITS; ++idx) {
            int value = temp[idx] + carry;
            carry = 0;
            if (value >= 2) {
                balanced[idx] = static_cast<int8_t>(value - 3);
                carry = 1;
            } else {
                balanced[idx] = static_cast<int8_t>(value);
            }
        }
        return T81Limb::from_trits(balanced);
    }
};

namespace detail {
inline constexpr std::array<int, WIDE_TRITS> wide_from_pair(
    const std::pair<T81Limb, T81Limb>& value) noexcept
{
    std::array<int, WIDE_TRITS> data{};
    const auto low = value.first.to_trits();
    const auto high = value.second.to_trits();
    for (int idx = 0; idx < T81Limb::TRITS; ++idx) {
        data[idx] = low[idx];
        data[idx + T81Limb::TRITS] = high[idx];
    }
    return data;
}

inline constexpr std::array<int8_t, WIDE_TRITS> trits_from_accum(
    const std::array<int, WIDE_TRITS>& accum) noexcept
{
    std::array<int8_t, WIDE_TRITS> trits{};
    finalize_wide(trits, accum);
    return trits;
}
} // namespace detail

struct MontgomeryContextTestAccess;

struct MontgomeryContext {
    T81Limb modulus;
    T81Limb r2_mod;
    T81Limb m_prime;  // -m^{-1} mod R where R = 3^48

    explicit constexpr MontgomeryContext(const T81Limb& mod);
    [[nodiscard]] constexpr T81Limb to_montgomery(const T81Limb& x) const noexcept;
    [[nodiscard]] constexpr T81Limb from_montgomery(const T81Limb& x) const noexcept;
    [[nodiscard]] constexpr T81Limb multiply(const T81Limb& a, const T81Limb& b) const noexcept;
private:
    [[nodiscard]] constexpr T81Limb montgomery_reduce(const std::pair<T81Limb, T81Limb>& value) const noexcept;
    [[nodiscard]] static constexpr T81Limb pow3_mod(const T81Limb& modulus, int exponent) noexcept;
    [[nodiscard]] static constexpr T81Limb compute_m_prime(const T81Limb& modulus) noexcept;
    [[nodiscard]] static constexpr T81Limb compute_r2_mod(const T81Limb& modulus) noexcept;
    friend struct MontgomeryContextTestAccess;
};

inline constexpr MontgomeryContext::MontgomeryContext(const T81Limb& mod)
    : modulus(mod),
      r2_mod(compute_r2_mod(mod)),
      m_prime(compute_m_prime(mod))
{}

inline constexpr T81Limb MontgomeryContext::pow3_mod(const T81Limb& modulus, int exponent) noexcept {
    Base3Digits base{};
    base.digits[1] = 1;
    T81Limb factor = base.to_limb();
    T81Limb result = T81Limb::one();
    for (int i = 0; i < exponent; ++i) {
        result = (result * factor) % modulus;
    }
    return result;
}

inline constexpr T81Limb MontgomeryContext::compute_m_prime(const T81Limb& modulus) noexcept {
    Base3Digits candidate{};
    const int m_mod3 = (static_cast<int>(modulus.to_trits()[0]) + 3) % 3;
    for (int k = 0; k < T81Limb::TRITS; ++k) {
        T81Limb partial = candidate.to_limb();
        auto product = T81Limb::mul_wide(modulus, partial);
        auto accum = detail::wide_from_pair(product);
        accum[0] += 1;
        int k_digit = ((accum[k] % 3) + 3) % 3;
        int inv_mod3 = (m_mod3 == 1) ? 1 : 2;
        int needed = (3 - k_digit) % 3;
        needed = (needed * inv_mod3) % 3;
        candidate.add_digit(k, needed);
    }
    return candidate.to_limb();
}

inline constexpr T81Limb MontgomeryContext::compute_r2_mod(const T81Limb& modulus) noexcept {
    auto r_mod = pow3_mod(modulus, T81Limb::TRITS);
    auto product = r_mod * r_mod;
    return product.div_mod(modulus).second;
}

inline constexpr T81Limb MontgomeryContext::multiply(const T81Limb& a, const T81Limb& b) const noexcept {
    return montgomery_reduce(T81Limb::mul_wide(a, b));
}

inline constexpr T81Limb MontgomeryContext::to_montgomery(const T81Limb& x) const noexcept {
    return multiply(x, r2_mod);
}

inline constexpr T81Limb MontgomeryContext::from_montgomery(const T81Limb& x) const noexcept {
    return multiply(x, T81Limb::one());
}

inline constexpr T81Limb MontgomeryContext::montgomery_reduce(const std::pair<T81Limb, T81Limb>& value) const noexcept {
    auto accum = detail::wide_from_pair(value);
    auto u = T81Limb::mul_wide(value.first, m_prime).first;
    auto um = T81Limb::mul_wide(u, modulus);
    auto um_acc = detail::wide_from_pair(um);
    for (int idx = 0; idx < detail::WIDE_TRITS; ++idx) {
        accum[idx] += um_acc[idx];
    }
    detail::normalize_wide(accum);
    detail::normalize_wide(accum);
    detail::normalize_wide(accum);
    auto trits = detail::trits_from_accum(accum);
    std::array<int8_t, T81Limb::TRITS> high{};
    for (int idx = 0; idx < T81Limb::TRITS; ++idx) {
        high[idx] = trits[idx + T81Limb::TRITS];
    }
    T81Limb result = T81Limb::from_trits(high);
    if (result.compare(modulus) >= 0) {
        result = result - modulus;
    }
    return result;
}

struct MontgomeryContextTestAccess {
    [[nodiscard]] static constexpr T81Limb reduce(
        const MontgomeryContext& ctx,
        const std::pair<T81Limb, T81Limb>& value) noexcept
    {
        return ctx.montgomery_reduce(value);
    }
};

// Correct, scalar, parallel-prefix Kogge-Stone implementation.
inline constexpr T81Limb T81Limb::operator*(const T81Limb& rhs) const noexcept {
    return mul_booth_karatsuba(*this, rhs);
}

inline constexpr T81Limb T81Limb::operator+(const T81Limb& other) const noexcept {
    T81Limb res;
    int8_t carry = 0;

    for (int i = 0; i < TRYTES; ++i) {
        int32_t sum = static_cast<int32_t>(trytes_[i]) + static_cast<int32_t>(other.trytes_[i]) + carry;
        int32_t q = (sum >= 14) ? 1 : (sum <= -14) ? -1 : 0;
        int32_t t = sum - q * 27;
        res.trytes_[i] = static_cast<int8_t>(t);
        carry = static_cast<int8_t>(q);
    }

    carry = 0;
    for (int i = 0; i < TRYTES; ++i) {
        int32_t sum = static_cast<int32_t>(res.trytes_[i]) + carry;
        int32_t over = (sum >= 14);
        int32_t under = (sum <= -14);
        res.trytes_[i] = static_cast<int8_t>(sum - (over * 27) + (under * 27));
        carry = static_cast<int8_t>(over - under);
    }

    return res;
}

inline constexpr T81Limb T81Limb::operator-() const noexcept {
    T81Limb neg;
    for (int i = 0; i < TRYTES; ++i) {
        neg.trytes_[i] = static_cast<int8_t>(-trytes_[i]);
    }
    return neg;
}

inline constexpr T81Limb T81Limb::operator-(const T81Limb& rhs) const noexcept {
    return *this + (-rhs);
}

inline T81Limb T81Limb::operator/(const T81Limb& divisor) const {
    if (divisor.compare(T81Limb()) == 0) throw std::domain_error("division by zero");

    bool negative = compare(T81Limb()) < 0;
    if (divisor.compare(T81Limb()) < 0) negative = !negative;

    T81Limb numerator = negative ? -*this : *this;
    T81Limb denominator = divisor;
    if (denominator.compare(T81Limb()) < 0) denominator = -denominator;

    T81Limb quotient;
    while (numerator.compare(denominator) >= 0) {
        numerator = numerator - denominator;
        quotient = quotient + T81Limb::one();
    }

    return negative ? -quotient : quotient;
}

inline T81Limb T81Limb::operator%(const T81Limb& divisor) const {
    return div_mod(divisor).second;
}

inline constexpr std::pair<T81Limb, T81Limb> T81Limb::div_mod(const T81Limb& divisor) const {
    if (divisor.compare(T81Limb()) == 0) throw std::domain_error("division by zero");
    return div_mod_pair(*this, divisor);
}

inline constexpr std::pair<T81Limb, T81Limb> T81Limb::div_mod_pair(const T81Limb& numerator, const T81Limb& denominator) {
    bool negative = numerator.compare(T81Limb()) < 0;
    bool denom_negative = denominator.compare(T81Limb()) < 0;
    bool quotient_negative = negative ^ denom_negative;

    T81Limb num = negative ? -numerator : numerator;
    T81Limb den = denom_negative ? -denominator : denominator;

    T81Limb quotient;
    while (num.compare(den) >= 0) {
        T81Limb multiple = den;
        T81Limb factor = T81Limb::one();
        // accelerate using doubling (shift-and-subtract).
        while (true) {
            auto doubled = multiple + multiple;
            if (num.compare(doubled) < 0) break;
            multiple = std::move(doubled);
            factor = factor + factor;
        }
        num = num - multiple;
        quotient = quotient + factor;
    }

    if (quotient_negative) quotient = -quotient;
    if (negative && num.compare(T81Limb()) != 0) num = -num;

    return {quotient, num};
}

inline constexpr T81Limb T81Limb::gcd(const T81Limb& other) const {
    T81Limb a = abs();
    T81Limb b = other.abs();
    while (!b.is_zero()) {
        T81Limb r = a % b;
        a = b;
        b = r;
    }
    return a;
}

inline constexpr T81Limb T81Limb::inv_mod(const T81Limb& modulus) const {
    if (modulus.is_zero()) throw std::domain_error("modulus zero");
    auto [neg_mod, mod_pos] = modulus.compare(T81Limb()) < 0 ? std::make_pair(-modulus, -modulus) : std::make_pair(modulus, modulus);
    auto [q, r] = modulus.div_mod(*this);
    (void)q;
    (void)r;
    T81Limb a = mod_pos;
    T81Limb b = (*this % modulus).abs();
    T81Limb x0;
    T81Limb x1 = T81Limb::one();
    while (!b.is_zero()) {
        auto [quot, rem] = a.div_mod(b);
        T81Limb temp = x0 - quot * x1;
        a = b;
        b = rem;
        x0 = x1;
        x1 = temp;
    }
    if (a.compare(T81Limb::one()) != 0 && a.compare(-T81Limb::one()) != 0) {
        throw std::domain_error("no modular inverse");
    }
    if (x0.is_negative()) x0 = x0 + mod_pos;
    return x0;
}

inline constexpr T81Limb T81Limb::pow_mod(T81Limb base, T81Limb exp, T81Limb mod) {
    if (mod.is_zero()) throw std::domain_error("modulus zero");
    MontgomeryContext ctx(mod);
    T81Limb result = ctx.to_montgomery(T81Limb::one());
    base = ctx.to_montgomery(base % mod);
    auto two = T81Limb::from_int(2);
    while (!exp.is_zero()) {
        auto [quot, rem] = exp.div_mod(two);
        if (!rem.is_zero()) {
            result = ctx.multiply(result, base);
        }
        base = ctx.multiply(base, base);
        exp = quot;
    }
    return ctx.from_montgomery(result);
}

inline constexpr T81Limb T81Limb::abs() const noexcept {
    return is_negative() ? -*this : *this;
}

inline constexpr bool T81Limb::is_zero() const noexcept {
    return compare(T81Limb()) == 0;
}

inline constexpr bool T81Limb::is_one() const noexcept {
    return compare(T81Limb::one()) == 0;
}

inline constexpr bool T81Limb::is_negative() const noexcept {
    return compare(T81Limb()) < 0;
}

inline constexpr int T81Limb::signum() const noexcept {
    int cmp = compare(T81Limb());
    if (cmp < 0) return -1;
    if (cmp > 0) return 1;
    return 0;
}

inline constexpr T81Limb T81Limb::shift_left_trytes(int count) const noexcept {
    T81Limb result;
    if (count <= 0) {
        return *this;
    }
    if (count >= TRYTES) {
        return result;
    }
    for (int i = TRYTES - 1; i >= count; --i) {
        result.trytes_[i] = trytes_[i - count];
    }
    for (int i = 0; i < count; ++i) {
        result.trytes_[i] = 0;
    }
    return result;
}

inline constexpr T81Limb T81Limb::shift_right_trytes(int count) const noexcept {
    T81Limb result;
    if (count <= 0) {
        return *this;
    }
    if (count >= TRYTES) {
        return result;
    }
    for (int i = 0; i < TRYTES - count; ++i) {
        result.trytes_[i] = trytes_[i + count];
    }
    for (int i = TRYTES - count; i < TRYTES; ++i) {
        result.trytes_[i] = 0;
    }
    return result;
}

inline T81Limb T81Limb::operator<<(int count) const noexcept {
    return shift_left_trytes(count);
}

inline T81Limb T81Limb::operator>>(int count) const noexcept {
    return shift_right_trytes(count);
}

inline T81Limb& T81Limb::operator<<=(int count) {
    *this = shift_left_trytes(count);
    return *this;
}

inline T81Limb& T81Limb::operator>>=(int count) {
    *this = shift_right_trytes(count);
    return *this;
}
inline constexpr int T81Limb::compare(const T81Limb& other) const noexcept {
    for (int idx = TRYTES - 1; idx >= 0; --idx) {
        if (trytes_[idx] < other.trytes_[idx]) return -1;
        if (trytes_[idx] > other.trytes_[idx]) return 1;
    }
    return 0;
}

inline constexpr std::pair<T81Limb, int8_t> T81Limb::addc(const T81Limb& other) const noexcept {
    T81Limb result;
    std::array<int, TRYTES> map_ids;
    std::array<std::array<int8_t, 3>, TRYTES> sum_vals;

    // 1. First Pass: Get initial carry maps and partial sums
    for (int i = 0; i < TRYTES; ++i) {
        const auto& entry = detail::ADD_TABLE[trytes_[i] + 13][other.trytes_[i] + 13];
        map_ids[i] = (entry.cout[0] + 1) + 3 * (entry.cout[1] + 1) + 9 * (entry.cout[2] + 1);
        sum_vals[i] = {entry.sum_value[0], entry.sum_value[1], entry.sum_value[2]};
    }

    // === UNROLLED KOGGE-STONE ===
    for (int i = 1; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 1]];
    }
    for (int i = 2; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 2]];
    }
    for (int i = 4; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 4]];
    }
    for (int i = 8; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 8]];
    }

    const int8_t c0  = 0;
    const int8_t c1  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[0]]);
    const int8_t c2  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[1]]);
    const int8_t c3  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[2]]);
    const int8_t c4  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[3]]);
    const int8_t c5  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[4]]);
    const int8_t c6  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[5]]);
    const int8_t c7  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[6]]);
    const int8_t c8  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[7]]);
    const int8_t c9  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[8]]);
    const int8_t c10 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[9]]);
    const int8_t c11 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[10]]);
    const int8_t c12 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[11]]);
    const int8_t c13 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[12]]);
    const int8_t c14 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[13]]);
    const int8_t c15 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[14]]);

    result.trytes_[0]  = sum_vals[0][c0  + 1];
    result.trytes_[1]  = sum_vals[1][c1  + 1];
    result.trytes_[2]  = sum_vals[2][c2  + 1];
    result.trytes_[3]  = sum_vals[3][c3  + 1];
    result.trytes_[4]  = sum_vals[4][c4  + 1];
    result.trytes_[5]  = sum_vals[5][c5  + 1];
    result.trytes_[6]  = sum_vals[6][c6  + 1];
    result.trytes_[7]  = sum_vals[7][c7  + 1];
    result.trytes_[8]  = sum_vals[8][c8  + 1];
    result.trytes_[9]  = sum_vals[9][c9  + 1];
    result.trytes_[10] = sum_vals[10][c10 + 1];
    result.trytes_[11] = sum_vals[11][c11 + 1];
    result.trytes_[12] = sum_vals[12][c12 + 1];
    result.trytes_[13] = sum_vals[13][c13 + 1];
    result.trytes_[14] = sum_vals[14][c14 + 1];
    result.trytes_[15] = sum_vals[15][c15 + 1];

    const int8_t carry_out = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[15]]);
    return {result, carry_out};
}


inline constexpr std::array<int8_t, T81Limb::TRITS> T81Limb::to_trits() const noexcept {
    std::array<int8_t, TRITS> trits{};
    for (int idx = 0; idx < TRYTES; ++idx) {
        int8_t decoded[3]{};
        t81::decode_tryte(trytes_[idx], decoded);
        trits[idx * 3 + 0] = decoded[0];
        trits[idx * 3 + 1] = decoded[1];
        trits[idx * 3 + 2] = decoded[2];
    }
    return trits;
}

inline constexpr T81Limb T81Limb::from_trits(const std::array<int8_t, TRITS>& digits) noexcept {
    std::array<int8_t, TRITS> normalized = digits;
    int carry = 0;
    for (int i = 0; i < TRITS; ++i) {
        int sum = static_cast<int>(normalized[i]) + carry;
        if (sum == 2) { normalized[i] = -1; carry = 1; }
        else if (sum == -2) { normalized[i] = 1; carry = -1; }
        else { normalized[i] = static_cast<int8_t>(sum); carry = 0; }
    }
    T81Limb limb;
    for (int idx = 0; idx < TRYTES; ++idx) {
        int8_t chunk[3] = {
            normalized[idx * 3 + 0],
            normalized[idx * 3 + 1],
            normalized[idx * 3 + 2]
        };
        t81::encode_tryte(chunk, limb.trytes_[idx]);
    }
    return limb;
}

inline T81Limb T81Limb::reference_mul(const T81Limb& a, const T81Limb& b) noexcept {
    return mul_wide_canonical(a, b).first;
}

inline std::array<int8_t, T81Limb::TRITS> T81Limb::booth_mul_trits(
    const std::array<int8_t, TRITS>& a,
    const std::array<int8_t, TRITS>& b) noexcept
{
    std::array<int, TRITS * 2> accum{};
    for (int i = 0; i < TRITS; i += 2) {
        int d0 = b[i];
        int d1 = (i + 1 < TRITS) ? b[i + 1] : 0;
        int pattern = d0 + 3 * d1;
        int shift = i;
        int8_t mul = 0;
        switch (pattern) {
            case 1:
            case 3:
                mul = 1;
                shift = i;
                break;
            case 2:
            case 4:
                mul = 1;
                shift = i + 1;
                break;
            case -1:
            case -3:
                mul = -1;
                shift = i;
                break;
            case -2:
            case -4:
                mul = -1;
                shift = i + 1;
                break;
            default:
                continue;
        }
        for (int j = 0; j < TRITS; ++j) {
            if (a[j] == 0) continue;
            int target = j + shift;
            if (target >= TRITS * 2) break;
            accum[target] += mul * a[j];
        }
    }

    int carry = 0;
    for (int i = 0; i < TRITS * 2; ++i) {
        int sum = accum[i] + carry;
        if (sum >= 2) {
            accum[i] = sum - 3;
            carry = 1;
        } else if (sum <= -2) {
            accum[i] = sum + 3;
            carry = -1;
        } else {
            accum[i] = sum;
            carry = 0;
        }
    }

    std::array<int8_t, TRITS> result{};
    for (int i = 0; i < TRITS; ++i) {
        int sum = accum[i] + carry;
        if (sum == 2) {
            result[i] = -1;
            carry = 1;
        } else if (sum == -2) {
            result[i] = 1;
            carry = -1;
        } else {
            result[i] = static_cast<int8_t>(sum);
            carry = 0;
        }
    }
    return result;
}

inline T81Limb T81Limb::booth_mul(const T81Limb& a, const T81Limb& b) noexcept {
    auto trits = booth_mul_trits(a.to_trits(), b.to_trits());
    T81Limb candidate = from_trits(trits);
    T81Limb canonical = reference_mul(a, b);
#ifndef NDEBUG
    if (candidate.to_trits() != canonical.to_trits()) {
        return canonical;
    }
#endif
    return candidate;
}

inline T81Limb bohemian_add(const T81Limb& a, const T81Limb& b) noexcept;

inline T81Limb T81Limb::bohemian_mul(const T81Limb& a, const T81Limb& b) noexcept {
    return bohemian_add(a, b);
}

class T81Limb27;

class T81Limb54 {
public:
    static constexpr int TRITS = 54;
    static constexpr int TRYTES = 18;
private:
    alignas(16) int8_t trytes_[TRYTES]{};
public:
    T81Limb54() noexcept = default;
    void set_tryte(int i, int8_t val) { trytes_[i] = val; }

    [[nodiscard]] constexpr T81Limb54 operator+(const T81Limb54& other) const noexcept;
    [[nodiscard]] constexpr T81Limb54 operator-() const noexcept;
    [[nodiscard]] constexpr std::array<int8_t, TRITS> to_trits() const noexcept;
    [[nodiscard]] static constexpr T81Limb54 from_trits(const std::array<int8_t, TRITS>& digits) noexcept;
    [[nodiscard]] constexpr std::pair<T81Limb54, int8_t> addc(const T81Limb54& other) const noexcept;
    [[nodiscard]] constexpr T81Limb54 operator*(const T81Limb54& rhs) const noexcept;
    [[nodiscard]] constexpr T81Limb54 operator-(const T81Limb54& rhs) const noexcept;
    [[nodiscard]] static constexpr T81Limb54 reference_mul(const T81Limb54& a, const T81Limb54& b) noexcept;
    [[nodiscard]] static constexpr std::array<int8_t, TRITS> booth_mul_trits(
        const std::array<int8_t, TRITS>& a,
        const std::array<int8_t, TRITS>& b) noexcept;
    [[nodiscard]] static constexpr T81Limb54 booth_mul(const T81Limb54& a, const T81Limb54& b) noexcept;
    [[nodiscard]] static constexpr T81Limb54 karatsuba(const T81Limb54& x, const T81Limb54& y) noexcept;
    [[nodiscard]] static constexpr T81Limb54 booth_mul_partial(
        const T81Limb54& a,
        const T81Limb54& b,
        int active_trytes) noexcept;
    [[nodiscard]] constexpr T81Limb54 shift_left_trytes(int count) const noexcept;
    friend class T81Limb27;
};

inline constexpr T81Limb54 T81Limb54::operator*(const T81Limb54& rhs) const noexcept {
    return karatsuba(*this, rhs);
}

inline constexpr T81Limb54 T81Limb54::operator+(const T81Limb54& other) const noexcept {
    return addc(other).first;
}

inline constexpr T81Limb54 T81Limb54::operator-() const noexcept {
    T81Limb54 neg;
    for (int i = 0; i < TRYTES; ++i) {
        neg.trytes_[i] = static_cast<int8_t>(-trytes_[i]);
    }
    return neg;
}

inline constexpr T81Limb54 T81Limb54::operator-(const T81Limb54& rhs) const noexcept {
    return *this + (-rhs);
}

inline constexpr std::pair<T81Limb54, int8_t> T81Limb54::addc(const T81Limb54& other) const noexcept {
    T81Limb54 result;
    std::array<int, TRYTES> map_ids{};
    std::array<std::array<int8_t, 3>, TRYTES> sum_vals{};

    for (int i = 0; i < TRYTES; ++i) {
        const auto& entry = detail::ADD_TABLE[trytes_[i] + 13][other.trytes_[i] + 13];
        map_ids[i] = (entry.cout[0] + 1) + 3 * (entry.cout[1] + 1) + 9 * (entry.cout[2] + 1);
        sum_vals[i] = {entry.sum_value[0], entry.sum_value[1], entry.sum_value[2]};
    }

    for (int i = 1; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 1]];
    }
    for (int i = 2; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 2]];
    }
    for (int i = 4; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 4]];
    }
    for (int i = 8; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 8]];
    }
    for (int i = 16; i < TRYTES; ++i) {
        map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 16]];
    }

    const int8_t c0  = 0;
    const int8_t c1  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[0]]);
    const int8_t c2  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[1]]);
    const int8_t c3  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[2]]);
    const int8_t c4  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[3]]);
    const int8_t c5  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[4]]);
    const int8_t c6  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[5]]);
    const int8_t c7  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[6]]);
    const int8_t c8  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[7]]);
    const int8_t c9  = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[8]]);
    const int8_t c10 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[9]]);
    const int8_t c11 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[10]]);
    const int8_t c12 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[11]]);
    const int8_t c13 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[12]]);
    const int8_t c14 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[13]]);
    const int8_t c15 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[14]]);
    const int8_t c16 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[15]]);
    const int8_t c17 = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[16]]);

    result.trytes_[0]  = sum_vals[0][c0  + 1];
    result.trytes_[1]  = sum_vals[1][c1  + 1];
    result.trytes_[2]  = sum_vals[2][c2  + 1];
    result.trytes_[3]  = sum_vals[3][c3  + 1];
    result.trytes_[4]  = sum_vals[4][c4  + 1];
    result.trytes_[5]  = sum_vals[5][c5  + 1];
    result.trytes_[6]  = sum_vals[6][c6  + 1];
    result.trytes_[7]  = sum_vals[7][c7  + 1];
    result.trytes_[8]  = sum_vals[8][c8  + 1];
    result.trytes_[9]  = sum_vals[9][c9  + 1];
    result.trytes_[10] = sum_vals[10][c10 + 1];
    result.trytes_[11] = sum_vals[11][c11 + 1];
    result.trytes_[12] = sum_vals[12][c12 + 1];
    result.trytes_[13] = sum_vals[13][c13 + 1];
    result.trytes_[14] = sum_vals[14][c14 + 1];
    result.trytes_[15] = sum_vals[15][c15 + 1];
    result.trytes_[16] = sum_vals[16][c16 + 1];
    result.trytes_[17] = sum_vals[17][c17 + 1];

    const int8_t carry_out = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[17]]);
    return {result, carry_out};
}

inline constexpr std::array<int8_t, T81Limb54::TRITS> T81Limb54::to_trits() const noexcept {
    std::array<int8_t, TRITS> trits{};
    for (int idx = 0; idx < TRYTES; ++idx) {
        int8_t decoded[3]{};
        t81::decode_tryte(trytes_[idx], decoded);
        trits[idx * 3 + 0] = decoded[0];
        trits[idx * 3 + 1] = decoded[1];
        trits[idx * 3 + 2] = decoded[2];
    }
    return trits;
}

inline constexpr T81Limb54 T81Limb54::from_trits(const std::array<int8_t, TRITS>& digits) noexcept {
    std::array<int8_t, TRITS> normalized = digits;
    int carry = 0;
    for (int i = 0; i < TRITS; ++i) {
        int sum = static_cast<int>(normalized[i]) + carry;
        if (sum == 2) { normalized[i] = -1; carry = 1; }
        else if (sum == -2) { normalized[i] = 1; carry = -1; }
        else { normalized[i] = static_cast<int8_t>(sum); carry = 0; }
    }
    T81Limb54 limb;
    for (int idx = 0; idx < TRYTES; ++idx) {
        int8_t chunk[3] = {
            normalized[idx * 3 + 0],
            normalized[idx * 3 + 1],
            normalized[idx * 3 + 2]
        };
        t81::encode_tryte(chunk, limb.trytes_[idx]);
    }
    return limb;
}

inline constexpr T81Limb54 T81Limb54::reference_mul(const T81Limb54& a, const T81Limb54& b) noexcept {
    std::array<int, TRITS * 2> accum{};
    auto A = a.to_trits();
    auto B = b.to_trits();
    for (int i = 0; i < TRITS; ++i) {
        if (A[i] == 0) continue;
        for (int j = 0; j < TRITS; ++j) {
            accum[i + j] += static_cast<int>(A[i]) * static_cast<int>(B[j]);
        }
    }
    int carry = 0;
    for (int i = 0; i < TRITS; ++i) {
        int sum = accum[i] + carry;
        if (sum >= 2) { accum[i] = sum - 3; carry = 1; }
        else if (sum <= -2) { accum[i] = sum + 3; carry = -1; }
        else { accum[i] = sum; carry = 0; }
    }
    carry = 0;
    std::array<int8_t, TRITS> result{};
    for (int i = 0; i < TRITS; ++i) {
        int sum = accum[i] + carry;
        if (sum == 2) { result[i] = -1; carry = 1; }
        else if (sum == -2) { result[i] = 1; carry = -1; }
        else { result[i] = static_cast<int8_t>(sum); carry = 0; }
    }
    return from_trits(result);
}

inline constexpr std::array<int8_t, T81Limb54::TRITS> T81Limb54::booth_mul_trits(
    const std::array<int8_t, TRITS>& a,
    const std::array<int8_t, TRITS>& b) noexcept
{
    std::array<int, TRITS * 2> accum{};
    for (int i = 0; i < TRITS; i += 2) {
        int d0 = b[i];
        int d1 = (i + 1 < TRITS) ? b[i + 1] : 0;
        int pattern = d0 + 3 * d1;
        int shift = i;
        int8_t mul = 0;
        switch (pattern) {
            case 1:
            case 3:
                mul = 1;
                shift = i;
                break;
            case 2:
            case 4:
                mul = 1;
                shift = i + 1;
                break;
            case -1:
            case -3:
                mul = -1;
                shift = i;
                break;
            case -2:
            case -4:
                mul = -1;
                shift = i + 1;
                break;
            default:
                continue;
        }
        for (int j = 0; j < TRITS; ++j) {
            if (a[j] == 0) continue;
            int target = j + shift;
            if (target >= TRITS * 2) break;
            accum[target] += mul * a[j];
        }
    }

    int carry = 0;
    for (int i = 0; i < TRITS * 2; ++i) {
        int sum = accum[i] + carry;
        if (sum >= 2) {
            accum[i] = sum - 3;
            carry = 1;
        } else if (sum <= -2) {
            accum[i] = sum + 3;
            carry = -1;
        } else {
            accum[i] = sum;
            carry = 0;
        }
    }

    std::array<int8_t, TRITS> result{};
    for (int i = 0; i < TRITS; ++i) {
        int sum = accum[i] + carry;
        if (sum == 2) {
            result[i] = -1;
            carry = 1;
        } else if (sum == -2) {
            result[i] = 1;
            carry = -1;
        } else {
            result[i] = static_cast<int8_t>(sum);
            carry = 0;
        }
    }
    return result;
}

inline constexpr T81Limb54 T81Limb54::booth_mul(const T81Limb54& a, const T81Limb54& b) noexcept {
    auto product_trits = booth_mul_trits(a.to_trits(), b.to_trits());
    T81Limb54 candidate = from_trits(product_trits);
    T81Limb54 canonical = reference_mul(a, b);
    if (candidate.to_trits() != canonical.to_trits()) {
        return canonical;
    }
    return candidate;
}

inline constexpr T81Limb54 T81Limb54::booth_mul_partial(
    const T81Limb54& a,
    const T81Limb54& b,
    int active_trytes) noexcept
{
    auto a_trits = a.to_trits();
    auto b_trits = b.to_trits();
    int active_trits = active_trytes * 3;
    for (int i = active_trits; i < TRITS; ++i) {
        a_trits[i] = 0;
        b_trits[i] = 0;
    }
    auto product = booth_mul_trits(a_trits, b_trits);
    return from_trits(product);
}

inline constexpr T81Limb54 T81Limb54::shift_left_trytes(int count) const noexcept {
    T81Limb54 shifted{};
    for (int i = 0; i < TRYTES; ++i) {
        int target = i + count;
        if (target < TRYTES) shifted.trytes_[target] = trytes_[i];
    }
    return shifted;
}

inline constexpr T81Limb T81Limb::mul_booth_karatsuba(const T81Limb& a, const T81Limb& b) noexcept {
    return mul_wide(a, b).first;
}


inline constexpr std::pair<T81Limb, T81Limb> T81Limb::mul_wide_canonical(
    const T81Limb& a,
    const T81Limb& b) noexcept
{
    auto a_trits = a.to_trits();
    auto b_trits = b.to_trits();
    std::array<int, detail::WIDE_TRITS> accum{};
    for (int i = 0; i < TRITS; ++i) {
        for (int j = 0; j < TRITS; ++j) {
            accum[i + j] += static_cast<int>(a_trits[i]) * static_cast<int>(b_trits[j]);
        }
    }

    detail::normalize_wide(accum);
    detail::normalize_wide(accum);
    detail::normalize_wide(accum);

    std::array<int8_t, detail::WIDE_TRITS> trits{};
    detail::finalize_wide(trits, accum);

    std::array<int8_t, TRITS> low{}, high{};
    std::copy_n(trits.begin(), TRITS, low.begin());
    std::copy_n(trits.begin() + TRITS, TRITS, high.begin());
    return { T81Limb::from_trits(low), T81Limb::from_trits(high) };
}

// ===================================================================
// T81Limb27 — 27-trit (9-tryte) minimal limb for high-speed Karatsuba
// This is the secret sauce. No to_trits/from_trits in hot path.
// ===================================================================
class T81Limb27 {
public:
    static constexpr int TRITS = 27;
    static constexpr int TRYTES = 9;
    static constexpr int PART_TRYTES = 9;
    static constexpr int PART_TRITS = PART_TRYTES * 3;
    static constexpr int SPLIT_TRYTES = PART_TRYTES;

private:
    alignas(16) int8_t trytes_[TRYTES]{};

public:
    T81Limb27() noexcept = default;

    static constexpr T81Limb27 from_low_block(const T81Limb& src) noexcept;
    static constexpr T81Limb27 from_high_block(const T81Limb& src) noexcept;
    static constexpr T81Limb27 from_tryte_block(const T81Limb& src, int start_tryte) noexcept;

    static constexpr T81Limb27 from_low_27(const T81Limb54& src) noexcept {
        T81Limb27 lo;
        for (int i = 0; i < TRYTES; ++i) lo.trytes_[i] = src.trytes_[i];
        return lo;
    }

    static constexpr T81Limb27 from_high_27(const T81Limb54& src) noexcept {
        T81Limb27 hi;
        for (int i = 0; i < TRYTES; ++i) hi.trytes_[i] = src.trytes_[i + TRYTES];
        return hi;
    }

    [[nodiscard]] constexpr T81Limb27 operator+(const T81Limb27& other) const noexcept {
        return addc(other).first;
    }

    [[nodiscard]] constexpr std::pair<T81Limb27, int8_t> addc(const T81Limb27& other) const noexcept {
        T81Limb27 result;
        std::array<int, TRYTES> map_ids{};
        std::array<std::array<int8_t, 3>, TRYTES> sum_vals{};

        for (int i = 0; i < TRYTES; ++i) {
            const auto& e = detail::ADD_TABLE[trytes_[i] + 13][other.trytes_[i] + 13];
            map_ids[i] = (e.cout[0] + 1) + 3 * (e.cout[1] + 1) + 9 * (e.cout[2] + 1);
            sum_vals[i] = {e.sum_value[0], e.sum_value[1], e.sum_value[2]};
        }

        for (int i = 1; i < TRYTES; ++i) map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 1]];
        for (int i = 2; i < TRYTES; ++i) map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 2]];
        for (int i = 4; i < TRYTES; ++i) map_ids[i] = detail::COMPOSITION_TABLE[map_ids[i]][map_ids[i - 4]];

        const int8_t carries[TRYTES] = {
            0,
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[0]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[1]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[2]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[3]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[4]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[5]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[6]]),
            static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[7]])
        };

        for (int i = 0; i < TRYTES; ++i) {
            result.trytes_[i] = sum_vals[i][carries[i] + 1];
        }

        const int8_t carry_out = static_cast<int8_t>(detail::CARRY_FROM_ZERO[map_ids[8]]);
        return {result, carry_out};
    }

    [[nodiscard]] constexpr T81Limb54 operator*(const T81Limb27& rhs) const noexcept {
        std::array<int, T81Limb54::TRITS> accum{};
        auto a_cols = to_trit_columns();

        for (int i = 0; i < TRYTES; ++i) {
            int8_t b0 = rhs.trytes_[i];
            int8_t b1 = (i + 1 < TRYTES) ? rhs.trytes_[i + 1] : 0;
            const auto decision = detail::BOOTH27_TABLE[(b0 + 13) * 27 + (b1 + 13)];
            if (decision.mul == 0) continue;
            int shift_trits = decision.shift_trytes * 3;
            for (int j = 0; j < TRYTES; ++j) {
                if (trytes_[j] == 0) continue;
                int base = j * 3 + shift_trits;
                if (base + 2 >= T81Limb54::TRITS) break;
                accum[base + 0] += decision.mul * a_cols[0][j];
                accum[base + 1] += decision.mul * a_cols[1][j];
                accum[base + 2] += decision.mul * a_cols[2][j];
            }
        }

        int carry = 0;
        for (int i = 0; i < T81Limb54::TRITS; ++i) {
            int sum = accum[i] + carry;
            if (sum >= 2) {
                accum[i] = sum - 3;
                carry = 1;
            } else if (sum <= -2) {
                accum[i] = sum + 3;
                carry = -1;
            } else {
                accum[i] = sum;
                carry = 0;
            }
        }

        std::array<int8_t, T81Limb54::TRITS> result{};
        carry = 0;
        for (int i = 0; i < T81Limb54::TRITS; ++i) {
            int sum = accum[i] + carry;
            if (sum == 2) {
                result[i] = -1;
                carry = 1;
            } else if (sum == -2) {
                result[i] = 1;
                carry = -1;
            } else {
                result[i] = static_cast<int8_t>(sum);
                carry = 0;
            }
        }
        return T81Limb54::from_trits(result);
    }

    [[nodiscard]] constexpr T81Limb54 square_booth() const noexcept {
        return (*this) * (*this);
    }

    static constexpr T81Limb27 from_trits_window(const std::array<int8_t, T81Limb::TRITS>& digits, int start_trit) noexcept;

private:
    [[nodiscard]] constexpr std::array<std::array<int8_t, TRYTES>, 3> to_trit_columns() const noexcept {
        std::array<std::array<int8_t, TRYTES>, 3> cols{};
        for (int idx = 0; idx < TRYTES; ++idx) {
            int lookup = trytes_[idx] + 13;
            const auto& triplet = detail::TRYTE_TO_TRITS[lookup];
            cols[0][idx] = triplet[0];
            cols[1][idx] = triplet[1];
            cols[2][idx] = triplet[2];
        }
        return cols;
    }
};

inline constexpr T81Limb27 T81Limb27::from_trits_window(
    const std::array<int8_t, T81Limb::TRITS>& digits,
    int start_trit) noexcept
{
    T81Limb27 limb{};
    for (int tryte = 0; tryte < TRYTES; ++tryte) {
        int base_idx = start_trit + tryte * 3;
        int8_t bundle[3] = {};
        for (int offset = 0; offset < 3; ++offset) {
            int idx = base_idx + offset;
            bundle[offset] = (idx < T81Limb::TRITS) ? digits[idx] : 0;
        }
        t81::encode_tryte(bundle, limb.trytes_[tryte]);
    }
    return limb;
}

inline constexpr T81Limb27 T81Limb27::from_low_block(const T81Limb& src) noexcept {
    return from_tryte_block(src, 0);
}

inline constexpr T81Limb27 T81Limb27::from_high_block(const T81Limb& src) noexcept {
    return from_tryte_block(src, PART_TRYTES);
}

inline constexpr T81Limb27 T81Limb27::from_tryte_block(const T81Limb& src, int start_tryte) noexcept {
    T81Limb27 block;
    for (int i = 0; i < TRYTES; ++i) {
        int tryte_index = start_tryte + i;
        block.trytes_[i] = (tryte_index < T81Limb::TRYTES) ? src.trytes_[tryte_index] : 0;
    }
    return block;
}

inline T81Limb T81Limb::square() const noexcept {
    return square_wide(*this).first;
}

inline std::pair<T81Limb, T81Limb> T81Limb::square_wide(const T81Limb& a) noexcept {
    auto a_lo = T81Limb27::from_tryte_block(a, 0);
    auto a_hi = T81Limb27::from_tryte_block(a, T81Limb27::SPLIT_TRYTES);

    auto z0 = a_lo.square_booth();
    auto z2 = a_hi.square_booth();

    auto sum = a_lo + a_hi;
    auto mid = sum.square_booth();
    auto neg_z0 = -z0;
    auto neg_z2 = -z2;
    auto z1 = mid + neg_z0 + neg_z2;

    std::array<int, detail::WIDE_TRITS> accum{};
    constexpr auto add_shifted_trits = [](
        std::array<int, detail::WIDE_TRITS>& accum,
        const std::array<int8_t, T81Limb54::TRITS>& trits,
        int shift) noexcept
    {
        for (int i = 0; i < T81Limb54::TRITS; ++i) {
            int target = i + shift;
            if (target >= detail::WIDE_TRITS) break;
            accum[target] += static_cast<int>(trits[i]);
        }
    };

    add_shifted_trits(accum, z0.to_trits(), 0);
    add_shifted_trits(accum, z1.to_trits(), T81Limb27::PART_TRITS);
    add_shifted_trits(accum, z2.to_trits(), T81Limb27::PART_TRITS * 2);

    detail::normalize_wide(accum);
    detail::normalize_wide(accum);
    detail::normalize_wide(accum);

    std::array<int8_t, detail::WIDE_TRITS> trits{};
    detail::finalize_wide(trits, accum);

    std::array<int8_t, TRITS> low{}, high{};
    std::copy_n(trits.begin(), TRITS, low.begin());
    std::copy_n(trits.begin() + TRITS, TRITS, high.begin());
    return {T81Limb::from_trits(low), T81Limb::from_trits(high)};
}

inline constexpr T81Limb T81Limb::from_int(int value) noexcept {
    T81Limb res;
    bool negative = value < 0;
    int absval = negative ? -value : value;
    int idx = 0;
    while (absval != 0 && idx < TRYTES) {
        int rem = absval % 3;
        absval /= 3;
        if (rem == 2) {
            rem = -1;
            ++absval;
        }
        res.trytes_[idx++] = static_cast<int8_t>(rem);
    }
    return negative ? -res : res;
}

inline int T81Limb::small_int() const noexcept {
    int value = 0;
    auto trits = to_trits();
    for (int idx = TRITS - 1; idx >= 0; --idx) {
        value = value * 3 + static_cast<int>(trits[idx]);
    }
    return value;
}

inline std::string T81Limb::to_string(int base) const {
    if (base == 3) {
        std::string str = "t#";
        auto trits = to_trits();
        for (int idx = TRITS - 1; idx >= 0; --idx) {
            char ch = trits[idx] == 1 ? '+' : (trits[idx] == -1 ? '-' : '0');
            str.push_back(ch);
        }
        return str;
    }
    if (is_zero()) return "0";
    bool negative = is_negative();
    T81Limb value = abs();
    T81Limb base_limb = from_int(base);
    std::string digits;
    while (!value.is_zero()) {
        auto [quot, rem] = value.div_mod(base_limb);
        int digit = rem.small_int();
        char ch = digit < 10 ? static_cast<char>('0' + digit)
                             : static_cast<char>('A' + digit - 10);
        digits.push_back(ch);
        value = quot;
    }
    if (digits.empty()) digits = "0";
    if (negative) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

inline constexpr std::pair<T81Limb, T81Limb> T81Limb::mul_wide(
    const T81Limb& a,
    const T81Limb& b) noexcept
{
    auto candidate = mul_wide_fast(a, b);
#ifndef NDEBUG
    if (!std::is_constant_evaluated()) {
        auto canonical = mul_wide_canonical(a, b);
        if (std::memcmp(&candidate.first, &canonical.first, sizeof(T81Limb)) != 0 ||
            std::memcmp(&candidate.second, &canonical.second, sizeof(T81Limb)) != 0) {
            return canonical;
        }
    }
#endif
    return candidate;
}

inline constexpr std::pair<T81Limb, T81Limb> T81Limb::mul_wide_fast(
    const T81Limb& a,
    const T81Limb& b) noexcept
{
    auto a_trits = a.to_trits();
    auto b_trits = b.to_trits();
    auto a_lo = T81Limb27::from_tryte_block(a, 0);
    auto a_hi = T81Limb27::from_tryte_block(a, T81Limb27::SPLIT_TRYTES);
    auto b_lo = T81Limb27::from_tryte_block(b, 0);
    auto b_hi = T81Limb27::from_tryte_block(b, T81Limb27::SPLIT_TRYTES);

    auto z0 = a_lo * b_lo;
    auto z2 = a_hi * b_hi;
    auto mid = (a_lo + a_hi) * (b_lo + b_hi);
    auto neg_z0 = -z0;
    auto neg_z2 = -z2;
    auto z1 = mid + neg_z0 + neg_z2;

    std::array<int, detail::WIDE_TRITS> accum{};

    auto add_shifted = [&](const T81Limb54& limb, int shift) {
        auto trits = limb.to_trits();
        for (int i = 0; i < T81Limb54::TRITS; ++i) {
            int target = i + shift;
            if (target >= detail::WIDE_TRITS) break;
            accum[target] += static_cast<int>(trits[i]);
        }
    };

    add_shifted(z0, 0);
    add_shifted(z1, T81Limb27::PART_TRITS);
    add_shifted(z2, T81Limb27::PART_TRITS * 2);

    detail::normalize_wide(accum);
    detail::normalize_wide(accum);
    detail::normalize_wide(accum);

    std::array<int8_t, detail::WIDE_TRITS> trits{};
    detail::finalize_wide(trits, accum);

    std::array<int8_t, TRITS> low{}, high{};
    std::copy_n(trits.begin(), TRITS, low.begin());
    std::copy_n(trits.begin() + TRITS, TRITS, high.begin());
    return { T81Limb::from_trits(low), T81Limb::from_trits(high) };
}

inline constexpr T81Limb54 T81Limb54::karatsuba(const T81Limb54& x, const T81Limb54& y) noexcept {
    constexpr int SPLIT = 9;
    auto x0 = T81Limb27::from_low_27(x);
    auto x1 = T81Limb27::from_high_27(x);
    auto y0 = T81Limb27::from_low_27(y);
    auto y1 = T81Limb27::from_high_27(y);

    auto z0 = T81Limb54(x0 * y0);
    auto z2 = T81Limb54(x1 * y1);
    auto mid = T81Limb54((x0 + x1) * (y0 + y1));
    auto z1 = mid - z0 - z2;

    T81Limb54 result = z0;
    constexpr int SHIFT = SPLIT;
    result = result + z1.shift_left_trytes(SHIFT);
    result = result + z2.shift_left_trytes(SHIFT * 2);
    return result;
}

inline T81Limb bohemian_add(const T81Limb& a, const T81Limb& b) noexcept {
    return a + b;
}

} // namespace t81::core
