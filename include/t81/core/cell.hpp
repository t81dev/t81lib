#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <utility>
#include <algorithm>
#include <cmath>

namespace t81::core {

using Trit = int8_t;

class Cell {
public:
    static constexpr int TRITS = 5;
    static constexpr int64_t MIN = -121;
    static constexpr int64_t MAX = +121;

private:
    std::array<Trit, TRITS> t_{};

public:
    constexpr Cell() noexcept = default;

    static constexpr Cell from_int(int64_t v) {
        if (v < MIN || v > MAX) throw std::overflow_error("Cell overflow in from_int");
        Cell c;
        bool negative = v < 0;
        if (negative) v = -v;

        for (int i = 0; v != 0 && i < TRITS; ++i) {
            int rem = static_cast<int>(v % 3);
            if (rem == 2) {
                c.t_[i] = static_cast<Trit>(-1);
                v = v / 3 + 1;
            } else {
                c.t_[i] = static_cast<Trit>(rem - 1);
                v /= 3;
            }
        }
        if (negative) c = -c;
        return c;
    }

    [[nodiscard]] constexpr int64_t to_int() const noexcept {
        int64_t val = 0;
        for (int i = TRITS - 1; i >= 0; --i) {
            int8_t digit = static_cast<int8_t>(t_[i]);
            val = val * 3 + (digit > 0 ? 1 : (digit < 0 ? -1 : 0));
        }
        return val;
    }

    [[nodiscard]] constexpr Cell operator-() const noexcept {
        Cell neg;
        for (int i = 0; i < TRITS; ++i) {
            neg.t_[i] = static_cast<Trit>(-static_cast<int>(t_[i]));
        }
        return neg;
    }

    [[nodiscard]] constexpr Cell operator+(const Cell& o) const {
        Cell r;
        int carry = 0;
        for (int i = 0; i < TRITS; ++i) {
            int sum = static_cast<int>(t_[i]) + static_cast<int>(o.t_[i]) + carry;
            if (sum == 3)       { r.t_[i] = static_cast<Trit>(1);  carry =  1; }
            else if (sum == -3) { r.t_[i] = static_cast<Trit>(-1); carry = -1; }
            else if (sum == 2)  { r.t_[i] = static_cast<Trit>(-1); carry =  1; }
            else if (sum == -2) { r.t_[i] = static_cast<Trit>(1);  carry = -1; }
            else                { r.t_[i] = static_cast<Trit>(sum); carry = 0; }
        }
        if (carry) throw std::overflow_error("Cell addition overflow");
        return r;
    }

    [[nodiscard]] constexpr Cell operator-(const Cell& o) const {
        return *this + (-o);
    }

    [[nodiscard]] constexpr Cell operator*(const Cell& o) const {
        Cell result;
        for (int i = 0; i < TRITS; ++i) {
            if (o.t_[i] == static_cast<Trit>(1))      { result = result + (*this << i); }
            else if (o.t_[i] == static_cast<Trit>(-1)) { result = result - (*this << i); }
        }
        return result;
    }

    [[nodiscard]] constexpr Cell operator<<(int n) const {
        if (n < 0) throw std::domain_error("Negative shift");
        if (n >= TRITS) throw std::overflow_error("Shift overflow");
        Cell shifted;
        for (int i = 0; i < TRITS - n; ++i) {
            shifted.t_[i + n] = t_[i];
        }
        return shifted;
    }

    [[nodiscard]] constexpr Cell operator/(const Cell& divisor) const {
        if (divisor == Cell::zero()) throw std::domain_error("Division by zero");
        Cell quotient;
        Cell remainder = *this;
        Cell abs_div = divisor.abs();

        for (int i = TRITS - 1; i >= 0; --i) {
            Cell candidate = abs_div << i;
            if (candidate.to_int() <= std::abs(remainder.to_int())) {
                quotient = quotient + (Cell::one() << i);
                Cell sub = divisor.to_int() < 0 ? -candidate : candidate;
                remainder = remainder - sub;
            }
        }
        if ((this->to_int() < 0) != (divisor.to_int() < 0)) {
            quotient = -quotient;
        }
        return quotient;
    }

    [[nodiscard]] constexpr Cell operator%(const Cell& divisor) const {
        Cell q = *this / divisor;
        return *this - q * divisor;
    }

    [[nodiscard]] friend constexpr Cell gcd(Cell a, Cell b) {
        a = a.abs();
        b = b.abs();
        while (b != Cell::zero()) {
            Cell t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    [[nodiscard]] constexpr bool operator==(const Cell& o) const noexcept {
        return t_ == o.t_;
    }

    [[nodiscard]] constexpr bool operator!=(const Cell& o) const noexcept {
        return !(*this == o);
    }

    [[nodiscard]] constexpr bool operator<(const Cell& o) const noexcept {
        return to_int() < o.to_int();
    }

    [[nodiscard]] constexpr bool operator<=(const Cell& o) const noexcept {
        return to_int() <= o.to_int();
    }

    [[nodiscard]] constexpr bool operator>(const Cell& o) const noexcept {
        return to_int() > o.to_int();
    }

    [[nodiscard]] constexpr bool operator>=(const Cell& o) const noexcept {
        return to_int() >= o.to_int();
    }

    [[nodiscard]] constexpr Cell abs() const noexcept {
        return to_int() < 0 ? -(*this) : *this;
    }

    static constexpr Cell zero() noexcept { return Cell(); }
    static constexpr Cell one() noexcept { return Cell::from_int(1); }
    static constexpr Cell minus_one() noexcept { return Cell::from_int(-1); }
};

} // namespace t81::core

namespace t81 {

inline constexpr int8_t encode_tryte(const int8_t trits[3]) noexcept {
    return static_cast<int8_t>(
        trits[0] +
        trits[1] * 3 +
        trits[2] * 9
    );
}

inline constexpr void encode_tryte(const int8_t trits[3], int8_t& target) noexcept {
    target = encode_tryte(trits);
}

inline constexpr void decode_tryte(int8_t tryte, int8_t out_trits[3]) noexcept {
    int value = static_cast<int>(tryte) + 13;
    for (int idx = 0; idx < 3; ++idx) {
        out_trits[idx] = static_cast<int8_t>((value % 3) - 1);
        value /= 3;
    }
}

inline constexpr std::array<int8_t, 3> decode_tryte(int8_t tryte) noexcept {
    std::array<int8_t, 3> digits{};
    decode_tryte(tryte, digits.data());
    return digits;
}

} // namespace t81
