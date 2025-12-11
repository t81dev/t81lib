// include/t81/t81lib.hpp — Umbrella header that exposes t81lib components.

#pragma once

// Umbrella header for t81lib.
// Users should generally include only this file.

#include <t81/core/limb.hpp>
#include <t81/core/bigint.hpp>
#include <t81/core/montgomery.hpp>
#include <t81/core/montgomery_helpers.hpp>
#include <t81/io/format.hpp>
#include <t81/io/parse.hpp>
#include <t81/util/random.hpp>
#include <t81/util/debug.hpp>

#include <algorithm>
#include <array>
#include <compare>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace t81 {

using Int = core::limb;
class Float {
public:
    Float() noexcept = default;
    explicit Float(core::limb mantissa, int exponent = 0) noexcept
        : mantissa_(std::move(mantissa)), exponent_(exponent) {
        normalize();
    }

    static Float zero() noexcept { return {}; }

    core::limb mantissa() const noexcept { return mantissa_; }
    int exponent() const noexcept { return exponent_; }
    bool is_zero() const noexcept { return mantissa_.is_zero(); }

    Float scaled_trits(int trits) const noexcept {
        return Float(mantissa_, exponent_ + trits);
    }

    Float scaled_trytes(int trytes) const noexcept {
        return scaled_trits(trytes * 3);
    }

    // Ensure mantissa drops trailing zero trits after multiplication.
    friend Float operator*(Float lhs, const Float& rhs) noexcept {
        lhs.mantissa_ *= rhs.mantissa_;
        lhs.exponent_ += rhs.exponent_;
        lhs.normalize();
        return lhs;
    }

    friend bool operator==(const Float& lhs, const Float& rhs) noexcept {
        return lhs.mantissa_ == rhs.mantissa_ && lhs.exponent_ == rhs.exponent_;
    }

private:
    void normalize() noexcept {
        if (mantissa_.is_zero()) {
            exponent_ = 0;
            mantissa_ = core::limb::zero();
            return;
        }
        const core::limb divisor = core::limb::from_value(3);
        while (true) {
            const auto [quotient, remainder] = core::limb::div_mod(mantissa_, divisor);
            if (!remainder.is_zero()) {
                break;
            }
            mantissa_ = quotient;
            --exponent_;
        }
    }
    core::limb mantissa_{core::limb::zero()};
    int exponent_ = 0;
};

class Ratio {
public:
    Ratio() noexcept = default;
    explicit Ratio(core::bigint numerator)
        : Ratio(std::move(numerator), core::bigint::one()) {}
    explicit Ratio(core::limb numerator)
        : Ratio(core::bigint(numerator)) {}
    Ratio(core::limb numerator, core::limb denominator)
        : Ratio(core::bigint(numerator), core::bigint(denominator)) {}
    Ratio(core::bigint numerator, core::bigint denominator) {
        normalize(std::move(numerator), std::move(denominator));
    }

    static Ratio zero() noexcept { return Ratio(); }

    const core::bigint& numerator() const noexcept { return numerator_; }
    const core::bigint& denominator() const noexcept { return denominator_; }
    bool is_zero() const noexcept { return numerator_.is_zero(); }

    friend Ratio operator+(const Ratio& lhs, const Ratio& rhs) {
        core::bigint numerator = lhs.numerator_ * rhs.denominator_ +
                                rhs.numerator_ * lhs.denominator_;
        core::bigint denominator = lhs.denominator_ * rhs.denominator_;
        return Ratio(std::move(numerator), std::move(denominator));
    }

    friend Ratio operator-(const Ratio& lhs, const Ratio& rhs) {
        core::bigint numerator = lhs.numerator_ * rhs.denominator_ -
                                rhs.numerator_ * lhs.denominator_;
        core::bigint denominator = lhs.denominator_ * rhs.denominator_;
        return Ratio(std::move(numerator), std::move(denominator));
    }

    friend Ratio operator*(const Ratio& lhs, const Ratio& rhs) {
        core::bigint numerator = lhs.numerator_ * rhs.numerator_;
        core::bigint denominator = lhs.denominator_ * rhs.denominator_;
        return Ratio(std::move(numerator), std::move(denominator));
    }

    friend Ratio operator/(const Ratio& lhs, const Ratio& rhs) {
        if (rhs.is_zero()) {
            throw std::domain_error("ratio division by zero");
        }
        core::bigint numerator = lhs.numerator_ * rhs.denominator_;
        core::bigint denominator = lhs.denominator_ * rhs.numerator_;
        return Ratio(std::move(numerator), std::move(denominator));
    }

    friend std::strong_ordering operator<=>(const Ratio& lhs, const Ratio& rhs) noexcept {
        const core::bigint left = lhs.numerator_ * rhs.denominator_;
        const core::bigint right = rhs.numerator_ * lhs.denominator_;
        return left <=> right;
    }

    friend bool operator==(const Ratio& lhs, const Ratio& rhs) noexcept {
        return lhs.numerator_ == rhs.numerator_ && lhs.denominator_ == rhs.denominator_;
    }

    friend bool operator!=(const Ratio& lhs, const Ratio& rhs) noexcept {
        return !(lhs == rhs);
    }

private:
    void normalize(core::bigint numerator, core::bigint denominator) {
        if (denominator.is_zero()) {
            throw std::domain_error("ratio denominator cannot be zero");
        }
        if (denominator.is_negative()) {
            numerator = -numerator;
            denominator = -denominator;
        }
        const core::bigint divisor = core::bigint::gcd(numerator.abs(), denominator);
        if (!divisor.is_zero()) {
            numerator /= divisor;
            denominator /= divisor;
        }
        numerator_ = std::move(numerator);
        denominator_ = std::move(denominator);
    }

    core::bigint numerator_{core::bigint::zero()};
    core::bigint denominator_{core::bigint::one()};
};

class Modulus {
public:
    explicit Modulus(core::limb modulus)
        : montgomery_context_(modulus),
          modulus_(montgomery_context_.modulus()),
          ternary_base_(core::limb::from_value(3)) {
    }

    const core::limb& modulus() const noexcept { return modulus_; }
 const core::MontgomeryContext<core::limb>& montgomery_context() const noexcept {
        return montgomery_context_;
    }
    core::limb to_montgomery(const core::limb& value) const {
        return montgomery_context_.to_montgomery(value);
    }
    core::limb from_montgomery(const core::limb& value) const {
        return montgomery_context_.from_montgomery(value);
    }
    core::limb mul(const core::limb& lhs, const core::limb& rhs) const {
        return montgomery_context_.mul(lhs, rhs);
    }
    core::limb pow3(std::size_t exponent) const {
        auto& cache = power_of_three_cache_for(modulus_);
        while (exponent >= cache.size()) {
            cache.push_back(modular_mul(cache.back(), ternary_base_));
        }
        return cache[exponent];
    }

private:
    static std::vector<core::limb>& power_of_three_cache_for(const core::limb& modulus) {
        thread_local std::unordered_map<core::limb, std::vector<core::limb>> cache_by_modulus;
        auto it = cache_by_modulus.try_emplace(modulus, std::vector<core::limb>{core::limb::one()}).first;
        return it->second;
    }

    core::limb modular_mul(const core::limb& lhs, const core::limb& rhs) const {
        return montgomery_context_.mul(lhs, rhs);
    }

    core::MontgomeryContext<core::limb> montgomery_context_;
    core::limb modulus_;
    core::limb ternary_base_;
};

class MontgomeryInt {
public:
    explicit MontgomeryInt(const Modulus& modulus)
        : modulus_(&modulus),
          value_(modulus.to_montgomery(core::limb::zero())) {}

    MontgomeryInt(const Modulus& modulus, core::limb plain)
        : modulus_(&modulus),
          value_(modulus.to_montgomery(reduce(plain))) {}

    const Modulus& modulus() const noexcept { return *modulus_; }

    core::limb to_limb() const {
        return modulus_->from_montgomery(value_);
    }

    MontgomeryInt& operator+=(const MontgomeryInt& other) {
        ensure_same_modulus(other);
        core::limb sum = value_ + other.value_;
        if (!(sum < modulus_->modulus())) {
            sum -= modulus_->modulus();
        }
        value_ = sum;
        return *this;
    }

    MontgomeryInt& operator-=(const MontgomeryInt& other) {
        ensure_same_modulus(other);
        core::limb diff = value_ - other.value_;
        if (diff.is_negative()) {
            diff += modulus_->modulus();
        }
        value_ = diff;
        return *this;
    }

    MontgomeryInt& operator*=(const MontgomeryInt& other) {
        ensure_same_modulus(other);
        value_ = modulus_->mul(value_, other.value_);
        return *this;
    }

    friend MontgomeryInt operator+(MontgomeryInt lhs, const MontgomeryInt& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend MontgomeryInt operator-(MontgomeryInt lhs, const MontgomeryInt& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend MontgomeryInt operator*(MontgomeryInt lhs, const MontgomeryInt& rhs) {
        lhs *= rhs;
        return lhs;
    }

private:
    core::limb reduce(core::limb value) const {
        while (!(value < modulus_->modulus())) {
            value -= modulus_->modulus();
        }
        return value;
    }

    void ensure_same_modulus(const MontgomeryInt& other) const {
        if (modulus_ != other.modulus_) {
            throw std::invalid_argument("MontgomeryInt modulus mismatch");
        }
    }

    const Modulus* modulus_;
    core::limb value_;
};

template <typename Component>
class Complex {
public:
    Complex() noexcept = default;
    Complex(Component real, Component imag) : real_(real), imag_(imag) {}

    const Component& real() const noexcept { return real_; }
    const Component& imag() const noexcept { return imag_; }

    Complex& operator+=(const Complex& other) {
        real_ += other.real_;
        imag_ += other.imag_;
        return *this;
    }

    Complex& operator-=(const Complex& other) {
        real_ -= other.real_;
        imag_ -= other.imag_;
        return *this;
    }

    Complex& operator*=(const Complex& other) {
        const Component real_part = real_ * other.real_ - imag_ * other.imag_;
        const Component imag_part = real_ * other.imag_ + imag_ * other.real_;
        real_ = real_part;
        imag_ = imag_part;
        return *this;
    }

    friend Complex operator+(Complex lhs, const Complex& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Complex operator-(Complex lhs, const Complex& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Complex operator*(Complex lhs, const Complex& rhs) {
        lhs *= rhs;
        return lhs;
    }

private:
    Component real_{};
    Component imag_{};
};

template <typename Coefficient>
class Polynomial {
public:
    Polynomial() noexcept = default;
    explicit Polynomial(std::vector<Coefficient> coeffs)
        : coeffs_(std::move(coeffs)) {
        normalize();
    }

    static Polynomial zero() noexcept { return Polynomial(); }
    static Polynomial constant(const Coefficient& value) {
        if (value == Coefficient{}) {
            return zero();
        }
        return Polynomial(std::vector<Coefficient>{value});
    }

    std::size_t degree() const noexcept {
        if (coeffs_.empty()) {
            return 0;
        }
        return coeffs_.size() - 1;
    }

    const std::vector<Coefficient>& coefficients() const noexcept {
        return coeffs_;
    }

    const Coefficient& operator[](std::size_t index) const {
        if (index >= coeffs_.size()) {
            static const Coefficient zero{};
            return zero;
        }
        return coeffs_[index];
    }

    Polynomial& operator+=(const Polynomial& other) {
        coeffs_.resize(std::max(coeffs_.size(), other.coeffs_.size()), Coefficient{});
        for (std::size_t index = 0; index < other.coeffs_.size(); ++index) {
            coeffs_[index] += other.coeffs_[index];
        }
        normalize();
        return *this;
    }

    Polynomial& operator-=(const Polynomial& other) {
        coeffs_.resize(std::max(coeffs_.size(), other.coeffs_.size()), Coefficient{});
        for (std::size_t index = 0; index < other.coeffs_.size(); ++index) {
            coeffs_[index] -= other.coeffs_[index];
        }
        normalize();
        return *this;
    }

    Polynomial& operator*=(const Polynomial& other) {
        if (coeffs_.empty() || other.coeffs_.empty()) {
            coeffs_.clear();
            return *this;
        }
        std::vector<Coefficient> product(coeffs_.size() + other.coeffs_.size() - 1,
                                         Coefficient{});
        for (std::size_t lhs = 0; lhs < coeffs_.size(); ++lhs) {
            for (std::size_t rhs = 0; rhs < other.coeffs_.size(); ++rhs) {
                product[lhs + rhs] += coeffs_[lhs] * other.coeffs_[rhs];
            }
        }
        coeffs_ = std::move(product);
        normalize();
        return *this;
    }

    Coefficient evaluate(const Coefficient& point) const {
        Coefficient result{};
        for (std::size_t index = coeffs_.size(); index-- > 0;) {
            result = result * point + coeffs_[index];
        }
        return result;
    }

    friend Polynomial operator+(Polynomial lhs, const Polynomial& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Polynomial operator-(Polynomial lhs, const Polynomial& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Polynomial operator*(Polynomial lhs, const Polynomial& rhs) {
        lhs *= rhs;
        return lhs;
    }

private:
    void normalize() {
        while (!coeffs_.empty() && coeffs_.back() == Coefficient{}) {
            coeffs_.pop_back();
        }
    }

    std::vector<Coefficient> coeffs_;
};

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

    const core::bigint& modulus() const noexcept { return modulus_; }

    core::bigint add(const core::bigint& lhs, const core::bigint& rhs) const {
        return lhs ^ rhs;
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
            value = value ^ shift_left(modulus_, shift);
            deg = bit_degree(value);
        }
        return value;
    }

private:
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

    core::bigint shift_left(const core::bigint& value, std::size_t bits) const {
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
                result = result ^ lhs;
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

template <int N>
class Fixed {
public:
    static_assert(N > 0, "Fixed requires positive trit width");

    Fixed() noexcept = default;
    explicit Fixed(core::bigint value) { value_ = normalize(std::move(value)); }

    static constexpr int trits = N;

    core::bigint to_bigint() const noexcept { return value_; }

    Fixed& operator+=(const Fixed& other) {
        value_ = normalize(value_ + other.value_);
        return *this;
    }

    Fixed& operator-=(const Fixed& other) {
        value_ = normalize(value_ - other.value_);
        return *this;
    }

    Fixed& operator*=(const Fixed& other) {
        value_ = normalize(value_ * other.value_);
        return *this;
    }

    friend Fixed operator+(Fixed lhs, const Fixed& rhs) {
        lhs += rhs;
        return lhs;
    }

    friend Fixed operator-(Fixed lhs, const Fixed& rhs) {
        lhs -= rhs;
        return lhs;
    }

    friend Fixed operator*(Fixed lhs, const Fixed& rhs) {
        lhs *= rhs;
        return lhs;
    }

private:
    static const core::bigint& modulus() {
        static core::bigint value = pow3(trits);
        return value;
    }

    static core::bigint pow3(int count) {
        if (count < 0 || count > trits) {
            throw std::out_of_range("Fixed pow3 count out of range");
        }
        static const auto cache = [] {
            std::array<core::bigint, trits + 1> values{};
            values[0] = core::bigint::one();
            for (int index = 1; index <= trits; ++index) {
                values[index] = values[index - 1] * core::bigint(3);
            }
            return values;
        }();
        return cache[count];
    }

    static core::bigint normalize(core::bigint value) {
        const auto [quotient, remainder] = core::bigint::div_mod(value, modulus());
        (void)quotient;
        core::bigint positive = remainder;
        if (positive.is_negative()) {
            positive += modulus();
        }
        const core::bigint half = modulus() / core::bigint(2);
        if (positive > half) {
            positive -= modulus();
        }
        return positive;
    }

    core::bigint value_{core::bigint::zero()};
};

inline constexpr int T81LIB_VERSION_MAJOR = 0;
inline constexpr int T81LIB_VERSION_MINOR = 1;
inline constexpr int T81LIB_VERSION_PATCH = 0;

} // namespace t81
