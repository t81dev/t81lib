// include/t81/io/fast_decimal.hpp — Fast decimal conversion helpers for I/O.

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <t81/core/bigint.hpp>
#include <t81/core/limb.hpp>

namespace t81::io {

    namespace detail {

        class DecimalAccumulator {
          public:
            static constexpr std::uint32_t BASE = 1'000'000'000u;

            DecimalAccumulator() {
                digits_.push_back(0);
            }

            void multiply_by_small(std::uint32_t factor) {
                if (factor == 0) {
                    digits_.assign(1, 0);
                    negative_ = false;
                    return;
                }
                std::uint64_t carry = 0;
                for (std::uint32_t &digit : digits_) {
                    const std::uint64_t product =
                        static_cast<std::uint64_t>(digit) * factor + carry;
                    digit = static_cast<std::uint32_t>(product % BASE);
                    carry = product / BASE;
                }
                if (carry != 0) {
                    digits_.push_back(static_cast<std::uint32_t>(carry));
                }
            }

            void multiply_by_power3(std::uint32_t exponent) {
                for (std::uint32_t i = 0; i < exponent; ++i) {
                    multiply_by_small(3);
                }
            }

            void add_abs_int(t81::core::detail::limb_int128 value) {
                if (value == 0) {
                    return;
                }
                std::uint64_t carry = 0;
                std::size_t index = 0;
                while (value != 0 || carry != 0) {
                    const std::uint64_t chunk = static_cast<std::uint64_t>(value % BASE);
                    value /= BASE;
                    if (index >= digits_.size()) {
                        digits_.push_back(0);
                    }
                    const std::uint64_t sum =
                        static_cast<std::uint64_t>(digits_[index]) + chunk + carry;
                    digits_[index] = static_cast<std::uint32_t>(sum % BASE);
                    carry = sum / BASE;
                    ++index;
                }
            }

            void add_signed_small(int value) {
                if (value == 0) {
                    return;
                }
                if (value < 0) {
                    if (negative_) {
                        add_abs_small(static_cast<std::uint32_t>(-value));
                    } else {
                        if (is_zero()) {
                            digits_.assign(1, static_cast<std::uint32_t>(-value));
                            negative_ = true;
                        } else if (subtract_small(static_cast<std::uint32_t>(-value))) {
                            negative_ = false;
                        } else {
                            negative_ = true;
                        }
                    }
                } else {
                    if (!negative_) {
                        add_abs_small(static_cast<std::uint32_t>(value));
                    } else {
                        if (is_zero()) {
                            digits_.assign(1, static_cast<std::uint32_t>(value));
                            negative_ = false;
                        } else if (subtract_small(static_cast<std::uint32_t>(value))) {
                            negative_ = true;
                        } else {
                            negative_ = false;
                        }
                    }
                }
            }

            std::string to_string() const {
                if (is_zero()) {
                    return "0";
                }
                std::string output;
                if (negative_) {
                    output.push_back('-');
                }
                for (std::size_t index = digits_.size(); index-- > 0;) {
                    if (index == digits_.size() - 1) {
                        output += std::to_string(digits_[index]);
                    } else {
                        char buffer[16];
                        std::snprintf(buffer, sizeof(buffer), "%09u", digits_[index]);
                        output += buffer;
                    }
                }
                return output;
            }

            bool is_zero() const {
                return digits_.empty() || (digits_.size() == 1 && digits_[0] == 0);
            }

            void set_negative(bool negative) {
                negative_ = negative;
                if (is_zero()) {
                    negative_ = false;
                }
            }

          private:
            void add_abs_small(std::uint32_t value) {
                std::uint64_t carry = value;
                std::size_t index = 0;
                while (carry != 0) {
                    if (index >= digits_.size()) {
                        digits_.push_back(0);
                    }
                    const std::uint64_t sum = static_cast<std::uint64_t>(digits_[index]) + carry;
                    digits_[index] = static_cast<std::uint32_t>(sum % BASE);
                    carry = sum / BASE;
                    ++index;
                }
            }

            bool subtract_small(std::uint32_t value) {
                if (is_zero()) {
                    digits_.assign(1, value);
                    return false;
                }
                std::uint64_t borrow = value;
                std::size_t index = 0;
                while (borrow != 0 && index < digits_.size()) {
                    if (digits_[index] >= borrow) {
                        digits_[index] -= static_cast<std::uint32_t>(borrow);
                        borrow = 0;
                    } else {
                        digits_[index] =
                            static_cast<std::uint32_t>(BASE - (borrow - digits_[index]));
                        borrow = 1;
                    }
                    ++index;
                }
                normalize();
                return borrow == 0;
            }

            void normalize() {
                while (digits_.size() > 1 && digits_.back() == 0) {
                    digits_.pop_back();
                }
            }

            std::vector<std::uint32_t> digits_;
            bool negative_ = false;
        };

    } // namespace detail

    inline std::string to_decimal(const t81::core::bigint &value) {
        if (value.is_zero()) {
            return "0";
        }
        const auto magnitude = value.abs();
        detail::DecimalAccumulator accumulator;
        for (std::size_t index = magnitude.limb_count(); index-- > 0;) {
            accumulator.multiply_by_power3(t81::core::limb::TRITS);
            const auto limb_value = magnitude.limb_at(index).to_value();
            accumulator.add_abs_int(limb_value);
        }
        accumulator.set_negative(value.is_negative());
        return accumulator.to_string();
    }

} // namespace t81::io
