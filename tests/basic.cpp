#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "t81/packing.hpp"
#include "t81/core/T81Limb.hpp"

int main() {
    using namespace t81;
    using namespace t81::core;

    // packing helpers round-trip 8 trits through the compact representation.
    std::array<Trit, 8> trits_input = {1, 0, -1, 1, 1, 0, 0, -1};
    auto packed = packing::pack_trits(trits_input);
    auto unpacked = packing::unpack_trits<8>(packed);
    assert(unpacked == trits_input);

    // encode/decode tryte helpers round-trip a small balanced digit window.
    std::array<int8_t, 3> tryte_digits = {1, -1, 0};
    int8_t encoded_tryte = 0;
    encode_tryte(tryte_digits.data(), encoded_tryte);
    auto decoded = decode_tryte(encoded_tryte);
    assert(decoded == tryte_digits);

    std::array<int8_t, T81Limb::TRITS> left_digits{};
    left_digits[0] = 1;
    left_digits[1] = -1;
    left_digits[2] = 1;
    left_digits[3] = 1;

    std::array<int8_t, T81Limb::TRITS> right_digits{};
    right_digits[0] = 1;
    right_digits[1] = 1;
    right_digits[2] = 1;

    auto left = T81Limb::from_trits(left_digits);
    auto right = T81Limb::from_trits(right_digits);

    auto sum = left + right;
    auto sum_trits = sum.to_trits();
    std::array<int8_t, T81Limb::TRITS> naive_sum{};
    int carry = 0;
    for (size_t idx = 0; idx < T81Limb::TRITS; ++idx) {
        int acc = static_cast<int>(left_digits[idx]) + static_cast<int>(right_digits[idx]) + carry;
        if (acc > 1) {
            naive_sum[idx] = static_cast<int8_t>(acc - 3);
            carry = 1;
        } else if (acc < -1) {
            naive_sum[idx] = static_cast<int8_t>(acc + 3);
            carry = -1;
        } else {
            naive_sum[idx] = static_cast<int8_t>(acc);
            carry = 0;
        }
    }
    for (size_t idx = 0; idx < T81Limb::TRITS; ++idx) {
        assert(sum_trits[idx] == naive_sum[idx]);
    }

    auto canonical_product = T81Limb::reference_mul(left, right);
    auto booth_product = T81Limb::booth_mul(left, right);
    assert(booth_product.to_trits() == canonical_product.to_trits());

    std::cout << "basic checks passed\n";
    return 0;
}
