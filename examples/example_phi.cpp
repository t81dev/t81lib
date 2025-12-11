#include <iostream>
#include <string_view>

#include <t81/t81lib.hpp>

using namespace t81::literals;

int main() {
    const std::string_view phi_literal(" 1.011011011101 ");
    const auto phi = t81::Float::from_string(t81::trim(phi_literal));
    const auto phi_square = phi * phi;

    std::cout << "φ ≈ " << t81::to_string(phi) << '\n';
    std::cout << "φ × φ ≈ " << t81::to_string(phi_square) << '\n';
    std::cout << "The result lands near \"10.000...\" ternary because φ² ≈ 2 + φ\n";
    return 0;
}
