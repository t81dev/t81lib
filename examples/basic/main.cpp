#include <iostream>

#include "t81/core/T81Limb.hpp"

int main() {
    using t81::core::T81Limb;

    T81Limb a;
    a.set_tryte(0, 1);
    T81Limb b = T81Limb::one();

    auto sum = a + b;
    auto diff = sum - b;
    auto sum_trits = sum.to_trits();

    std::cout << "t81lib sample: first trit of sum = " << static_cast<int>(sum_trits[0]) << "\n";
    std::cout << "sum - b equals a: " << (diff.compare(a) == 0 ? "yes\n" : "no\n");

    return 0;
}
