#include <t81/t81lib.hpp>
#include <iostream>

int main() {
    t81::core::limb a;
    t81::core::limb b;
    if (!(a == b)) {
        std::cerr << "limb equality failed in skeleton test\n";
        return 1;
    }
    std::cout << "limb_basic skeleton test passed\n";
    return 0;
}
