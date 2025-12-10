// tests/unit/test_bigint_basic.cpp — Unit tests for basic bigint properties.

#include <t81/t81lib.hpp>
#include <iostream>

int main() {
    t81::core::bigint a;
    t81::core::bigint b;
    if (!(a == b)) {
        std::cerr << "bigint equality failed in skeleton test\n";
        return 1;
    }
    std::cout << "bigint_basic skeleton test passed\n";
    return 0;
}
