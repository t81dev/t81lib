#include <chrono>
#include <iostream>
#include <random>

#include <t81/t81lib.hpp>
#include <t81/io/format.hpp>
#include <t81/util/random.hpp>

int main() {
    std::mt19937_64 rng(0x5eedc0de);
    t81::core::limb accumulator = t81::core::limb::zero();
    t81::core::limb stride = t81::util::random_limb(rng);
    constexpr int iterations = 100'000;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        accumulator += stride;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "bench_limb_add: " << iterations << " additions in " << elapsed
              << " µs; result = " << t81::io::to_string(accumulator) << '\n';
    return 0;
}
