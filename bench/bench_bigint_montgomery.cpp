#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string_view>

#include <t81/t81lib.hpp>

namespace {

template <typename Func>
void run_bench(std::string_view label, int iterations, Func&& work) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        work();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                              .count();
    std::cout << std::left << std::setw(32) << label << " "
              << iterations << " iterations"
              << " → " << duration << " µs\n";
}

} // namespace

int main() {
    std::mt19937_64 rng(0x5eedf00d);
    std::uniform_int_distribution<int> tryte_dist(0, t81::core::limb::TRYTES - 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);

    auto random_limb = [&](t81::core::limb::tryte_t tryte) {
        t81::core::limb value;
        value.set_tryte(tryte, t81::core::limb::tryte_t(tryte_dist(rng)));
        return value;
    };

    const auto limb_rand = [&]() {
        t81::core::limb value;
        for (int i = 0; i < t81::core::limb::TRYTES; ++i) {
            value.set_tryte(i, static_cast<t81::core::limb::tryte_t>(tryte_dist(rng)));
        }
        if (sign_dist(rng) == 1) {
            value = -value;
        }
        return value;
    };

    run_bench("limb add", 50'000, [&]() {
        auto a = limb_rand();
        auto b = limb_rand();
        auto result = a + b;
        (void)result;
    });

    run_bench("limb mul", 25'000, [&]() {
        auto a = limb_rand();
        auto b = limb_rand();
        try {
            auto _ = a * b;
            (void)_;
        } catch (...) {
        }
    });

    run_bench("bigint div_mod", 5'000, [&]() {
        auto a = t81::util::random_bigint(rng, 3, false);
        auto b = t81::util::random_bigint(rng, 2, false);
        if (b.is_zero()) {
            b = t81::core::bigint(1);
        }
        volatile auto result = t81::core::bigint::div_mod(a, b);
        (void)result;
    });

    const auto modulus = t81::core::bigint(197);
    const t81::core::MontgomeryContext<t81::core::bigint> ctx(modulus);

    run_bench("montgomery mul", 2'000, [&]() {
        auto a = t81::util::random_bigint(rng, 2, false);
        auto b = t81::util::random_bigint(rng, 2, false);
        if (a.is_negative()) {
            a = -a;
        }
        if (b.is_negative()) {
            b = -b;
        }
        auto ma = ctx.to_montgomery(a);
        auto mb = ctx.to_montgomery(b);
        auto result = ctx.mul(ma, mb);
        (void)result;
    });

    run_bench("montgomery pow", 1'000, [&]() {
        auto base = t81::util::random_bigint(rng, 2, false);
        if (base.is_negative()) {
            base = -base;
        }
        auto mb = ctx.to_montgomery(base);
        auto result = ctx.pow(mb, t81::core::bigint(5));
        (void)result;
    });

    return 0;
}
