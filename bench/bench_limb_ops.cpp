// bench/bench_limb_ops.cpp — Benchmark for limb arithmetic primitives.

#include <cstdint>
#include <random>

#include <benchmark/benchmark.h>

#include <t81/core/limb.hpp>
#include <t81/util/random.hpp>

namespace {

    enum class Operation : std::uint64_t {
        Add,
        Subtract,
        Multiply,
        Divide,
        Modulo,
    };

    constexpr bool needs_nonzero_rhs(Operation op) noexcept {
        return op == Operation::Divide || op == Operation::Modulo;
    }

    inline t81::core::limb random_nonzero_limb(std::mt19937_64 &rng) {
        t81::core::limb value;
        do {
            value = t81::util::random_limb(rng);
        } while (value.is_zero());
        return value;
    }

    constexpr std::uint64_t seed_offset(Operation op, bool binary) noexcept {
        return 0xa5a5a5a5a5a5a5a5ull + (static_cast<std::uint64_t>(op) << 4) +
               (binary ? 0x2u : 0x1u);
    }

    template <bool Binary>
    void bench_limb_operation(benchmark::State &state, Operation op, std::uint64_t seed) {
        std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
        while (state.KeepRunning()) {
            const auto lhs = t81::util::random_limb(rng);
            const auto rhs =
                needs_nonzero_rhs(op) ? random_nonzero_limb(rng) : t81::util::random_limb(rng);
            if constexpr (Binary) {
                const auto lhs_value = lhs.to_value();
                const auto rhs_value = rhs.to_value();
                t81::core::detail::limb_int128 result = 0;
                switch (op) {
                case Operation::Add:
                    result = lhs_value + rhs_value;
                    break;
                case Operation::Subtract:
                    result = lhs_value - rhs_value;
                    break;
                case Operation::Multiply:
                    result = lhs_value * rhs_value;
                    break;
                case Operation::Divide:
                    result = lhs_value / rhs_value;
                    break;
                case Operation::Modulo:
                    result = lhs_value % rhs_value;
                    break;
                }
                benchmark::DoNotOptimize(result);
            } else {
                t81::core::limb result;
                switch (op) {
                case Operation::Add:
                    result = lhs + rhs;
                    break;
                case Operation::Subtract:
                    result = lhs - rhs;
                    break;
                case Operation::Multiply:
                    result = lhs * rhs;
                    break;
                case Operation::Divide:
                    result = lhs / rhs;
                    break;
                case Operation::Modulo:
                    result = lhs % rhs;
                    break;
                }
                benchmark::DoNotOptimize(result);
            }
        }
    }

} // namespace

BENCHMARK_CAPTURE(bench_limb_operation<false>,
                  add_ternary,
                  Operation::Add,
                  seed_offset(Operation::Add, false));
BENCHMARK_CAPTURE(bench_limb_operation<true>,
                  add_binary,
                  Operation::Add,
                  seed_offset(Operation::Add, true));

BENCHMARK_CAPTURE(bench_limb_operation<false>,
                  subtract_ternary,
                  Operation::Subtract,
                  seed_offset(Operation::Subtract, false));
BENCHMARK_CAPTURE(bench_limb_operation<true>,
                  subtract_binary,
                  Operation::Subtract,
                  seed_offset(Operation::Subtract, true));

BENCHMARK_CAPTURE(bench_limb_operation<false>,
                  multiply_ternary,
                  Operation::Multiply,
                  seed_offset(Operation::Multiply, false));
BENCHMARK_CAPTURE(bench_limb_operation<true>,
                  multiply_binary,
                  Operation::Multiply,
                  seed_offset(Operation::Multiply, true));

BENCHMARK_CAPTURE(bench_limb_operation<false>,
                  divide_ternary,
                  Operation::Divide,
                  seed_offset(Operation::Divide, false));
BENCHMARK_CAPTURE(bench_limb_operation<true>,
                  divide_binary,
                  Operation::Divide,
                  seed_offset(Operation::Divide, true));

BENCHMARK_CAPTURE(bench_limb_operation<false>,
                  modulo_ternary,
                  Operation::Modulo,
                  seed_offset(Operation::Modulo, false));
BENCHMARK_CAPTURE(bench_limb_operation<true>,
                  modulo_binary,
                  Operation::Modulo,
                  seed_offset(Operation::Modulo, true));

BENCHMARK_MAIN();
