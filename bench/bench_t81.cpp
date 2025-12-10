// bench/bench_t81.cpp — Benchmark exercising the core bigint paths.

#include <array>
#include <cstdint>
#include <algorithm>
#include <random>
#include <string>

#include <benchmark/benchmark.h>

#include <t81/t81lib.hpp>
#include <t81/core/detail/base_digits.hpp>
#include <t81/io/fast_decimal.hpp>
#include <t81/util/random.hpp>

#include <utility>
#include <vector>
#include <stdexcept>

namespace {

enum class ArithmeticOp : std::uint64_t {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
};

enum class CompareOp : std::uint64_t {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
};

constexpr std::uint64_t seed_offset(ArithmeticOp op, bool bigint) noexcept {
    return 0x1b1b1b1bULL + (static_cast<std::uint64_t>(op) << 4) +
           (bigint ? 0x100U : 0U);
}

constexpr std::uint64_t seed_offset(CompareOp op, bool bigint) noexcept {
    return 0x3c3c3c3cULL + (static_cast<std::uint64_t>(op) << 4) +
           (bigint ? 0x200U : 0U);
}

constexpr bool requires_nonzero_rhs(ArithmeticOp op) noexcept {
    return op == ArithmeticOp::Divide || op == ArithmeticOp::Modulo;
}

template <typename Int>
Int random_value(std::mt19937_64& rng, std::size_t limbs);

template <>
t81::core::limb random_value<t81::core::limb>(std::mt19937_64& rng,
                                               std::size_t) {
    return t81::util::random_limb(rng);
}

template <>
t81::core::bigint random_value<t81::core::bigint>(std::mt19937_64& rng,
                                                   std::size_t limbs) {
    return t81::util::random_bigint(rng, limbs, true);
}

template <typename Int>
Int random_nonzero_value(std::mt19937_64& rng, std::size_t limbs) {
    Int value;
    do {
        value = random_value<Int>(rng, limbs);
    } while (value.is_zero());
    return value;
}

constexpr std::size_t kCacheSize = 64;

template <typename Int>
std::vector<Int> create_random_cache(std::mt19937_64& rng,
                                     std::size_t limbs,
                                     std::size_t count,
                                     bool nonzero) {
    std::vector<Int> cache;
    cache.reserve(count);
    while (cache.size() < count) {
        Int value = random_value<Int>(rng, limbs);
        if (nonzero && value.is_zero()) {
            continue;
        }
        cache.push_back(std::move(value));
    }
    return cache;
}

template <typename Int>
Int next_cache_value(const std::vector<Int>& cache, std::size_t& index) {
    const Int value = cache[index];
    index = (index + 1) % cache.size();
    return value;
}

std::string decimal_string(std::size_t digits) {
    std::string result;
    result.reserve(digits);
    for (std::size_t index = 0; index < digits; ++index) {
        result.push_back(static_cast<char>('1' + (index % 9)));
    }
    return result;
}

std::string base81_string(std::size_t digits) {
    std::string result;
    result.reserve(digits);
    const auto& alphabet = t81::core::detail::BASE81_DIGITS;
    for (std::size_t index = 0; index < digits; ++index) {
        result.push_back(alphabet[index % alphabet.size()]);
    }
    return result;
}

t81::core::bigint build_large_bigint(std::size_t limbs) {
    t81::core::bigint result = t81::core::bigint::zero();
    for (std::size_t index = 0; index < limbs; ++index) {
        result = result.shift_limbs(1);
        result += t81::core::bigint(t81::core::limb::max());
    }
    return result;
}

t81::core::bigint bigint_pow(t81::core::bigint base, std::uint64_t exponent) {
    t81::core::bigint result = t81::core::bigint::one();
    while (exponent != 0) {
        if ((exponent & 1) != 0) {
            result *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    return result;
}

t81::core::bigint bigint_gcd(t81::core::bigint a, t81::core::bigint b) {
    if (a.is_zero()) {
        return b.abs();
    }
    if (b.is_zero()) {
        return a.abs();
    }
    while (!b.is_zero()) {
        const auto remainder = t81::core::bigint::div_mod(a, b).second;
        a = b;
        b = remainder;
    }
    return a.abs();
}

std::vector<std::uint8_t> serialize_bigint(const t81::core::bigint& value) {
    std::vector<std::uint8_t> output;
    output.reserve(value.limb_count() * t81::core::limb::BYTES);
    for (std::size_t index = 0; index < value.limb_count(); ++index) {
        const auto chunk = value.limb_at(index).to_bytes();
        output.insert(output.end(), chunk.begin(), chunk.end());
    }
    return output;
}

t81::core::bigint deserialize_bigint(const std::vector<std::uint8_t>& data) {
    t81::core::bigint result = t81::core::bigint::zero();
    const auto chunk_size = t81::core::limb::BYTES;
    const std::size_t chunks = data.size() / chunk_size;
    for (std::size_t index = chunks; index-- > 0;) {
        std::array<std::uint8_t, t81::core::limb::BYTES> chunk{};
        std::copy_n(data.data() + index * chunk_size, chunk_size, chunk.data());
        result = result.shift_limbs(1);
        result += t81::core::bigint(t81::core::limb::from_bytes(chunk));
    }
    return result;
}

template <typename Int>
Int apply_operation(const Int& lhs, const Int& rhs, ArithmeticOp op);

template <>
t81::core::limb apply_operation<t81::core::limb>(const t81::core::limb& lhs,
                                                  const t81::core::limb& rhs,
                                                  ArithmeticOp op) {
    switch (op) {
    case ArithmeticOp::Add:
        return lhs + rhs;
    case ArithmeticOp::Subtract:
        return lhs - rhs;
    case ArithmeticOp::Multiply:
        return lhs * rhs;
    case ArithmeticOp::Divide:
        return lhs / rhs;
    case ArithmeticOp::Modulo:
        return lhs % rhs;
    }
    return lhs;
}

template <>
t81::core::bigint apply_operation<t81::core::bigint>(const t81::core::bigint& lhs,
                                                      const t81::core::bigint& rhs,
                                                      ArithmeticOp op) {
    switch (op) {
    case ArithmeticOp::Add:
        return lhs + rhs;
    case ArithmeticOp::Subtract:
        return lhs - rhs;
    case ArithmeticOp::Multiply:
        return lhs * rhs;
    case ArithmeticOp::Divide:
        return lhs / rhs;
    case ArithmeticOp::Modulo:
        return lhs % rhs;
    }
    return lhs;
}

template <typename Int>
void bench_negation(benchmark::State& state,
                    std::uint64_t seed,
                    std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<Int>(rng, limbs, kCacheSize, false);
    std::size_t cache_index = 0;
    while (state.KeepRunning()) {
        auto value = next_cache_value(cache, cache_index);
        try {
            auto result = -value;
            benchmark::DoNotOptimize(std::move(result));
        } catch (const std::overflow_error&) {
            continue;
        }
    }
}

template <typename Int>
bool apply_compare(const Int& lhs, const Int& rhs, CompareOp op);

template <>
bool apply_compare<t81::core::limb>(const t81::core::limb& lhs,
                                     const t81::core::limb& rhs,
                                     CompareOp op) {
    switch (op) {
    case CompareOp::Equal:
        return lhs == rhs;
    case CompareOp::NotEqual:
        return lhs != rhs;
    case CompareOp::Less:
        return lhs < rhs;
    case CompareOp::LessEqual:
        return lhs <= rhs;
    case CompareOp::Greater:
        return lhs > rhs;
    case CompareOp::GreaterEqual:
        return lhs >= rhs;
    }
    return false;
}

template <>
bool apply_compare<t81::core::bigint>(const t81::core::bigint& lhs,
                                       const t81::core::bigint& rhs,
                                       CompareOp op) {
    switch (op) {
    case CompareOp::Equal:
        return lhs == rhs;
    case CompareOp::NotEqual:
        return lhs != rhs;
    case CompareOp::Less:
        return lhs < rhs;
    case CompareOp::LessEqual:
        return lhs <= rhs;
    case CompareOp::Greater:
        return lhs > rhs;
    case CompareOp::GreaterEqual:
        return lhs >= rhs;
    }
    return false;
}

template <typename Int>
void bench_compare(benchmark::State& state,
                   CompareOp op,
                   std::uint64_t seed,
                   std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto lhs_cache = create_random_cache<Int>(rng, limbs, kCacheSize, false);
    const auto rhs_cache = create_random_cache<Int>(rng, limbs, kCacheSize, false);
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    while (state.KeepRunning()) {
        const auto lhs = next_cache_value(lhs_cache, lhs_index);
        const auto rhs = next_cache_value(rhs_cache, rhs_index);
        const auto result = apply_compare<Int>(lhs, rhs, op);
        benchmark::DoNotOptimize(static_cast<int>(result));
    }
}

template <typename Int>
void bench_arithmetic(benchmark::State& state,
                      ArithmeticOp op,
                      std::uint64_t seed,
                      std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto lhs_cache = create_random_cache<Int>(rng, limbs, kCacheSize, false);
    const auto rhs_cache = create_random_cache<Int>(rng, limbs, kCacheSize, requires_nonzero_rhs(op));
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    while (state.KeepRunning()) {
        const auto lhs = next_cache_value(lhs_cache, lhs_index);
        const auto rhs = next_cache_value(rhs_cache, rhs_index);
        try {
            auto result = apply_operation<Int>(lhs, rhs, op);
            benchmark::DoNotOptimize(std::move(result));
        } catch (const std::overflow_error&) {
            continue;
        }
    }
}

constexpr std::size_t kBigintLimbs = 3;
constexpr std::size_t kLargeBigintLimbs = 32;
constexpr std::uint64_t kLargePowExponent = 32;

constexpr std::uint64_t negation_seed(bool bigint) noexcept {
    return 0x2d2d2d2dULL + (bigint ? 0x300U : 0U);
}

template <bool Left>
void bench_bigint_shift(benchmark::State& state,
                        std::uint64_t seed,
                        std::size_t limbs,
                        int shift) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto value = next_cache_value(cache, index);
        t81::core::bigint result = Left ? (value << shift) : (value >> shift);
        benchmark::DoNotOptimize(result);
    }
}

void bench_bigint_decimal_to_string(benchmark::State& state,
                                    const t81::core::bigint& value) {
    while (state.KeepRunning()) {
        auto text = t81::io::to_decimal(value);
        benchmark::DoNotOptimize(std::move(text));
    }
}

void bench_bigint_decimal_from_string(benchmark::State& state,
                                      const std::string& text) {
    while (state.KeepRunning()) {
        auto parsed = t81::io::from_string<t81::core::bigint>(text, 10);
        benchmark::DoNotOptimize(std::move(parsed));
    }
}

void bench_bigint_base81_to_string(benchmark::State& state,
                                   const t81::core::bigint& value) {
    while (state.KeepRunning()) {
        auto text = t81::io::to_string(value, 81);
        benchmark::DoNotOptimize(std::move(text));
    }
}

void bench_bigint_base81_from_string(benchmark::State& state,
                                     const std::string& text) {
    while (state.KeepRunning()) {
        auto parsed = t81::io::from_string<t81::core::bigint>(text, 81);
        benchmark::DoNotOptimize(std::move(parsed));
    }
}

void bench_bigint_multiply_large(benchmark::State& state,
                                 std::uint64_t seed,
                                 std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto lhs_cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    const auto rhs_cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    while (state.KeepRunning()) {
        const auto lhs = next_cache_value(lhs_cache, lhs_index);
        const auto rhs = next_cache_value(rhs_cache, rhs_index);
        auto result = lhs * rhs;
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_bigint_pow(benchmark::State& state,
                      std::uint64_t seed,
                      std::size_t limbs,
                      std::uint64_t exponent) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto base = next_cache_value(cache, index);
        auto result = bigint_pow(base.abs(), exponent);
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_bigint_pow_large(benchmark::State& state,
                            std::uint64_t seed,
                            std::size_t limbs,
                            std::uint64_t exponent) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto base = next_cache_value(cache, index);
        auto result = bigint_pow(base.abs(), exponent);
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_bigint_gcd(benchmark::State& state,
                      std::uint64_t seed,
                      std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, true);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto a = next_cache_value(cache, index);
        const auto b = next_cache_value(cache, index);
        auto result = bigint_gcd(a.abs(), b.abs());
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_bigint_gcd_large(benchmark::State& state,
                            std::uint64_t seed,
                            std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, true);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto a = next_cache_value(cache, index);
        const auto b = next_cache_value(cache, index);
        auto result = bigint_gcd(a.abs(), b.abs());
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_bigint_serialization(benchmark::State& state,
                                std::uint64_t seed,
                                std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto cache = create_random_cache<t81::core::bigint>(rng, limbs, kCacheSize, false);
    std::size_t index = 0;
    while (state.KeepRunning()) {
        const auto value = next_cache_value(cache, index);
        const auto serialized = serialize_bigint(value);
        auto decoded = deserialize_bigint(serialized);
        benchmark::DoNotOptimize(std::move(decoded));
    }
}

void bench_random_bigint(benchmark::State& state, std::uint64_t seed, std::size_t limbs) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    while (state.KeepRunning()) {
        auto value = t81::util::random_bigint(rng, limbs, true);
        benchmark::DoNotOptimize(std::move(value));
    }
}

void bench_montgomery_mul(benchmark::State& state,
                          std::uint64_t seed,
                          const t81::core::bigint& modulus) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto context = t81::core::MontgomeryContext<t81::core::bigint>(modulus);
    auto lhs_cache = create_random_cache<t81::core::bigint>(rng, 2, kCacheSize, false);
    auto rhs_cache = create_random_cache<t81::core::bigint>(rng, 2, kCacheSize, false);
    for (auto& value : lhs_cache) {
        value = context.to_montgomery(value);
    }
    for (auto& value : rhs_cache) {
        value = context.to_montgomery(value);
    }
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    while (state.KeepRunning()) {
        const auto lhs = next_cache_value(lhs_cache, lhs_index);
        const auto rhs = next_cache_value(rhs_cache, rhs_index);
        auto result = context.mul(lhs, rhs);
        benchmark::DoNotOptimize(std::move(result));
    }
}

void bench_montgomery_mul_large(benchmark::State& state,
                                std::uint64_t seed,
                                const t81::core::bigint& modulus) {
    std::mt19937_64 rng(seed + static_cast<std::uint64_t>(state.thread_index()));
    const auto context = t81::core::MontgomeryContext<t81::core::bigint>(modulus);
    auto lhs_cache =
        create_random_cache<t81::core::bigint>(rng, kLargeBigintLimbs, kCacheSize, false);
    auto rhs_cache =
        create_random_cache<t81::core::bigint>(rng, kLargeBigintLimbs, kCacheSize, false);
    for (auto& value : lhs_cache) {
        value = context.to_montgomery(value);
    }
    for (auto& value : rhs_cache) {
        value = context.to_montgomery(value);
    }
    std::size_t lhs_index = 0;
    std::size_t rhs_index = 0;
    while (state.KeepRunning()) {
        const auto lhs = next_cache_value(lhs_cache, lhs_index);
        const auto rhs = next_cache_value(rhs_cache, rhs_index);
        auto result = context.mul(lhs, rhs);
        benchmark::DoNotOptimize(std::move(result));
    }
}

} // namespace

BENCHMARK_CAPTURE(bench_arithmetic<t81::core::limb>, limb_add, ArithmeticOp::Add,
                  seed_offset(ArithmeticOp::Add, false), 0);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::limb>, limb_subtract, ArithmeticOp::Subtract,
                  seed_offset(ArithmeticOp::Subtract, false), 0);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::limb>, limb_multiply, ArithmeticOp::Multiply,
                  seed_offset(ArithmeticOp::Multiply, false), 0);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::limb>, limb_divide, ArithmeticOp::Divide,
                  seed_offset(ArithmeticOp::Divide, false), 0);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::limb>, limb_modulo, ArithmeticOp::Modulo,
                  seed_offset(ArithmeticOp::Modulo, false), 0);

BENCHMARK_CAPTURE(bench_arithmetic<t81::core::bigint>, bigint_add, ArithmeticOp::Add,
                  seed_offset(ArithmeticOp::Add, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::bigint>, bigint_subtract, ArithmeticOp::Subtract,
                  seed_offset(ArithmeticOp::Subtract, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::bigint>, bigint_multiply, ArithmeticOp::Multiply,
                  seed_offset(ArithmeticOp::Multiply, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::bigint>, bigint_divide, ArithmeticOp::Divide,
                  seed_offset(ArithmeticOp::Divide, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_arithmetic<t81::core::bigint>, bigint_modulo, ArithmeticOp::Modulo,
                  seed_offset(ArithmeticOp::Modulo, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_bigint_multiply_large, bigint_multiply_large, 0xfeedc0d5, kLargeBigintLimbs);

BENCHMARK_CAPTURE(bench_negation<t81::core::limb>, limb_negate, negation_seed(false), 0);
BENCHMARK_CAPTURE(bench_negation<t81::core::bigint>, bigint_negate, negation_seed(true), kBigintLimbs);

BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_equal, CompareOp::Equal,
                  seed_offset(CompareOp::Equal, false), 0);
BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_not_equal, CompareOp::NotEqual,
                  seed_offset(CompareOp::NotEqual, false), 0);
BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_less, CompareOp::Less,
                  seed_offset(CompareOp::Less, false), 0);
BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_less_equal, CompareOp::LessEqual,
                  seed_offset(CompareOp::LessEqual, false), 0);
BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_greater, CompareOp::Greater,
                  seed_offset(CompareOp::Greater, false), 0);
BENCHMARK_CAPTURE(bench_compare<t81::core::limb>, limb_greater_equal, CompareOp::GreaterEqual,
                  seed_offset(CompareOp::GreaterEqual, false), 0);

BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_equal, CompareOp::Equal,
                  seed_offset(CompareOp::Equal, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_not_equal, CompareOp::NotEqual,
                  seed_offset(CompareOp::NotEqual, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_less, CompareOp::Less,
                  seed_offset(CompareOp::Less, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_less_equal, CompareOp::LessEqual,
                  seed_offset(CompareOp::LessEqual, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_greater, CompareOp::Greater,
                  seed_offset(CompareOp::Greater, true), kBigintLimbs);
BENCHMARK_CAPTURE(bench_compare<t81::core::bigint>, bigint_greater_equal, CompareOp::GreaterEqual,
                  seed_offset(CompareOp::GreaterEqual, true), kBigintLimbs);

const auto decimal_100_text = decimal_string(100);
const auto decimal_1000_text = decimal_string(1000);
const auto decimal_100_value =
    t81::io::from_string<t81::core::bigint>(decimal_100_text, 10);
const auto decimal_1000_value =
    t81::io::from_string<t81::core::bigint>(decimal_1000_text, 10);
const auto base81_100_text = base81_string(100);
const auto base81_256_text = base81_string(256);
const auto base81_100_value =
    t81::io::from_string<t81::core::bigint>(base81_100_text, 81);
const auto base81_256_value =
    t81::io::from_string<t81::core::bigint>(base81_256_text, 81);
const auto large_bigint_value = build_large_bigint(kLargeBigintLimbs);

BENCHMARK_CAPTURE(bench_bigint_shift<true>, shift_left_1, 0xabcde01, kBigintLimbs, 1);
BENCHMARK_CAPTURE(bench_bigint_shift<true>, shift_left_10, 0xabcde02, kBigintLimbs, 10);
BENCHMARK_CAPTURE(bench_bigint_shift<true>, shift_left_64, 0xabcde03, kBigintLimbs, 64);
BENCHMARK_CAPTURE(bench_bigint_shift<true>, shift_left_512, 0xabcde04, kBigintLimbs, 512);
BENCHMARK_CAPTURE(bench_bigint_shift<false>, shift_right_1, 0xabcde05, kBigintLimbs, 1);
BENCHMARK_CAPTURE(bench_bigint_shift<false>, shift_right_10, 0xabcde06, kBigintLimbs, 10);
BENCHMARK_CAPTURE(bench_bigint_shift<false>, shift_right_64, 0xabcde07, kBigintLimbs, 64);
BENCHMARK_CAPTURE(bench_bigint_shift<false>, shift_right_512, 0xabcde08, kBigintLimbs, 512);

BENCHMARK_CAPTURE(bench_bigint_decimal_to_string, decimal_to_string_100, decimal_100_value);
BENCHMARK_CAPTURE(bench_bigint_decimal_to_string, decimal_to_string_1000, decimal_1000_value);
BENCHMARK_CAPTURE(bench_bigint_decimal_from_string, decimal_from_string_100, decimal_100_text);
BENCHMARK_CAPTURE(bench_bigint_decimal_from_string, decimal_from_string_1000, decimal_1000_text);
BENCHMARK_CAPTURE(bench_bigint_base81_to_string, base81_to_string_100, base81_100_value);
BENCHMARK_CAPTURE(bench_bigint_base81_to_string, base81_to_string_256, base81_256_value);
BENCHMARK_CAPTURE(bench_bigint_base81_from_string, base81_from_string_100, base81_100_text);
BENCHMARK_CAPTURE(bench_bigint_base81_from_string, base81_from_string_256, base81_256_text);

BENCHMARK_CAPTURE(bench_bigint_pow, bigint_pow_128, 0xfeedc0de, kBigintLimbs, 128);
BENCHMARK_CAPTURE(bench_bigint_pow, bigint_pow_512, 0xfeedc0df, kBigintLimbs, 512);
BENCHMARK_CAPTURE(bench_bigint_pow_large, bigint_pow_large, 0xfeedc0d2,
                  kLargeBigintLimbs, kLargePowExponent);

BENCHMARK_CAPTURE(bench_bigint_gcd, bigint_gcd_small, 0xc0deba5e, kBigintLimbs);
BENCHMARK_CAPTURE(bench_bigint_gcd_large, bigint_gcd_large, 0xc0deba5f, kLargeBigintLimbs);

BENCHMARK_CAPTURE(bench_bigint_serialization, bigint_serialize_small, 0xbaadc0de, kBigintLimbs);

BENCHMARK_CAPTURE(bench_random_bigint, random_bigint_small, 0xfeedf00d, kBigintLimbs);

const t81::core::bigint mont_modulus = t81::core::bigint(197);
const t81::core::bigint mont_modulus_large = large_bigint_value;
BENCHMARK_CAPTURE(bench_montgomery_mul, montgomery_mul_small, 0xdeadbeef, mont_modulus);
BENCHMARK_CAPTURE(bench_montgomery_mul_large, montgomery_mul_large, 0xdeadface, mont_modulus_large);
BENCHMARK_MAIN();
