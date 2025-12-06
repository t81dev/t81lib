#include <array>
#include <benchmark/benchmark.h>
#include <random>

#include "t81/core/T81Limb.hpp"

namespace {

std::mt19937 rng(42);
std::uniform_int_distribution<int> trit_dist(-13, 13);

void fill_random_t81(t81::core::T81Limb& value) {
    for (int i = 0; i < t81::core::T81Limb::TRYTES; ++i) {
        value.set_tryte(i, trit_dist(rng));
    }
}

} // namespace

namespace {

using t81::core::T81Limb;

[[nodiscard]] T81Limb table_add(const T81Limb& lhs, const T81Limb& rhs) {
    constexpr std::array<std::array<int8_t, 2>, 5> table = {{
        {{1, -1}},  // -2
        {{-1, 0}},  // -1
        {{0, 0}},   // 0
        {{1, 0}},   // +1
        {{-1, 1}}   // +2
    }};

    auto la = lhs.to_trits();
    auto lb = rhs.to_trits();

    std::array<int8_t, T81Limb::TRITS> temp{};
    std::array<int8_t, T81Limb::TRITS + 1> carry1{};
    std::array<int8_t, T81Limb::TRITS + 1> carry2{};
    for (int i = 0; i < T81Limb::TRITS; ++i) {
        int sum = la[i] + lb[i];
        int idx = sum + 2;
        temp[i] = table[idx][0];
        carry1[i + 1] = table[idx][1];
    }

    std::array<int8_t, T81Limb::TRITS> result_trits{};
    for (int i = 0; i < T81Limb::TRITS; ++i) {
        int sum = temp[i] + carry1[i];
        int idx = sum + 2;
        result_trits[i] = table[idx][0];
        carry2[i + 1] = table[idx][1];
    }

    return T81Limb::from_trits(result_trits);
}

} // namespace

static void BM_T81_Add(benchmark::State& state) {
    T81Limb a, b;
    fill_random_t81(a);
    fill_random_t81(b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(a + b);
    }
}

static void BM_T81_Add_Table(benchmark::State& state) {
    T81Limb a, b;
    fill_random_t81(a);
    fill_random_t81(b);
    for (auto _ : state) {
        benchmark::DoNotOptimize(table_add(a, b));
    }
}

BENCHMARK(BM_T81_Add)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_T81_Add_Table)->Unit(benchmark::kNanosecond);
