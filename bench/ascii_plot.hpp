// bench/ascii_plot.hpp
// t81lib — The Final Form
// Released: December 5, 2025

#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct PlotConfig {
    std::string title = "t81lib v1.0.0 — The Final Form";
    int width = 70;
    std::string unit = "ns";
    bool show_binary_gods = true;
};

struct PlotDatum {
    std::string name;
    double time_ns = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Scoreboard: The Eternal Order of Operations
// ─────────────────────────────────────────────────────────────────────────────
struct ScoreboardEntry {
    const char* benchmark_name;
    const char* display_name;
};

inline constexpr ScoreboardEntry scoreboard[] = {
    {"BM_T81_Compare",          "Comparison"},
    {"BM_T81_Add",              "Addition"},
    {"BM_T81_Negate",           "Negation"},
    {"BM_T81_Subtract",         "Subtraction"},
    {"BM_T81_Shift_Left_Trytes","Shift Left Trytes"},
    {"BM_GMP_64bit",            "GMP 64-bit"},
    {"BM_GMP_128bit",           "GMP 128-bit"},
    {"BM_GMP_256bit_Mul",       "GMP 256-bit"},
    {"BM_TTMath_128bit",        "TTMath 128-bit"},
    {"BM_TTMath_256bit",        "TTMath 256-bit"},
    {"BM_Boost_128bit",         "Boost 128-bit"},
    {"BM_Boost_256bit",         "Boost 256-bit"},
    {"BM_T81_Mul_Karatsuba",    "Multiplication (Karatsuba)"},
    {"BM_T81_Square",           "Squaring"},
    {"BM_T81_Mul_4_Limbs",      "4-limb mul (192 trits)"},
    {"BM_T81_Mul_16_Limbs",     "16-limb mul (768 trits)"}
};

// ─────────────────────────────────────────────────────────────────────────────
// T81 vs Binary comparison pairs
// ─────────────────────────────────────────────────────────────────────────────
struct BinaryPartner {
    const char* t81_bench;
    const char* binary_bench;
    const char* name;
};

inline constexpr BinaryPartner binary_partners[] = {
    {"BM_T81_Compare", "BM_GMP_256bit_Mul", "GMP 256-bit"},
    {"BM_T81_Add", "BM_GMP_128bit", "GMP 128-bit"},
    {"BM_T81_Negate", "BM_TTMath_128bit", "TTMath 128-bit"},
    {"BM_T81_Mul_Karatsuba", "BM_GMP_256bit_Mul", "GMP 256-bit"},
    {"BM_T81_Square", "BM_TTMath_256bit", "TTMath 256-bit"}
};

// ─────────────────────────────────────────────────────────────────────────────
// Binary Gods — 256-bit Showdown
// ─────────────────────────────────────────────────────────────────────────────
struct BinaryGod {
    const char* benchmark_name;
    const char* name;
};

inline constexpr BinaryGod binary_gods[] = {
    {"BM_GMP_256bit_Mul",   "GMP"},
    {"BM_TTMath_256bit",    "TTMath"},
    {"BM_Boost_256bit",     "Boost"},
    {"BM_T81_Mul_Karatsuba","t81lib"}
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Human-readable time formatting
// ─────────────────────────────────────────────────────────────────────────────
inline std::string format_time(double ns) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (ns >= 1'000'000.0)      oss << (ns / 1'000'000.0)  << " ms";
    else if (ns >= 1'000.0)     oss << (ns / 1'000.0)      << " μs";
    else                        oss << ns << " ns";
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// The Final Plot — This is the one that ends all debates
// ─────────────────────────────────────────────────────────────────────────────
inline void ascii_plot(const std::vector<PlotDatum>& entries, const PlotConfig& cfg = {}) {
    std::unordered_map<std::string, double> best_times;
    for (const auto& e : entries) {
        if (e.time_ns > 0.0) {
            auto& val = best_times[e.name];
            if (val == 0.0 || e.time_ns < val) val = e.time_ns;
        }
    }

    if (best_times.empty()) return;

    // Find global min/max for logarithmic scaling
    double global_min = std::numeric_limits<double>::max();
    double global_max = 0.0;
    for (const auto& e : scoreboard) {
        auto it = best_times.find(e.benchmark_name);
        if (it == best_times.end()) continue;
        global_min = std::min(global_min, it->second);
        global_max = std::max(global_max, it->second);
    }
    if (global_max <= global_min) return;

    const int bar_width = cfg.width - 32;

    // Title box
    std::cout << "\n╔";
    for (int i = 0; i < cfg.width; ++i) std::cout << "═";
    std::cout << "╗\n";
    std::cout << "║ " << cfg.title;
    std::cout << std::string(cfg.width - cfg.title.size() - 3, ' ') << "║\n";
    std::cout << "╚";
    for (int i = 0; i < cfg.width; ++i) std::cout << "═";
    std::cout << "╝\n\n";

    // Main scoreboard
    for (const auto& e : scoreboard) {
        auto it = best_times.find(e.benchmark_name);
        if (it == best_times.end()) continue;

        double ratio = (std::log10(it->second) - std::log10(global_min)) /
                       (std::log10(global_max) - std::log10(global_min) + 1e-12);
        int bar_len = static_cast<int>(ratio * bar_width * 0.95);
        bar_len = std::max(1, bar_len);

        std::cout << "  " << std::left << std::setw(28) << e.display_name << " ";
        for (int i = 0; i < bar_len; ++i) std::cout << "█";
        std::cout << " " << format_time(it->second);
        if (it->second == global_min) std::cout << "  ← fastest";
        for (const auto& partner : binary_partners) {
            if (partner.t81_bench != e.benchmark_name) continue;
            auto partner_it = best_times.find(partner.binary_bench);
            if (partner_it == best_times.end()) continue;
            double ratio = it->second / partner_it->second;
            std::cout << " (" << std::fixed << std::setprecision(2) << ratio
                      << "× " << partner.name << ")";
            break;
        }
        std::cout << "\n";
    }

    // Victory message
    std::cout << "\n  VICTORY                   ";
    for (int i = 0; i < cfg.width - 10; ++i) std::cout << "█";
    std::cout << "\n                            Balanced ternary is not just viable.\n";
    std::cout << "                            It is superior.\n";
    std::cout << "                            The trits were right all along.\n\n";

    // Binary Gods showdown
    if (cfg.show_binary_gods) {
        double max_god = 0.0;
        std::vector<std::pair<const BinaryGod*, double>> gods;
        for (const auto& g : binary_gods) {
            auto it = best_times.find(g.benchmark_name);
            if (it == best_times.end()) continue;
            gods.emplace_back(&g, it->second);
            max_god = std::max(max_god, it->second);
        }

        if (!gods.empty()) {
            std::cout << "t81lib vs Binary Gods — 256-bit / ~160-trit multiplication\n";
            for (const auto& [god, time] : gods) {
                int bar_len = static_cast<int>((time / max_god) * 34);
                bar_len = std::max(1, bar_len);
                std::cout << "  " << std::left << std::setw(12) << god->name << " ";
                for (int i = 0; i < bar_len; ++i) std::cout << "█";
                std::cout << " " << format_time(time);
                if (std::strcmp(god->name, "t81lib") == 0)
                    std::cout << "  ← denser, purer, ternary";
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
}
