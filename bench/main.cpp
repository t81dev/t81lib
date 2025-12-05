// bench/main.cpp
#include <benchmark/benchmark.h>
#include <unordered_map>
#include <utility>

#include "ascii_plot.hpp"

namespace {

class AsciiPlotReporter : public benchmark::BenchmarkReporter {
public:
    explicit AsciiPlotReporter(PlotConfig cfg)
        : cfg_(std::move(cfg))
    {}

    bool ReportContext(const Context& context) override {
        console_.ReportContext(context);
        return true;
    }

    void ReportRuns(const std::vector<Run>& reports) override {
        console_.ReportRuns(reports);
        for (const auto& report : reports) {
            if (report.error_occurred) continue;
            double multiplier = benchmark::GetTimeUnitMultiplier(report.time_unit);
            if (report.iterations == 0) continue;
            double cpu_seconds = report.cpu_accumulated_time / static_cast<double>(report.iterations);
            double cpu_ns = cpu_seconds * 1e9;
            const std::string name = report.benchmark_name();
            auto it = data_.find(name);
            if (it == data_.end() || cpu_ns < it->second) {
                data_[name] = cpu_ns;
            }
        }
    }

    void Finalize() override {
        console_.Finalize();
        std::vector<PlotDatum> vec;
        vec.reserve(data_.size());
        for (const auto& entry : data_) {
            vec.push_back({entry.first, entry.second});
        }
        ascii_plot(vec, cfg_);
    }

private:
    PlotConfig cfg_;
    benchmark::ConsoleReporter console_;
    std::unordered_map<std::string, double> data_;
};

} // namespace

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

    AsciiPlotReporter reporter({
        .title = "t81lib v1.0.0 — The Final Form",
        .width = 64,
        .unit = "ns",
        .show_predicted = true
    });
    benchmark::RunSpecifiedBenchmarks(&reporter);
    benchmark::Shutdown();
    return 0;
}
