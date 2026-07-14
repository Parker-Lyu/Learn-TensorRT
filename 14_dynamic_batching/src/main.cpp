#include "batch_layout.hpp"
#include "dynamic_batch_runner.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path engine;
    std::filesystem::path output;
    int warmup{3};
    int iterations{10};
};

Options parse_args(int argc, char** argv) {
    const auto executable = std::filesystem::absolute(argv[0]).parent_path();
    const auto root = (executable / ".." / "..").lexically_normal();
    Options options{root / "06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine",
                    executable / ".." / "outputs/batch_benchmark.csv"};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() {
            if (++index >= argc) throw std::invalid_argument(argument + " requires a value");
            return std::string(argv[index]);
        };
        if (argument == "--engine") options.engine = value();
        else if (argument == "--output") options.output = value();
        else if (argument == "--warmup") options.warmup = std::stoi(value());
        else if (argument == "--iterations") options.iterations = std::stoi(value());
        else if (argument == "--help") {
            std::cout << "Usage: dynamic_batching [--engine PATH] [--output CSV] "
                         "[--warmup N] [--iterations N]\n";
            std::exit(EXIT_SUCCESS);
        } else throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument("warmup must be non-negative and iterations must be positive");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        lesson14::DynamicBatchRunner runner(options.engine.string());
        std::filesystem::create_directories(options.output.parent_path());
        std::ofstream csv(options.output);
        if (!csv) throw std::runtime_error("failed to create benchmark CSV");
        csv << "batch,mean_compute_ms,images_per_second,mean_h2d_ms,mean_d2h_ms\n";

        std::cout << "input=" << runner.input_name() << " output=" << runner.output_name() << '\n';
        for (const std::size_t batch : {1U, 2U, 4U}) {
            const auto input = lesson14::make_batched_input(batch, 3, 640, 640);
            for (int iteration = 0; iteration < options.warmup; ++iteration) runner.infer(input, batch);
            std::vector<lesson14::InferenceTiming> samples;
            for (int iteration = 0; iteration < options.iterations; ++iteration) {
                samples.push_back(runner.infer(input, batch));
            }
            auto mean = [&](auto member) {
                double sum = 0.0;
                for (const auto& sample : samples) sum += sample.*member;
                return sum / static_cast<double>(samples.size());
            };
            const double compute = mean(&lesson14::InferenceTiming::compute_ms);
            const double throughput = static_cast<double>(batch) * 1000.0 / compute;
            csv << batch << ',' << compute << ',' << throughput << ','
                << mean(&lesson14::InferenceTiming::h2d_ms) << ','
                << mean(&lesson14::InferenceTiming::d2h_ms) << '\n';
            std::cout << std::fixed << std::setprecision(3) << "batch=" << batch
                      << " compute=" << compute << " ms throughput=" << throughput
                      << " images/s output_offset[1]="
                      << (batch > 1 ? lesson14::output_batch_offset(1, samples.back().output_shape) : 0)
                      << '\n';
        }
        std::cout << "saved " << options.output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
