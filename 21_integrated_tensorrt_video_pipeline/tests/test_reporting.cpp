#include "metrics.hpp"
#include "result_writer.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)

namespace {
void submission_is_observable_before_completion(const std::filesystem::path& output) {
    lesson21::PipelineMetrics metrics(output);
    metrics.record_submission(2);
    CHECK(metrics.submitted() == 2 && metrics.completed() == 0);
    lesson21::GpuBatchResult result;
    result.metadata.frames.resize(1);
    metrics.record_batch(result, 1.0, 0.5, 2.0, 3.0, {4.0});
    CHECK(metrics.submitted() == 2 && metrics.completed() == 1);
    // One completed frame and one submitted frame that failed before result dispatch.
    metrics.write(output / "metrics.json", {}, {}, {2, 2, 0, 0, 2, 1, 1, 0}, 1, 1.0, false);
    std::ifstream stream(output / "metrics.json");
    const std::string json((std::istreambuf_iterator<char>(stream)), {});
    CHECK(json.find("\"failed_in_flight\":1") != std::string::npos);
    CHECK(json.find("\"result_output_ms\":3.000000") != std::string::npos);
    metrics.record_batch(result, 0, 0, 0, 0, {0});
    bool rejected = false;
    try { metrics.record_batch(result, 0, 0, 0, 0, {0}); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
}

void writer_checks_shape_and_separates_stage_timings(const std::filesystem::path& output) {
    lesson21::ResultWriter writer(output, 1);
    lesson21::GpuBatchResult result;
    result.metadata.frames.push_back({0, 0, 0, lesson21::Clock::now(), {1, 0, 0, 640, 640}});
    result.output_shape = {1, 84, 1};
    result.output.resize(84, 0.0F);  // Controlled CPU fixture: no confident detections.
    std::vector<cv::Mat> images{cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0))};
    std::vector<double> latencies;
    const auto timing = writer.write(result, images, latencies);
    CHECK(timing.cpu_postprocess_ms >= 0 && timing.result_output_ms > 0);
    CHECK(latencies.size() == 1);
    CHECK(std::filesystem::exists(output / "annotated_0.jpg"));
    // Reaching the record limit still processes a frame and records its dispatch latency.
    writer.write(result, images, latencies);
    CHECK(latencies.size() == 2);
    result.output_shape = {1, 42, 1};
    bool rejected = false;
    try { writer.write(result, images, latencies); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected test output directory");
        const std::filesystem::path output(argv[1]);
        submission_is_observable_before_completion(output / "metrics");
        writer_checks_shape_and_separates_stage_timings(output / "writer");
        std::cout << "reporting tests passed (submission, failure accounting, output layout, timing)\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
