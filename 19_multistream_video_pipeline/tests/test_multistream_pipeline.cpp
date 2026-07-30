#include "multistream_pipeline.hpp"

#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>

void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

int main() {
    try {
        lesson19::PipelineConfig config;
        config.queue_capacity = 3;
        config.max_batch_size = 4;
        const std::vector<lesson19::StreamConfig> streams = {
            {7, "", 60, std::chrono::milliseconds(0)},
            {42, "", 60, std::chrono::milliseconds(1)}};
        lesson19::MultiStreamPipeline pipeline(config, streams);
        const auto report = pipeline.run();
        std::set<std::pair<std::size_t, std::size_t>> identities;
        for (const auto& result : report.results) {
            require(result.stream_id == 7 || result.stream_id == 42, "result routed to unknown stream");
            require(identities.insert({result.stream_id, result.frame_id}).second,
                    "duplicate result identity");
        }
        require(report.results.size() == report.streams[0].processed + report.streams[1].processed,
                "dispatcher result count mismatch");
        for (const auto& stream : report.streams) {
            require(stream.queue_high_watermark <= config.queue_capacity, "queue exceeded capacity");
            require(stream.captured == stream.processed + stream.dropped, "stream accounting mismatch");
        }

        auto isolated_streams = streams;
        isolated_streams[0].fail_at_frame = 3;
        lesson19::MultiStreamPipeline isolated(config, isolated_streams);
        const auto isolated_report = isolated.run();
        require(isolated_report.streams[0].source_failures == 1, "isolated failure not counted");
        require(isolated_report.streams[1].processed > 0, "healthy stream did not continue");

        config.fail_inference_batch = 1;
        bool failed = false;
        try { lesson19::MultiStreamPipeline broken(config, streams); (void)broken.run(); }
        catch (const std::exception&) { failed = true; }
        require(failed, "inference failure was not propagated");
        std::cout << "All multistream pipeline tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
