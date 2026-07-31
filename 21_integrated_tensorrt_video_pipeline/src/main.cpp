#include "pipeline_core.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string output = argc > 1 ? argv[1] : "21_integrated_tensorrt_video_pipeline/output/detections.jsonl";
    try {
        lesson21::SlotPool slots(2);
        lesson21::IdentityDispatcher dispatcher;
        const std::size_t first = slots.reserve();
        slots.mark_submitted(first, {0, {{0, 0, 0, lesson21::Clock::now(), {}}}});
        const std::size_t second = slots.reserve();
        slots.mark_submitted(second, {1, {{1, 0, 0, lesson21::Clock::now(), {}}}});
        // Reverse completion is intentional: it exercises the same identity boundary used by the
        // TensorRT collector without pretending deterministic CPU work is GPU evidence.
        dispatcher.dispatch(slots.begin_collection(second)); slots.release(second);
        dispatcher.dispatch(slots.begin_collection(first)); slots.release(first);
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("failed to open output: " + output);
        for (const auto& result : dispatcher.results())
            stream << "{\"stream_id\":" << result.stream_id << ",\"frame_id\":"
                   << result.frame_id << ",\"backend\":\"identity_test\"}\n";
        std::cout << "wrote " << dispatcher.results().size() << " identity records to " << output
                  << "\nThis CPU run is not TensorRT performance evidence.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
