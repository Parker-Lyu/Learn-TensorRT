#include "tensorrt_raii.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using lesson07::FailureStage;
    if (lesson07::parse_failure_stage("first-buffer") !=
        FailureStage::kAfterFirstBufferAllocation) {
        std::cerr << "failure stage parsing returned the wrong enum\n";
        return 1;
    }
    if (std::string(lesson07::failure_stage_name(FailureStage::kBeforeEnqueue)) != "enqueue") {
        std::cerr << "failure stage formatting returned the wrong name\n";
        return 1;
    }

    const std::filesystem::path temp_engine =
        std::filesystem::temp_directory_path() / "lesson07_invalid_engine.bin";
    {
        std::ofstream output(temp_engine, std::ios::binary);
        output << "not-a-real-engine";
    }

    try {
        (void)lesson07::run_smoke_inference({temp_engine.string()});
    } catch (const std::runtime_error&) {
        std::filesystem::remove(temp_engine);
        return 0;
    }

    std::filesystem::remove(temp_engine);
    std::cerr << "invalid engine input did not fail as expected\n";
    return 1;
}
