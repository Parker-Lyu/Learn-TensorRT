#include "tensorrt_raii.hpp"

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
    try {
        (void)lesson07::parse_failure_stage("not-a-stage");
    } catch (const std::runtime_error&) {
        return 0;
    }
    std::cerr << "invalid failure stage was accepted\n";
    return 1;
}
