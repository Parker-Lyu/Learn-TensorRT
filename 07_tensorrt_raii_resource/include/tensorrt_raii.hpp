#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lesson07 {

enum class FailureStage {
    kNone,
    kAfterEngineRead,
    kAfterRuntimeCreation,
    kAfterEngineDeserialization,
    kAfterContextCreation,
    kAfterFirstBufferAllocation,
    kAfterStreamCreation,
    kBeforeEnqueue,
};

FailureStage parse_failure_stage(const std::string& name);
const char* failure_stage_name(FailureStage stage) noexcept;

struct InputShape {
    std::string tensor_name;
    std::vector<int32_t> dimensions;
};

struct RunConfig {
    std::string engine_path = "../06_trtexec_engine/outputs/yolov8n_static_fp32.engine";
    std::vector<InputShape> input_shapes;
    int warmup_iterations = 1;
    int measured_iterations = 3;
    FailureStage injected_failure = FailureStage::kNone;
};

struct TensorReport {
    std::string name;
    std::string mode;
    std::string location;
    std::string data_type;
    std::vector<int64_t> dimensions;
    std::size_t byte_count = 0;
};

struct InferenceReport {
    std::string engine_path;
    std::vector<TensorReport> tensors;
    std::size_t total_device_bytes = 0;
    float average_enqueue_ms = 0.0F;
};

InferenceReport run_smoke_inference(const RunConfig& config);

struct LifecycleConfig {
    RunConfig run;
    int repetitions = 10;
    std::size_t memory_tolerance_bytes = 16U * 1024U * 1024U;
};

struct LifecycleReport {
    int repetitions = 0;
    int completed_runs = 0;
    int expected_failures = 0;
    std::size_t device_bytes_before = 0;
    std::size_t device_bytes_after = 0;
    std::size_t host_rss_bytes_before = 0;
    std::size_t host_rss_bytes_after = 0;
    bool memory_stable = false;
};

LifecycleReport run_repeated_lifecycle_test(const LifecycleConfig& config);

}  // namespace lesson07
