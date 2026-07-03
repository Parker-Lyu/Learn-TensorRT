#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lesson07 {

struct InputShape {
    std::string tensor_name;
    std::vector<int32_t> dimensions;
};

struct RunConfig {
    std::string engine_path = "../06_trtexec_engine/outputs/yolov8n_static_fp32.engine";
    std::vector<InputShape> input_shapes;
    int warmup_iterations = 1;
    int measured_iterations = 3;
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

}  // namespace lesson07
