#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lesson08 {

struct InputShape {
    std::string tensor_name;
    std::vector<int32_t> dimensions;
};

struct AppConfig {
    std::string onnx_path = "../05_torch_to_onnx/outputs/yolov8n.onnx";
    std::string engine_path = "outputs/yolov8n_cpp_basic.engine";
    std::string timing_cache_path;
    std::vector<InputShape> input_shapes;
    bool load_engine_only = false;
    bool allow_tf32 = false;
    std::size_t workspace_mib = 2048;
    int warmup_iterations = 1;
    int measured_iterations = 3;
};

struct TensorSummary {
    std::string name;
    std::string mode;
    std::string location;
    std::string data_type;
    std::vector<int64_t> dimensions;
    std::size_t byte_count = 0;
    std::uint64_t output_checksum = 0;
};

struct AppReport {
    std::string onnx_path;
    std::string engine_path;
    std::string timing_cache_path;
    bool engine_built = false;
    bool strongly_typed = true;
    bool tf32_allowed = false;
    bool timing_cache_loaded = false;
    bool timing_cache_written = false;
    std::vector<TensorSummary> tensors;
    std::size_t engine_bytes = 0;
    std::size_t timing_cache_bytes = 0;
    std::size_t total_device_bytes = 0;
    std::size_t total_host_bytes = 0;
    float average_enqueue_ms = 0.0F;
};

AppReport run_tensorrt_cpp_basic(const AppConfig& config);

}  // namespace lesson08
