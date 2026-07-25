#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lesson14 {

struct RuntimeIdentity {
    std::string gpu_name;
    int compute_capability_major{0};
    int compute_capability_minor{0};
    int tensorrt_major{0};
    int tensorrt_minor{0};
    int tensorrt_patch{0};
    int cuda_runtime_version{0};
    int cuda_driver_version{0};
};

struct InferenceTiming {
    std::size_t batch_size{0};
    float h2d_ms{0.0F};
    float compute_ms{0.0F};
    float d2h_ms{0.0F};
    std::vector<int64_t> output_shape;
    double output_checksum{0.0};
};

class DynamicBatchRunner {
public:
    explicit DynamicBatchRunner(const std::string& engine_path);
    ~DynamicBatchRunner();
    DynamicBatchRunner(const DynamicBatchRunner&) = delete;
    DynamicBatchRunner& operator=(const DynamicBatchRunner&) = delete;

    InferenceTiming infer(const std::vector<float>& input, std::size_t batch_size);
    std::string input_name() const;
    std::string output_name() const;
    RuntimeIdentity runtime_identity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson14
