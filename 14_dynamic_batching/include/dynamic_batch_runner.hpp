#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lesson14 {

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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson14
