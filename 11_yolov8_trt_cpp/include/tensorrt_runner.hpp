#pragma once

#include "yolo_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace lesson11 {

class TensorRtRunner {
public:
    explicit TensorRtRunner(const std::string& engine_path);
    ~TensorRtRunner();

    TensorRtRunner(const TensorRtRunner&) = delete;
    TensorRtRunner& operator=(const TensorRtRunner&) = delete;

    std::vector<int64_t> input_shape() const;
    std::string input_name() const;
    std::string output_name() const;

    InferenceOutput infer(const std::vector<float>& input_tensor);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson11
