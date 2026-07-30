#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lesson11 {

struct LetterboxInfo {
    int original_width = 0;
    int original_height = 0;
    int input_width = 0;
    int input_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;
    float scale = 1.0F;
};

struct Box {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
};

struct Detection {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0F;
    Box box;
};

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::size_t byte_count = 0;
};

struct InferenceOutput {
    std::string output_name;
    std::vector<int64_t> output_shape;
    std::vector<float> values;
    float h2d_ms = 0.0F;
    float enqueue_host_ms = 0.0F;
    float gpu_compute_ms = 0.0F;
    float d2h_ms = 0.0F;
};

}  // namespace lesson11
