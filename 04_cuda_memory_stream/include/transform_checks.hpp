#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace lesson04 {

constexpr float kScale = 2.0F;
constexpr float kBias = 1.0F;
constexpr float kTolerance = 1.0e-5F;
constexpr int kThreadsPerBlock = 256;

inline std::size_t float_buffer_bytes(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("Requested buffer is too large.");
    }
    return count * sizeof(float);
}

inline int transform_block_count(std::size_t count) {
    if (count == 0) {
        return 0;
    }
    // Avoid count + block_size - 1, which can wrap at the size_t limit.
    const std::size_t blocks = 1 + (count - 1) / kThreadsPerBlock;
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Requested transform needs too many CUDA blocks.");
    }
    return static_cast<int>(blocks);
}

inline bool validate_transform(const float* input, const float* output, std::size_t count,
                               std::size_t* bad_index) {
    if (count != 0 && (input == nullptr || output == nullptr)) {
        throw std::runtime_error("Nonempty transform requires input and output buffers.");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const float expected = input[i] * kScale + kBias;
        if (!std::isfinite(expected) || !std::isfinite(output[i]) ||
            std::abs(output[i] - expected) > kTolerance) {
            if (bad_index != nullptr) {
                *bad_index = i;
            }
            return false;
        }
    }
    return true;
}

}  // namespace lesson04
