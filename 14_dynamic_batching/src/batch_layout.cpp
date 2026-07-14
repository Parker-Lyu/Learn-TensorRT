#include "batch_layout.hpp"

#include <limits>
#include <stdexcept>

namespace lesson14 {

std::size_t checked_volume(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("tensor shape must not be empty");
    }
    std::size_t volume = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 || volume > std::numeric_limits<std::size_t>::max() /
                                         static_cast<std::size_t>(dimension)) {
            throw std::invalid_argument("tensor shape contains an invalid or overflowing dimension");
        }
        volume *= static_cast<std::size_t>(dimension);
    }
    return volume;
}

std::size_t input_batch_offset(std::size_t batch_index, int channels, int height, int width) {
    if (channels <= 0 || height <= 0 || width <= 0) {
        throw std::invalid_argument("input dimensions must be positive");
    }
    return batch_index * checked_volume({channels, height, width});
}

std::size_t output_batch_offset(std::size_t batch_index,
                                const std::vector<int64_t>& output_shape) {
    if (output_shape.size() < 2 || output_shape.front() <= 0) {
        throw std::invalid_argument("output shape must contain a positive batch dimension");
    }
    std::vector<int64_t> per_sample(output_shape.begin() + 1, output_shape.end());
    return batch_index * checked_volume(per_sample);
}

std::vector<float> make_batched_input(std::size_t batch_size, int channels, int height, int width) {
    if (batch_size == 0) {
        throw std::invalid_argument("batch size must be positive");
    }
    const std::size_t sample_elements = input_batch_offset(1, channels, height, width);
    std::vector<float> tensor(batch_size * sample_elements);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        const std::size_t offset = input_batch_offset(batch, channels, height, width);
        const float marker = static_cast<float>(batch + 1) / static_cast<float>(batch_size + 1);
        for (std::size_t element = 0; element < sample_elements; ++element) {
            tensor[offset + element] = marker;
        }
    }
    return tensor;
}

}  // namespace lesson14
