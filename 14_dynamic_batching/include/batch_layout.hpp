#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lesson14 {

std::size_t checked_volume(const std::vector<int64_t>& shape);
std::size_t input_batch_offset(std::size_t batch_index, int channels, int height, int width);
std::size_t output_batch_offset(std::size_t batch_index,
                                const std::vector<int64_t>& output_shape);
std::vector<float> make_batched_input(std::size_t batch_size, int channels, int height, int width);

}  // namespace lesson14
