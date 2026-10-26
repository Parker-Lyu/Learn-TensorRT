#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lesson09 {

// Dense linear storage only. Empty shape denotes a scalar; zero/dynamic extents are unsupported.
inline std::size_t checked_tensor_bytes(const std::vector<int64_t>& shape, std::size_t bits) {
    if (bits != 4 && bits != 8 && bits != 16 && bits != 32 && bits != 64) {
        throw std::runtime_error("Unsupported tensor element bit width.");
    }
    std::size_t elements = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || static_cast<std::uint64_t>(dim) >
                            std::numeric_limits<std::size_t>::max() / elements) {
            throw std::runtime_error("Tensor has invalid extents or too many elements.");
        }
        elements *= static_cast<std::size_t>(dim);
    }
    if (bits == 4) {
        return elements / 2 + elements % 2;  // Round up without overflowing elements + 1.
    }
    const std::size_t bytes_per_element = bits / 8;
    if (elements > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
        throw std::runtime_error("Tensor byte count overflowed.");
    }
    return elements * bytes_per_element;
}

}  // namespace lesson09
