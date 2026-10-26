#include "transform_checks.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("transform contract check failed");
    }
}
template <typename Function>
void require_throws(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("expected invalid transform argument to throw");
}
}  // namespace

int main() {
    using namespace lesson04;
    require(transform_block_count(0) == 0);
    require(transform_block_count(1) == 1);
    require(transform_block_count(255) == 1);
    require(transform_block_count(256) == 1);
    require(transform_block_count(257) == 2);
    require_throws([] { transform_block_count(std::numeric_limits<std::size_t>::max()); });
    require(float_buffer_bytes(257) == 1028);
    require_throws([] { float_buffer_bytes(std::numeric_limits<std::size_t>::max()); });
    require(validate_transform(nullptr, nullptr, 0, nullptr));
    require_throws([] { validate_transform(nullptr, nullptr, 1, nullptr); });
    for (std::size_t count : {1, 255, 256, 257}) {
        std::vector<float> input(count), output(count);
        for (std::size_t i = 0; i < count; ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 3) - 1);
            output[i] = 2.0F * input[i] + 1.0F;
        }
        require(validate_transform(input.data(), output.data(), count, nullptr));
        output.back() += 1.0F;
        std::size_t bad = 0;
        require(!validate_transform(input.data(), output.data(), count, &bad) && bad == count - 1);
    }
    const float finite = 1.0F;
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::max()}) {
        require(!validate_transform(&invalid, &finite, 1, nullptr));
        require(!validate_transform(&finite, &invalid, 1, nullptr));
    }
    std::cout << "transform host contract tests passed\n";
}
