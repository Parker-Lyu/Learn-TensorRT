#include "tensor_size.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("tensor storage check failed");
    }
}
template <typename Function>
void require_throws(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("expected invalid tensor storage to throw");
}
}  // namespace

int main() {
    using lesson09::checked_tensor_bytes;
    require(checked_tensor_bytes({}, 32) == 4);
    require(checked_tensor_bytes({1, 3, 640, 640}, 32) == 4915200);
    require(checked_tensor_bytes({3}, 4) == 2);
    require(checked_tensor_bytes({4}, 4) == 2);
    require_throws([] { checked_tensor_bytes({1, -1}, 32); });
    require_throws([] { checked_tensor_bytes({0}, 32); });
    require_throws([] { checked_tensor_bytes({1}, 0); });
    require_throws([] { checked_tensor_bytes({std::numeric_limits<int64_t>::max(), 3}, 8); });
    require_throws([] { checked_tensor_bytes({std::numeric_limits<int64_t>::max()}, 32); });
    require(checked_tensor_bytes({std::numeric_limits<int64_t>::max()}, 4) ==
            static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) / 2 + 1);
    std::cout << "tensor storage tests passed\n";
}
