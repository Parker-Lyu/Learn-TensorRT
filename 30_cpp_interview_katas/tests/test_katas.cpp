#include "katas.hpp"

#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        using namespace lesson30;

        require(iou({0, 0, 10, 10, 0, 0}, {0, 0, 10, 10, 0, 0}) == 1.0F,
                "identical IoU");
        require(iou({0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0}) == 0.0F,
                "degenerate IoU");
        const auto kept = nms(
            {{0, 0, 10, 10, 0.9F, 0}, {1, 1, 9, 9, 0.8F, 0}, {1, 1, 9, 9, 0.7F, 1}},
            0.5F);
        require(kept.size() == 2, "class-aware NMS");
        require(nms({}, 0.5F).empty(), "empty NMS");
        require_invalid_argument([] { (void)nms({}, 1.1F); }, "invalid NMS threshold");

        const Image image{2, 2, 1, {0, 10, 20, 30}};
        require(std::abs(bilinear_sample(image, 0.5F, 0.5F, 0) - 15.0F) < 1e-6F,
                "bilinear interpolation");
        require(bilinear_sample(image, -2.0F, 4.0F, 0) == 20.0F,
                "bilinear boundary clamp");
        require_invalid_argument(
            [&] { (void)bilinear_sample(image, 0.0F, 0.0F, 1); }, "invalid image channel");
        require_invalid_argument(
            [] { (void)hwc_to_chw({2, 2, 1, {1.0F}}); }, "invalid HWC data size");

        const Image rgb{1, 2, 3, {1, 2, 3, 4, 5, 6}};
        require(hwc_to_chw(rgb) == std::vector<float>({1, 4, 2, 5, 3, 6}), "HWC to CHW");
        const auto mapped = map_from_letterbox(
            {10, 20, 110, 220, 0, 0}, {2, 10, 20, 50, 100});
        require(mapped.x1 == 0 && mapped.y1 == 0 && mapped.x2 == 50 && mapped.y2 == 100,
                "letterbox clamp");
        require(top_k_indices({1, 3, 3, 2}, 3) == std::vector<std::size_t>({1, 2, 3}),
                "stable Top-K");

        RingBuffer<int> ring(2);
        require(ring.push(1) && ring.push(2) && !ring.push(3), "ring full");
        require(ring.pop() == 1 && ring.push(3) && ring.pop() == 2 && ring.pop() == 3 &&
                    !ring.pop(),
                "ring wrap");

        BoundedQueue<int> queue(1);
        require(queue.push(7), "initial queue push");
        auto blocked = std::async(std::launch::async, [&] { return queue.push(8); });
        queue.close();
        require(!blocked.get(), "close wakes blocked producer");
        require(queue.pop() == 7 && !queue.pop(), "queue drain after close");

        std::cout << "All CPU C++ katas passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
