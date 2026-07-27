#include "katas.hpp"

#include <iostream>

int main() {
    const lesson23::Box first{0.0F, 0.0F, 10.0F, 10.0F, 0.9F, 0};
    const lesson23::Box second{5.0F, 5.0F, 15.0F, 15.0F, 0.8F, 0};
    std::cout << "IoU=" << lesson23::iou(first, second) << '\n';
}
