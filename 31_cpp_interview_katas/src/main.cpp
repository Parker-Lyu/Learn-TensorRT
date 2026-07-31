#include "katas.hpp"

#include <iostream>

int main() {
    const lesson31::Box first{0.0F, 0.0F, 10.0F, 10.0F, 0.9F, 0};
    const lesson31::Box second{5.0F, 5.0F, 15.0F, 15.0F, 0.8F, 0};
    std::cout << "IoU=" << lesson31::iou(first, second) << '\n';
}
