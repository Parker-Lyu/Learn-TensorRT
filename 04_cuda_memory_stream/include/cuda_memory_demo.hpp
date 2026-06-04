#pragma once

#include <cstddef>

struct DemoConfig {
    // Number of float elements in the synthetic "input tensor" buffer.
    std::size_t element_count = 1U * 3U * 640U * 640U;

    // Repeats only the measured copy/compute path. Initialization is kept outside timing.
    int iterations = 20;
};

int run_cuda_memory_demo(const DemoConfig& config);
