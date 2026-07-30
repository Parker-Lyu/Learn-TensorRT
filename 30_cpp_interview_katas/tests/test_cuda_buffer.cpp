#include "cuda_buffer.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA buffer test skipped: " << cudaGetErrorString(device_status) << '\n';
        return 77;
    }

    try {
        lesson30::CudaBuffer first(64);
        lesson30::CudaBuffer second(std::move(first));
        if (first.get() != nullptr || first.size() != 0 || second.get() == nullptr ||
            second.size() != 64) {
            throw std::runtime_error("CUDA ownership transfer failed");
        }
        std::cout << "CUDA buffer ownership test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
