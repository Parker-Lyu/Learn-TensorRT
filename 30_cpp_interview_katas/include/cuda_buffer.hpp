#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace lesson30 {
class CudaBuffer {
public:
    explicit CudaBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes == 0) {
            throw std::invalid_argument("CUDA buffer size must be positive");
        }
        const auto status = cudaMalloc(&pointer_, bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(status));
        }
    }

    ~CudaBuffer() {
        if (pointer_ != nullptr) {
            // Destructors cannot report a cudaFree failure without risking termination.
            (void)cudaFree(pointer_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : pointer_(other.pointer_), bytes_(other.bytes_) {
        other.pointer_ = nullptr;
        other.bytes_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (pointer_ != nullptr) {
                (void)cudaFree(pointer_);
            }
            pointer_ = other.pointer_;
            bytes_ = other.bytes_;
            other.pointer_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    [[nodiscard]] void* get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

private:
    void* pointer_{nullptr};
    std::size_t bytes_{0};
};

}  // namespace lesson30
