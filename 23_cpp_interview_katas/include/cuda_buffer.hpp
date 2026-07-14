#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace lesson23 {
class CudaBuffer {
public:
    explicit CudaBuffer(std::size_t bytes) : bytes_(bytes) {
        if (!bytes) throw std::invalid_argument("CUDA buffer size must be positive");
        const auto status = cudaMalloc(&pointer_, bytes);
        if (status != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc: ") + cudaGetErrorString(status));
    }
    ~CudaBuffer() { if (pointer_) (void)cudaFree(pointer_); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept : pointer_(other.pointer_), bytes_(other.bytes_) {
        other.pointer_ = nullptr; other.bytes_ = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) { if (pointer_) (void)cudaFree(pointer_); pointer_ = other.pointer_;
            bytes_ = other.bytes_; other.pointer_ = nullptr; other.bytes_ = 0; }
        return *this;
    }
    void* get() const { return pointer_; }
    std::size_t size() const { return bytes_; }
private:
    void* pointer_{nullptr}; std::size_t bytes_{0};
};
}  // namespace lesson23
