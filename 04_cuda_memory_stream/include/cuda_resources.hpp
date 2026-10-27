#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace lesson04 {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

// Destructors preserve active exceptions; CUDA cleanup failures remain observable.
inline void report_cuda_cleanup(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed during cleanup: %s\n", operation,
                     cudaGetErrorString(status));
    }
}

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() noexcept {
        if (stream_ != nullptr) {
            // Destroy alone does not wait: finish work before dependent resources unwind.
            report_cuda_cleanup(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
            report_cuda_cleanup(cudaStreamDestroy(stream_), "cudaStreamDestroy");
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept {
        return stream_;
    }

    void synchronize() const {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        check_cuda(cudaEventCreateWithFlags(&event_, flags), "cudaEventCreateWithFlags");
    }

    ~CudaEvent() noexcept {
        if (event_ != nullptr) {
            report_cuda_cleanup(cudaEventDestroy(event_), "cudaEventDestroy");
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const noexcept {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename FreeFn>
class UniqueCudaMemory {
    static_assert(std::is_nothrow_move_constructible_v<FreeFn>);
    static_assert(std::is_nothrow_move_assignable_v<FreeFn>);
    static_assert(std::is_nothrow_invocable_r_v<cudaError_t, FreeFn&, float*>);

public:
    UniqueCudaMemory() = default;

    UniqueCudaMemory(float* ptr, std::size_t element_count, FreeFn free_fn)
        : ptr_(ptr), element_count_(element_count), free_fn_(std::move(free_fn)) {}

    ~UniqueCudaMemory() noexcept {
        reset();
    }

    UniqueCudaMemory(const UniqueCudaMemory&) = delete;
    UniqueCudaMemory& operator=(const UniqueCudaMemory&) = delete;

    UniqueCudaMemory(UniqueCudaMemory&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          element_count_(std::exchange(other.element_count_, 0)),
          free_fn_(std::move(other.free_fn_)) {}

    UniqueCudaMemory& operator=(UniqueCudaMemory&& other) noexcept {
        if (this != &other) {
            reset();
            free_fn_ = std::move(other.free_fn_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            element_count_ = std::exchange(other.element_count_, 0);
        }
        return *this;
    }

    float* get() const noexcept {
        return ptr_;
    }

    std::size_t element_count() const noexcept {
        return element_count_;
    }

    void reset() noexcept {
        if (ptr_ != nullptr) {
            report_cuda_cleanup(free_fn_(ptr_), "CUDA memory release");
            ptr_ = nullptr;
        }
        element_count_ = 0;
    }

private:
    float* ptr_ = nullptr;
    std::size_t element_count_ = 0;
    FreeFn free_fn_{};
};

struct CudaFreeDevice {
    cudaError_t operator()(float* ptr) const noexcept {
        return cudaFree(ptr);
    }
};

struct CudaFreeHost {
    cudaError_t operator()(float* ptr) const noexcept {
        return cudaFreeHost(ptr);
    }
};

using DeviceBuffer = UniqueCudaMemory<CudaFreeDevice>;
using PinnedHostBuffer = UniqueCudaMemory<CudaFreeHost>;
using ManagedBuffer = UniqueCudaMemory<CudaFreeDevice>;

}  // namespace lesson04
