#include "cuda_memory_demo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kPathColumnWidth = 50;
constexpr float kScale = 2.0F;
constexpr float kBias = 1.0F;
constexpr float kTolerance = 1.0e-5F;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

cudaError_t prefetch_to_device_async(const void* pointer,
                                     std::size_t byte_count,
                                     int device_id,
                                     cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    // CUDA 13 replaced the integer destination with an explicit location and flags.
    const cudaMemLocation location{cudaMemLocationTypeDevice, device_id};
    return cudaMemPrefetchAsync(pointer, byte_count, location, 0U, stream);
#else
    return cudaMemPrefetchAsync(pointer, byte_count, device_id, stream);
#endif
}

cudaError_t prefetch_to_host_async(const void* pointer,
                                   std::size_t byte_count,
                                   cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    const cudaMemLocation location{cudaMemLocationTypeHost, 0};
    return cudaMemPrefetchAsync(pointer, byte_count, location, 0U, stream);
#else
    return cudaMemPrefetchAsync(pointer, byte_count, cudaCpuDeviceId, stream);
#endif
}

std::size_t byte_count_for_float_count(std::size_t element_count) {
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::runtime_error("Requested buffer is too large.");
    }
    return element_count * sizeof(float);
}

struct DeviceInfo {
    int device_id = 0;
    std::string name;
    bool can_map_host_memory = false;
    bool integrated_gpu = false;
};

DeviceInfo get_device_info() {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device is visible.");
    }

    int device_id = 0;
    check_cuda(cudaGetDevice(&device_id), "cudaGetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");

    DeviceInfo info;
    info.device_id = device_id;
    info.name = prop.name;
    info.can_map_host_memory = prop.canMapHostMemory != 0;
    info.integrated_gpu = prop.integrated != 0;
    return info;
}

class CudaStream {
public:
    CudaStream() {
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const {
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

    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const {
        return event_;
    }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename FreeFn>
class UniqueCudaMemory {
public:
    UniqueCudaMemory() = default;

    UniqueCudaMemory(float* ptr, std::size_t element_count, FreeFn free_fn)
        : ptr_(ptr), element_count_(element_count), free_fn_(std::move(free_fn)) {}

    ~UniqueCudaMemory() {
        reset();
    }

    UniqueCudaMemory(const UniqueCudaMemory&) = delete;
    UniqueCudaMemory& operator=(const UniqueCudaMemory&) = delete;

    UniqueCudaMemory(UniqueCudaMemory&& other) noexcept {
        *this = std::move(other);
    }

    UniqueCudaMemory& operator=(UniqueCudaMemory&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            element_count_ = other.element_count_;
            free_fn_ = std::move(other.free_fn_);
            other.ptr_ = nullptr;
            other.element_count_ = 0;
        }
        return *this;
    }

    float* get() const {
        return ptr_;
    }

    std::size_t element_count() const {
        return element_count_;
    }

    void reset() {
        if (ptr_ != nullptr) {
            (void)free_fn_(ptr_);
            ptr_ = nullptr;
            element_count_ = 0;
        }
    }

private:
    float* ptr_ = nullptr;
    std::size_t element_count_ = 0;
    FreeFn free_fn_{};
};

struct CudaFreeDevice {
    cudaError_t operator()(float* ptr) const {
        return cudaFree(ptr);
    }
};

struct CudaFreeHost {
    cudaError_t operator()(float* ptr) const {
        return cudaFreeHost(ptr);
    }
};

using DeviceBuffer = UniqueCudaMemory<CudaFreeDevice>;
using PinnedHostBuffer = UniqueCudaMemory<CudaFreeHost>;
using ManagedBuffer = UniqueCudaMemory<CudaFreeDevice>;

DeviceBuffer make_device_buffer(std::size_t element_count) {
    float* ptr = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), byte_count_for_float_count(element_count)),
               "cudaMalloc");
    return DeviceBuffer(ptr, element_count, CudaFreeDevice{});
}

PinnedHostBuffer make_pinned_host_buffer(std::size_t element_count) {
    float* ptr = nullptr;
    check_cuda(cudaMallocHost(reinterpret_cast<void**>(&ptr),
                              byte_count_for_float_count(element_count)),
               "cudaMallocHost");
    return PinnedHostBuffer(ptr, element_count, CudaFreeHost{});
}

PinnedHostBuffer make_mapped_host_buffer(std::size_t element_count) {
    float* ptr = nullptr;
    check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                             byte_count_for_float_count(element_count),
                             cudaHostAllocMapped),
               "cudaHostAllocMapped");
    return PinnedHostBuffer(ptr, element_count, CudaFreeHost{});
}

ManagedBuffer make_managed_buffer(std::size_t element_count) {
    float* ptr = nullptr;
    check_cuda(cudaMallocManaged(reinterpret_cast<void**>(&ptr),
                                 byte_count_for_float_count(element_count)),
               "cudaMallocManaged");
    return ManagedBuffer(ptr, element_count, CudaFreeDevice{});
}

__global__ void transform_kernel(const float* input, float* output, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] * kScale + kBias;
    }
}

void launch_transform(const float* input,
                      float* output,
                      std::size_t element_count,
                      cudaStream_t stream) {
    const std::size_t max_elements =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) * kThreadsPerBlock;
    if (element_count > max_elements) {
        throw std::runtime_error("Requested element_count needs more CUDA blocks than this "
                                 "lesson target supports.");
    }

    const int block_count =
        static_cast<int>((element_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    transform_kernel<<<block_count, kThreadsPerBlock, 0, stream>>>(input, output, element_count);
    check_cuda(cudaGetLastError(), "transform_kernel launch");
}

void fill_input(float* data, std::size_t element_count) {
    for (std::size_t i = 0; i < element_count; ++i) {
        // Deterministic but not constant, so indexing mistakes show up during validation.
        data[i] = static_cast<float>((i % 1024U) - 512.0F) / 128.0F;
    }
}

bool validate_output(const float* input,
                     const float* output,
                     std::size_t element_count,
                     std::size_t* bad_index) {
    for (std::size_t i = 0; i < element_count; ++i) {
        const float expected = input[i] * kScale + kBias;
        if (std::abs(output[i] - expected) > kTolerance) {
            if (bad_index != nullptr) {
                *bad_index = i;
            }
            return false;
        }
    }
    return true;
}

struct Measurement {
    std::string name;
    float average_ms = 0.0F;
    float bandwidth_gib_per_s = 0.0F;
    bool passed = false;
};

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

float compute_bandwidth_gib_per_s(std::size_t bytes_moved, float average_ms) {
    if (average_ms <= 0.0F) {
        return 0.0F;
    }

    const double gib = static_cast<double>(bytes_moved) / (1024.0 * 1024.0 * 1024.0);
    const double seconds = static_cast<double>(average_ms) / 1000.0;
    return static_cast<float>(gib / seconds);
}

Measurement measure_explicit_copy_flow(const std::string& name,
                                       const float* host_input,
                                       float* host_output,
                                       std::size_t element_count,
                                       int iterations,
                                       cudaMemcpyKind host_to_device_kind) {
    const std::size_t byte_count = byte_count_for_float_count(element_count);
    DeviceBuffer device_input = make_device_buffer(element_count);
    DeviceBuffer device_output = make_device_buffer(element_count);
    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;

    check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        check_cuda(cudaMemcpyAsync(device_input.get(), host_input, byte_count,
                                   host_to_device_kind, stream.get()),
                   "cudaMemcpyAsync(host-to-device)");
        launch_transform(device_input.get(), device_output.get(), element_count, stream.get());
        check_cuda(cudaMemcpyAsync(host_output, device_output.get(), byte_count,
                                   cudaMemcpyDeviceToHost, stream.get()),
                   "cudaMemcpyAsync(device-to-host)");
    }
    check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

    const float total_ms = elapsed_ms(start, stop);
    std::size_t bad_index = 0;
    const bool passed = validate_output(host_input, host_output, element_count, &bad_index);
    if (!passed) {
        std::cerr << name << " validation failed at index " << bad_index << '\n';
    }

    return Measurement{name,
                       total_ms / static_cast<float>(iterations),
                       compute_bandwidth_gib_per_s(byte_count * 2U, total_ms / iterations),
                       passed};
}

Measurement measure_mapped_host_flow(float* mapped_input,
                                     float* mapped_output,
                                     std::size_t element_count,
                                     int iterations) {
    float* device_mapped_input = nullptr;
    float* device_mapped_output = nullptr;
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_mapped_input),
                                        mapped_input,
                                        0),
               "cudaHostGetDevicePointer(input)");
    check_cuda(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_mapped_output),
                                        mapped_output,
                                        0),
               "cudaHostGetDevicePointer(output)");

    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;

    check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        launch_transform(device_mapped_input, device_mapped_output, element_count, stream.get());
    }
    check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

    const float total_ms = elapsed_ms(start, stop);
    std::size_t bad_index = 0;
    const bool passed = validate_output(mapped_input, mapped_output, element_count, &bad_index);
    if (!passed) {
        std::cerr << "Mapped pinned validation failed at index " << bad_index << '\n';
    }

    const std::size_t byte_count = byte_count_for_float_count(element_count);
    return Measurement{"mapped pinned: kernel reads/writes host memory",
                       total_ms / static_cast<float>(iterations),
                       compute_bandwidth_gib_per_s(byte_count * 2U, total_ms / iterations),
                       passed};
}

Measurement measure_managed_flow(float* managed_input,
                                 float* managed_output,
                                 std::size_t element_count,
                                 int iterations,
                                 int device_id) {
    fill_input(managed_input, element_count);
    check_cuda(cudaMemset(managed_output, 0, byte_count_for_float_count(element_count)),
               "cudaMemset(managed output)");

    CudaStream stream;
    CudaEvent start;
    CudaEvent stop;

    // Prefetch makes page migration explicit instead of hiding it inside the first kernel launch.
    check_cuda(prefetch_to_device_async(managed_input,
                                        byte_count_for_float_count(element_count),
                                        device_id,
                                        stream.get()),
               "cudaMemPrefetchAsync(input to device)");
    check_cuda(prefetch_to_device_async(managed_output,
                                        byte_count_for_float_count(element_count),
                                        device_id,
                                        stream.get()),
               "cudaMemPrefetchAsync(output to device)");

    check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
    for (int i = 0; i < iterations; ++i) {
        launch_transform(managed_input, managed_output, element_count, stream.get());
    }
    check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");

    check_cuda(prefetch_to_host_async(managed_output,
                                      byte_count_for_float_count(element_count),
                                      stream.get()),
               "cudaMemPrefetchAsync(output to host)");
    stream.synchronize();

    const float total_ms = elapsed_ms(start, stop);
    std::size_t bad_index = 0;
    const bool passed = validate_output(managed_input, managed_output, element_count, &bad_index);
    if (!passed) {
        std::cerr << "Unified Memory validation failed at index " << bad_index << '\n';
    }

    return Measurement{"unified memory: prefetched kernel access",
                       total_ms / static_cast<float>(iterations),
                       0.0F,
                       passed};
}

void print_measurement(const Measurement& measurement) {
    std::cout << std::left << std::setw(kPathColumnWidth) << measurement.name << std::right << std::setw(12)
              << std::fixed << std::setprecision(3) << measurement.average_ms << " ms"
              << std::setw(12);
    if (measurement.bandwidth_gib_per_s > 0.0F) {
        std::cout << std::setprecision(2) << measurement.bandwidth_gib_per_s << " GiB/s";
    } else {
        std::cout << "n/a";
    }
    std::cout << std::setw(10) << (measurement.passed ? "pass" : "fail") << '\n';
}

void print_transfer_note() {
    std::cout << "\nNotes:\n"
              << "- cudaMemcpyAsync from pageable host memory can force staging work and may not "
                 "overlap the way pinned memory can.\n"
              << "- Mapped pinned memory removes explicit cudaMemcpy calls, but a discrete GPU "
                 "still reads and writes host memory across PCIe.\n"
              << "- Unified Memory is convenient for ownership, but page migration must be "
                 "understood before using it in latency-sensitive inference.\n"
              << "- TensorRT enqueue calls use the same stream idea shown here: queue work, then "
                 "synchronize only at the boundary where the CPU needs results.\n";
}

}  // namespace

int run_cuda_memory_demo(const DemoConfig& config) {
    if (config.element_count == 0) {
        throw std::runtime_error("element_count must be positive.");
    }
    if (config.iterations <= 0) {
        throw std::runtime_error("iterations must be positive.");
    }

    check_cuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags(cudaDeviceMapHost)");
    const DeviceInfo device_info = get_device_info();
    check_cuda(cudaSetDevice(device_info.device_id), "cudaSetDevice");

    const std::size_t byte_count = byte_count_for_float_count(config.element_count);
    std::cout << "CUDA device:    " << device_info.device_id << " - " << device_info.name << '\n';
    std::cout << "Mapped host:    " << (device_info.can_map_host_memory ? "supported" : "not supported")
              << '\n';
    std::cout << "GPU type:       " << (device_info.integrated_gpu ? "integrated" : "discrete")
              << '\n';
    std::cout << "Elements:       " << config.element_count << " float32 values\n";
    std::cout << "Buffer size:    " << byte_count << " bytes\n";
    std::cout << "Iterations:     " << config.iterations << "\n\n";

    std::vector<float> pageable_input(config.element_count);
    std::vector<float> pageable_output(config.element_count, 0.0F);
    fill_input(pageable_input.data(), pageable_input.size());

    PinnedHostBuffer pinned_input = make_pinned_host_buffer(config.element_count);
    PinnedHostBuffer pinned_output = make_pinned_host_buffer(config.element_count);
    std::copy(pageable_input.begin(), pageable_input.end(), pinned_input.get());
    std::fill(pinned_output.get(), pinned_output.get() + pinned_output.element_count(), 0.0F);

    std::vector<Measurement> measurements;
    measurements.push_back(measure_explicit_copy_flow("pageable host: H2D + kernel + D2H",
                                                      pageable_input.data(),
                                                      pageable_output.data(),
                                                      config.element_count,
                                                      config.iterations,
                                                      cudaMemcpyHostToDevice));

    measurements.push_back(measure_explicit_copy_flow("pinned host: async H2D + kernel + D2H",
                                                      pinned_input.get(),
                                                      pinned_output.get(),
                                                      config.element_count,
                                                      config.iterations,
                                                      cudaMemcpyHostToDevice));

    if (device_info.can_map_host_memory) {
        PinnedHostBuffer mapped_input = make_mapped_host_buffer(config.element_count);
        PinnedHostBuffer mapped_output = make_mapped_host_buffer(config.element_count);
        std::copy(pageable_input.begin(), pageable_input.end(), mapped_input.get());
        std::fill(mapped_output.get(), mapped_output.get() + mapped_output.element_count(), 0.0F);
        measurements.push_back(measure_mapped_host_flow(mapped_input.get(),
                                                        mapped_output.get(),
                                                        config.element_count,
                                                        config.iterations));
    }

    ManagedBuffer managed_input = make_managed_buffer(config.element_count);
    ManagedBuffer managed_output = make_managed_buffer(config.element_count);
    measurements.push_back(measure_managed_flow(managed_input.get(),
                                                managed_output.get(),
                                                config.element_count,
                                                config.iterations,
                                                device_info.device_id));

    std::cout << std::left << std::setw(kPathColumnWidth) << "Path" << std::right
              << std::setw(15) << "avg time" << std::setw(18) << "copy bandwidth"
              << std::setw(10) << "check" << '\n';
    std::cout << std::string(kPathColumnWidth + 43, '-') << '\n';
    for (const Measurement& measurement : measurements) {
        print_measurement(measurement);
    }

    print_transfer_note();

    const bool all_passed =
        std::all_of(measurements.begin(), measurements.end(), [](const Measurement& measurement) {
            return measurement.passed;
        });
    return all_passed ? 0 : 1;
}
