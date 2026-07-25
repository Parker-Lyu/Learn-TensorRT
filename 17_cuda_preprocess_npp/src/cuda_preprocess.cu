#include "preprocess.hpp"

#include <cuda_runtime_api.h>
#include <nppcore.h>
#include <nppi_geometry_transforms.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace lesson17 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}
void check_npp(NppStatus status, const char* operation) {
    if (status != NPP_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with NPP status " +
                                 std::to_string(static_cast<int>(status)));
}

__global__ void bgr_to_rgb_nchw(const unsigned char* input, float* output,
                                int width, int height, int input_step) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::size_t pixel = static_cast<std::size_t>(y) * input_step + x * 3;
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t offset = static_cast<std::size_t>(y) * width + x;
    output[offset] = input[pixel + 2] / 255.0F;
    output[plane + offset] = input[pixel + 1] / 255.0F;
    output[2 * plane + offset] = input[pixel] / 255.0F;
}

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&value_), "cudaEventCreate"); }
    ~Event() { if (value_) (void)cudaEventDestroy(value_); }
    cudaEvent_t get() const { return value_; }
private:
    cudaEvent_t value_{nullptr};
};
float elapsed(const Event& first, const Event& second) {
    float value = 0.0F;
    check_cuda(cudaEventElapsedTime(&value, first.get(), second.get()), "cudaEventElapsedTime");
    return value;
}

NppStreamContext make_npp_stream_context(cudaStream_t stream) {
    NppStreamContext context{};
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&context.nCudaDeviceId), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, context.nCudaDeviceId),
               "cudaGetDeviceProperties");
    context.hStream = stream;
    context.nMultiProcessorCount = properties.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
    context.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
    context.nSharedMemPerBlock = properties.sharedMemPerBlock;
    context.nCudaDevAttrComputeCapabilityMajor = properties.major;
    context.nCudaDevAttrComputeCapabilityMinor = properties.minor;
    check_cuda(cudaStreamGetFlags(stream, &context.nStreamFlags), "cudaStreamGetFlags");
    return context;
}

}  // namespace

struct GpuPreprocessor::Impl {
    Impl(cv::Size source_size, cv::Size target_size, HostMemoryMode memory_mode)
        : source(source_size), target(target_size), mode(memory_mode) {
        if (source.width <= 0 || source.height <= 0 || target.width <= 0 || target.height <= 0)
            throw std::invalid_argument("source and target dimensions must be positive");
        source_bytes = static_cast<std::size_t>(source.width) * source.height * 3;
        resized_bytes = static_cast<std::size_t>(target.width) * target.height * 3;
        output_bytes = static_cast<std::size_t>(target.width) * target.height * 3 * sizeof(float);
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMalloc(&device_resized, resized_bytes), "cudaMalloc resized image");

        if (mode == HostMemoryMode::Mapped) {
            check_cuda(cudaHostAlloc(&host_input, source_bytes, cudaHostAllocMapped),
                       "cudaHostAlloc mapped input");
            check_cuda(cudaHostAlloc(&host_output, output_bytes, cudaHostAllocMapped),
                       "cudaHostAlloc mapped output");
            check_cuda(cudaHostGetDevicePointer(&device_input, host_input, 0),
                       "cudaHostGetDevicePointer input");
            check_cuda(cudaHostGetDevicePointer(&device_output, host_output, 0),
                       "cudaHostGetDevicePointer output");
        } else {
            check_cuda(cudaMalloc(&device_input, source_bytes), "cudaMalloc input");
            check_cuda(cudaMalloc(&device_output, output_bytes), "cudaMalloc output");
            if (mode == HostMemoryMode::Pinned) {
                check_cuda(cudaMallocHost(&host_input, source_bytes), "cudaMallocHost input");
                check_cuda(cudaMallocHost(&host_output, output_bytes), "cudaMallocHost output");
            }
        }
        // CUDA 13 makes NPP stream contexts application-managed. Fill every documented field
        // once and pass it explicitly so this object never changes NPP global stream state.
        npp_context = make_npp_stream_context(stream);
    }

    ~Impl() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (mode == HostMemoryMode::Mapped) {
            if (host_input) (void)cudaFreeHost(host_input);
            if (host_output) (void)cudaFreeHost(host_output);
        } else {
            if (device_input) (void)cudaFree(device_input);
            if (device_output) (void)cudaFree(device_output);
            if (host_input) (void)cudaFreeHost(host_input);
            if (host_output) (void)cudaFreeHost(host_output);
        }
        if (device_resized) (void)cudaFree(device_resized);
        if (stream) (void)cudaStreamDestroy(stream);
    }

    GpuPreprocessResult run(const cv::Mat& image) {
        if (image.empty() || image.type() != CV_8UC3 || image.size() != source || !image.isContinuous())
            throw std::invalid_argument("GPU input must be continuous CV_8UC3 with configured dimensions");
        GpuPreprocessResult result;
        result.tensor_nchw.resize(output_bytes / sizeof(float));
        const auto staging_start = std::chrono::steady_clock::now();
        const void* copy_source = image.data;
        if (mode != HostMemoryMode::Pageable) {
            std::memcpy(host_input, image.data, source_bytes);
            copy_source = host_input;
        }
        const auto staging_stop = std::chrono::steady_clock::now();
        result.timing.host_staging_ms =
            std::chrono::duration<float, std::milli>(staging_stop - staging_start).count();

        Event h2d_start, h2d_stop, preprocess_start, preprocess_stop, d2h_start, d2h_stop;
        check_cuda(cudaEventRecord(h2d_start.get(), stream), "record H2D start");
        if (mode != HostMemoryMode::Mapped)
            check_cuda(cudaMemcpyAsync(device_input, copy_source, source_bytes,
                                       cudaMemcpyHostToDevice, stream), "copy source to device");
        check_cuda(cudaEventRecord(h2d_stop.get(), stream), "record H2D stop");
        check_cuda(cudaEventRecord(preprocess_start.get(), stream), "record preprocess start");
        check_npp(nppiResize_8u_C3R_Ctx(
            static_cast<const Npp8u*>(device_input), source.width * 3,
            {source.width, source.height}, {0, 0, source.width, source.height},
            static_cast<Npp8u*>(device_resized), target.width * 3,
            {target.width, target.height}, {0, 0, target.width, target.height},
            NPPI_INTER_LINEAR, npp_context), "nppiResize_8u_C3R_Ctx");
        const dim3 block(16, 16);
        const dim3 grid((target.width + block.x - 1) / block.x,
                        (target.height + block.y - 1) / block.y);
        bgr_to_rgb_nchw<<<grid, block, 0, stream>>>(
            static_cast<const unsigned char*>(device_resized), static_cast<float*>(device_output),
            target.width, target.height, target.width * 3);
        check_cuda(cudaGetLastError(), "launch bgr_to_rgb_nchw");
        check_cuda(cudaEventRecord(preprocess_stop.get(), stream), "record preprocess stop");
        check_cuda(cudaEventRecord(d2h_start.get(), stream), "record D2H start");
        if (mode != HostMemoryMode::Mapped) {
            void* destination = mode == HostMemoryMode::Pinned ? host_output : result.tensor_nchw.data();
            check_cuda(cudaMemcpyAsync(destination, device_output, output_bytes,
                                       cudaMemcpyDeviceToHost, stream), "copy output to host");
        }
        check_cuda(cudaEventRecord(d2h_stop.get(), stream), "record D2H stop");
        check_cuda(cudaEventSynchronize(d2h_stop.get()), "wait for preprocessing");
        if (mode == HostMemoryMode::Pinned)
            std::memcpy(result.tensor_nchw.data(), host_output, output_bytes);
        else if (mode == HostMemoryMode::Mapped)
            std::memcpy(result.tensor_nchw.data(), host_output, output_bytes);
        result.timing.h2d_ms = elapsed(h2d_start, h2d_stop);
        result.timing.gpu_preprocess_ms = elapsed(preprocess_start, preprocess_stop);
        result.timing.d2h_ms = elapsed(d2h_start, d2h_stop);
        return result;
    }

    cv::Size source;
    cv::Size target;
    HostMemoryMode mode;
    std::size_t source_bytes{0}, resized_bytes{0}, output_bytes{0};
    cudaStream_t stream{nullptr};
    void* host_input{nullptr};
    void* host_output{nullptr};
    void* device_input{nullptr};
    void* device_resized{nullptr};
    void* device_output{nullptr};
    NppStreamContext npp_context{};
};

GpuPreprocessor::GpuPreprocessor(cv::Size source, cv::Size target, HostMemoryMode mode)
    : impl_(std::make_unique<Impl>(source, target, mode)) {}
GpuPreprocessor::~GpuPreprocessor() = default;
GpuPreprocessResult GpuPreprocessor::run(const cv::Mat& bgr) { return impl_->run(bgr); }

}  // namespace lesson17
