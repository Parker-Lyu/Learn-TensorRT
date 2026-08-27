#include "kernel_variants.hpp"

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace lesson31 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

__global__ void bgr_to_rgb_nchw_2d(const unsigned char* input, float* output,
                                   int width, int height, int input_step) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::size_t input_offset = static_cast<std::size_t>(y) * input_step + x * 3;
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t output_offset = static_cast<std::size_t>(y) * width + x;
    output[output_offset] = input[input_offset + 2] / 255.0F;
    output[plane + output_offset] = input[input_offset + 1] / 255.0F;
    output[2 * plane + output_offset] = input[input_offset] / 255.0F;
}

__global__ void bgr_to_rgb_nchw_linear(const unsigned char* input, float* output,
                                       int width, int height, int input_step) {
    const std::size_t count = static_cast<std::size_t>(width) * height;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const int y = static_cast<int>(index / static_cast<std::size_t>(width));
        const int x = static_cast<int>(index % static_cast<std::size_t>(width));
        const std::size_t input_offset = static_cast<std::size_t>(y) * input_step + x * 3;
        output[index] = input[input_offset + 2] / 255.0F;
        output[count + index] = input[input_offset + 1] / 255.0F;
        output[2 * count + index] = input[input_offset] / 255.0F;
    }
}

__global__ void bgr_to_rgb_nchw_vectorized(const unsigned char* input, float* output,
                                           int width, int height, int input_step) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int first_x = group * 4;
    if (first_x >= width || y >= height) return;

    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const unsigned char* row = input + static_cast<std::size_t>(y) * input_step;
    if (first_x + 3 < width && input_step % 4 == 0) {
        const uchar4* packed = reinterpret_cast<const uchar4*>(row + first_x * 3);
        const uchar4 a = packed[0];
        const uchar4 b = packed[1];
        const uchar4 c = packed[2];
        const unsigned char pixels[12] = {
            a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w};
        for (int local = 0; local < 4; ++local) {
            const std::size_t output_offset =
                static_cast<std::size_t>(y) * width + first_x + local;
            output[output_offset] = pixels[local * 3 + 2] / 255.0F;
            output[plane + output_offset] = pixels[local * 3 + 1] / 255.0F;
            output[2 * plane + output_offset] = pixels[local * 3] / 255.0F;
        }
        return;
    }

    for (int local = 0; local < 4 && first_x + local < width; ++local) {
        const int x = first_x + local;
        const std::size_t input_offset = static_cast<std::size_t>(x) * 3;
        const std::size_t output_offset = static_cast<std::size_t>(y) * width + x;
        output[output_offset] = row[input_offset + 2] / 255.0F;
        output[plane + output_offset] = row[input_offset + 1] / 255.0F;
        output[2 * plane + output_offset] = row[input_offset] / 255.0F;
    }
}

__global__ void bgr_to_rgb_hwc_unfused(const unsigned char* input, float* intermediate,
                                       int width, int height, int input_step) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (index >= count) return;
    const int y = static_cast<int>(index / static_cast<std::size_t>(width));
    const int x = static_cast<int>(index % static_cast<std::size_t>(width));
    const std::size_t input_offset = static_cast<std::size_t>(y) * input_step + x * 3;
    intermediate[index * 3] = input[input_offset + 2] / 255.0F;
    intermediate[index * 3 + 1] = input[input_offset + 1] / 255.0F;
    intermediate[index * 3 + 2] = input[input_offset] / 255.0F;
}

__global__ void hwc_to_chw_unfused(const float* intermediate, float* output,
                                   std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = intermediate[index * 3];
    output[count + index] = intermediate[index * 3 + 1];
    output[2 * count + index] = intermediate[index * 3 + 2];
}

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { if (event_ != nullptr) (void)cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const noexcept { return event_; }
private:
    cudaEvent_t event_{nullptr};
};

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_ != nullptr) (void)cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() noexcept { return data_; }
private:
    void* data_{nullptr};
};

int launch_variant(KernelVariant variant, const unsigned char* input, float* output,
                   float* intermediate, int width, int height, cudaStream_t stream) {
    const int input_step = width * 3;
    const std::size_t count = static_cast<std::size_t>(width) * height;
    if (variant == KernelVariant::Baseline16x16 || variant == KernelVariant::Block32x8) {
        const dim3 block = variant == KernelVariant::Baseline16x16 ? dim3(16, 16) : dim3(32, 8);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        bgr_to_rgb_nchw_2d<<<grid, block, 0, stream>>>(input, output, width, height, input_step);
        return 1;
    }
    if (variant == KernelVariant::Linear) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        bgr_to_rgb_nchw_linear<<<blocks, threads, 0, stream>>>(
            input, output, width, height, input_step);
        return 1;
    }
    if (variant == KernelVariant::Vectorized) {
        const dim3 block(32, 8);
        const int groups = (width + 3) / 4;
        const dim3 grid((groups + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        bgr_to_rgb_nchw_vectorized<<<grid, block, 0, stream>>>(
            input, output, width, height, input_step);
        return 1;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    bgr_to_rgb_hwc_unfused<<<blocks, threads, 0, stream>>>(
        input, intermediate, width, height, input_step);
    hwc_to_chw_unfused<<<blocks, threads, 0, stream>>>(intermediate, output, count);
    return 2;
}

float percentile(std::vector<float> values, float fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<float>(values.size())) - 1.0F);
    return values[std::min(index, values.size() - 1)];
}

TimingSummary summarize(const std::vector<float>& values) {
    return {
        *std::min_element(values.begin(), values.end()),
        std::accumulate(values.begin(), values.end(), 0.0F) /
            static_cast<float>(values.size()),
        percentile(values, 0.50F),
        percentile(values, 0.90F),
        *std::max_element(values.begin(), values.end()),
    };
}

std::vector<unsigned char> make_input(int width, int height) {
    std::vector<unsigned char> input(static_cast<std::size_t>(width) * height * 3);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<unsigned char>((index * 37 + 17) % 256);
    }
    return input;
}

std::vector<float> make_reference(const std::vector<unsigned char>& input,
                                  int width, int height) {
    const std::size_t count = static_cast<std::size_t>(width) * height;
    std::vector<float> reference(count * 3);
    for (std::size_t index = 0; index < count; ++index) {
        reference[index] = input[index * 3 + 2] / 255.0F;
        reference[count + index] = input[index * 3 + 1] / 255.0F;
        reference[2 * count + index] = input[index * 3] / 255.0F;
    }
    return reference;
}

}  // namespace

const char* variant_name(KernelVariant variant) noexcept {
    switch (variant) {
        case KernelVariant::Baseline16x16: return "baseline_16x16";
        case KernelVariant::Block32x8: return "block_32x8";
        case KernelVariant::Linear: return "linear";
        case KernelVariant::Vectorized: return "vectorized";
        case KernelVariant::Unfused: return "unfused";
    }
    return "unknown";
}

KernelVariant parse_variant(const std::string& name) {
    for (const auto variant : all_variants()) {
        if (name == variant_name(variant)) return variant;
    }
    throw std::invalid_argument("unknown kernel variant: " + name);
}

std::vector<KernelVariant> all_variants() {
    return {KernelVariant::Unfused, KernelVariant::Baseline16x16,
            KernelVariant::Block32x8, KernelVariant::Linear, KernelVariant::Vectorized};
}

std::vector<BenchmarkResult> benchmark_variants(
    const BenchmarkConfig& config, const std::vector<KernelVariant>& variants) {
    if (config.width <= 0 || config.height <= 0 || config.warmup_iterations < 0 ||
        config.measured_iterations <= 0 || variants.empty()) {
        throw std::invalid_argument("benchmark dimensions, iterations, and variants are invalid");
    }
    const auto width = static_cast<std::size_t>(config.width);
    const auto height = static_cast<std::size_t>(config.height);
    if (config.width > std::numeric_limits<int>::max() / 3 ||
        width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / (3 * sizeof(float))) {
        throw std::invalid_argument("benchmark dimensions overflow buffer or row-stride limits");
    }

    const auto input = make_input(config.width, config.height);
    const auto reference = make_reference(input, config.width, config.height);
    const std::size_t output_bytes = reference.size() * sizeof(float);
    DeviceBuffer device_input(input.size());
    DeviceBuffer device_output(output_bytes);
    DeviceBuffer device_intermediate(output_bytes);
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    try {
        check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), input.size(),
                                   cudaMemcpyHostToDevice, stream), "upload benchmark input");
        check_cuda(cudaStreamSynchronize(stream), "wait for benchmark upload");

        std::vector<BenchmarkResult> results;
        for (const auto variant : variants) {
            const std::string range_name = std::string("variant_") + variant_name(variant);
            NvtxRange variant_range(range_name.c_str());
            for (int iteration = 0; iteration < config.warmup_iterations; ++iteration) {
                NvtxRange iteration_range("warmup_iteration");
                (void)launch_variant(
                    variant, static_cast<const unsigned char*>(device_input.get()),
                    static_cast<float*>(device_output.get()),
                    static_cast<float*>(device_intermediate.get()), config.width, config.height,
                    stream);
            }
            check_cuda(cudaGetLastError(), "launch warmup kernel");
            check_cuda(cudaStreamSynchronize(stream), "wait for warmup kernels");

            Event start;
            Event stop;
            std::vector<float> samples;
            samples.reserve(static_cast<std::size_t>(config.measured_iterations));
            int launches = 0;
            for (int iteration = 0; iteration < config.measured_iterations; ++iteration) {
                NvtxRange iteration_range("measured_iteration");
                check_cuda(cudaEventRecord(start.get(), stream), "record benchmark start");
                launches = launch_variant(
                    variant, static_cast<const unsigned char*>(device_input.get()),
                    static_cast<float*>(device_output.get()),
                    static_cast<float*>(device_intermediate.get()), config.width, config.height,
                    stream);
                check_cuda(cudaEventRecord(stop.get(), stream), "record benchmark stop");
                check_cuda(cudaGetLastError(), "launch measured kernel");
                check_cuda(cudaEventSynchronize(stop.get()), "wait for measured kernel");
                float elapsed_ms = 0.0F;
                check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()),
                           "measure kernel duration");
                samples.push_back(elapsed_ms);
            }
            std::vector<float> output(reference.size());
            check_cuda(cudaMemcpy(output.data(), device_output.get(), output_bytes,
                                  cudaMemcpyDeviceToHost), "download benchmark output");
            float maximum_error = 0.0F;
            double error_sum = 0.0;
            for (std::size_t index = 0; index < output.size(); ++index) {
                const float error = std::abs(output[index] - reference[index]);
                maximum_error = std::max(maximum_error, error);
                error_sum += error;
            }
            results.push_back({variant, summarize(samples), maximum_error,
                               static_cast<float>(error_sum / output.size()), launches});
        }
        check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return results;
    } catch (...) {
        (void)cudaStreamDestroy(stream);
        throw;
    }
}

}  // namespace lesson31
