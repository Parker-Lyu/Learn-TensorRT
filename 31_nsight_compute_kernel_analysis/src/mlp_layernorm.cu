#include "mlp_layernorm.hpp"

#include <cublas_v2.h>
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

constexpr unsigned int kFullWarpMask = 0xffffffffU;
constexpr int kExpectedWarpSize = 32;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) (void)cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() noexcept { return data_; }
private:
    void* data_{nullptr};
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~Stream() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() {
        if (event_ != nullptr) (void)cudaEventDestroy(event_);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_{nullptr};
};

class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t stream) {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        try {
            check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
            check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                         "cublasSetMathMode(CUBLAS_PEDANTIC_MATH)");
        } catch (...) {
            (void)cublasDestroy(handle_);
            throw;
        }
    }
    ~CublasHandle() {
        if (handle_ != nullptr) (void)cublasDestroy(handle_);
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_{nullptr};
};

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

__device__ float warp_sum(float value) {
    for (int offset = kExpectedWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    return value;
}

__device__ void block_sum_pair(float& sum, float& square_sum) {
    __shared__ float warp_sums[kExpectedWarpSize];
    __shared__ float warp_square_sums[kExpectedWarpSize];
    const int lane = threadIdx.x % kExpectedWarpSize;
    const int warp = threadIdx.x / kExpectedWarpSize;
    const int warp_count = (blockDim.x + kExpectedWarpSize - 1) / kExpectedWarpSize;

    sum = warp_sum(sum);
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = sum;
        warp_square_sums[warp] = square_sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < warp_count ? warp_sums[lane] : 0.0F;
        square_sum = lane < warp_count ? warp_square_sums[lane] : 0.0F;
        sum = warp_sum(sum);
        square_sum = warp_sum(square_sum);
        if (lane == 0) {
            warp_sums[0] = sum;
            warp_square_sums[0] = square_sum;
        }
    }
    __syncthreads();
    sum = warp_sums[0];
    square_sum = warp_square_sums[0];
}

__global__ void layer_norm_stats_kernel(const float* input, float* means,
                                        float* inverse_stddevs, int features, float epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const std::size_t row_offset = static_cast<std::size_t>(row) * features;
    float sum = 0.0F;
    float square_sum = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < features; column += blockDim.x) {
        const float value = input[row_offset + column];
        sum += value;
        square_sum += value * value;
    }
    block_sum_pair(sum, square_sum);
    if (threadIdx.x == 0) {
        const float mean = sum / static_cast<float>(features);
        const float variance = fmaxf(square_sum / static_cast<float>(features) - mean * mean, 0.0F);
        means[row] = mean;
        inverse_stddevs[row] = rsqrtf(variance + epsilon);
    }
}

__global__ void layer_norm_apply_kernel(const float* input, const float* means,
                                        const float* inverse_stddevs, const float* gamma,
                                        const float* beta, float* output, int rows, int features) {
    const std::size_t count = static_cast<std::size_t>(rows) * features;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const int column = static_cast<int>(index % static_cast<std::size_t>(features));
        const int row = static_cast<int>(index / static_cast<std::size_t>(features));
        output[index] = (input[index] - means[row]) * inverse_stddevs[row] * gamma[column] +
                        beta[column];
    }
}

__global__ void layer_norm_fused_kernel(const float* input, const float* gamma,
                                        const float* beta, float* output, int features,
                                        float epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const std::size_t row_offset = static_cast<std::size_t>(row) * features;
    float sum = 0.0F;
    float square_sum = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < features; column += blockDim.x) {
        const float value = input[row_offset + column];
        sum += value;
        square_sum += value * value;
    }
    block_sum_pair(sum, square_sum);
    const float mean = sum / static_cast<float>(features);
    const float variance = fmaxf(square_sum / static_cast<float>(features) - mean * mean, 0.0F);
    const float inverse_stddev = rsqrtf(variance + epsilon);
    for (int column = static_cast<int>(threadIdx.x); column < features; column += blockDim.x) {
        const std::size_t index = row_offset + column;
        output[index] = (input[index] - mean) * inverse_stddev * gamma[column] + beta[column];
    }
}

int occupancy_block_size(const void* kernel) {
    int minimum_grid_size = 0;
    int suggested_block_size = 0;
    check_cuda(cudaOccupancyMaxPotentialBlockSize(
                   &minimum_grid_size, &suggested_block_size, kernel, 0, 0),
               "cudaOccupancyMaxPotentialBlockSize");
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.warpSize != kExpectedWarpSize) {
        throw std::runtime_error("the reduction requires CUDA's 32-thread warp contract");
    }
    suggested_block_size = std::min(suggested_block_size, properties.maxThreadsPerBlock);
    const int aligned = (suggested_block_size / properties.warpSize) * properties.warpSize;
    return std::max(properties.warpSize, aligned);
}

struct LaunchConfig {
    int reduction_block_size;
    int apply_block_size;
    int apply_grid_size;
};

LaunchConfig make_launch_config(int rows, int features, LayerNormVariant variant) {
    const void* reduction_kernel = variant == LayerNormVariant::Baseline
        ? reinterpret_cast<const void*>(layer_norm_stats_kernel)
        : reinterpret_cast<const void*>(layer_norm_fused_kernel);
    const int occupancy_reduction_block_size = occupancy_block_size(reduction_kernel);
    const int threads_for_one_row =
        ((features + kExpectedWarpSize - 1) / kExpectedWarpSize) * kExpectedWarpSize;
    const int reduction_block_size = std::min(
        occupancy_reduction_block_size, std::max(kExpectedWarpSize, threads_for_one_row));
    const int apply_block_size = occupancy_block_size(
        reinterpret_cast<const void*>(layer_norm_apply_kernel));
    int minimum_grid_size = 0;
    int ignored_block_size = 0;
    check_cuda(cudaOccupancyMaxPotentialBlockSize(
                   &minimum_grid_size, &ignored_block_size, layer_norm_apply_kernel, 0, 0),
               "query apply-kernel occupancy");
    const std::size_t count = static_cast<std::size_t>(rows) * features;
    const std::size_t required_grid =
        (count + static_cast<std::size_t>(apply_block_size) - 1) / apply_block_size;
    const int apply_grid_size = static_cast<int>(std::min<std::size_t>(
        required_grid, static_cast<std::size_t>(std::max(1, minimum_grid_size))));
    return {reduction_block_size, apply_block_size, apply_grid_size};
}

void launch_layer_norm(LayerNormVariant variant, const LaunchConfig& launch,
                       const float* input, const float* gamma, const float* beta,
                       float* output, float* means, float* inverse_stddevs,
                       int rows, int features, float epsilon, cudaStream_t stream) {
    if (variant == LayerNormVariant::Baseline) {
        layer_norm_stats_kernel<<<rows, launch.reduction_block_size, 0, stream>>>(
            input, means, inverse_stddevs, features, epsilon);
        layer_norm_apply_kernel<<<launch.apply_grid_size, launch.apply_block_size, 0, stream>>>(
            input, means, inverse_stddevs, gamma, beta, output, rows, features);
    } else {
        layer_norm_fused_kernel<<<rows, launch.reduction_block_size, 0, stream>>>(
            input, gamma, beta, output, features, epsilon);
    }
    check_cuda(cudaPeekAtLastError(), "launch LayerNorm kernel");
}

void linear(cublasHandle_t handle, const float* input, const float* weights, float* output,
            int rows, int input_features, int output_features) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    // Row-major C = A * B is evaluated as column-major C^T = B^T * A^T.
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             output_features, rows, input_features,
                             &alpha, weights, output_features,
                             input, input_features, &beta, output, output_features),
                 "cublasSgemm");
}

void run_network(LayerNormVariant variant, const LaunchConfig& launch, cublasHandle_t handle,
                 cudaStream_t stream, const BenchmarkConfig& config, const float* input,
                 const float* first_weights, const float* gamma, const float* beta,
                 const float* second_weights, float* hidden, float* normalized, float* output,
                 float* means, float* inverse_stddevs) {
    NvtxRange network_range(variant == LayerNormVariant::Baseline
                                ? "network_baseline" : "network_fused");
    {
        NvtxRange range("linear_1");
        linear(handle, input, first_weights, hidden, config.rows,
               config.input_features, config.hidden_features);
    }
    {
        NvtxRange range(variant == LayerNormVariant::Baseline
                            ? "layernorm_baseline" : "layernorm_fused");
        launch_layer_norm(variant, launch, hidden, gamma, beta, normalized, means,
                          inverse_stddevs, config.rows, config.hidden_features,
                          config.epsilon, stream);
    }
    {
        NvtxRange range("linear_2");
        linear(handle, normalized, second_weights, output, config.rows,
               config.hidden_features, config.output_features);
    }
}

template <typename Operation>
TimingSummary measure(Operation operation, int warmup_iterations, int measured_iterations,
                      cudaStream_t stream) {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) operation();
    check_cuda(cudaStreamSynchronize(stream), "wait for warmup");
    Event start;
    Event end;
    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(measured_iterations));
    for (int iteration = 0; iteration < measured_iterations; ++iteration) {
        check_cuda(cudaEventRecord(start.get(), stream), "record timing start");
        operation();
        check_cuda(cudaEventRecord(end.get(), stream), "record timing end");
        check_cuda(cudaEventSynchronize(end.get()), "wait for timing end");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), end.get()),
                   "read elapsed time");
        samples.push_back(elapsed_ms);
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](float fraction) {
        const auto index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<float>(samples.size())) - 1.0F);
        return samples[std::min(index, samples.size() - 1)];
    };
    return {
        samples.front(),
        std::accumulate(samples.begin(), samples.end(), 0.0F) /
            static_cast<float>(samples.size()),
        percentile(0.50F),
        percentile(0.90F),
        samples.back(),
    };
}

std::vector<float> make_values(std::size_t count, float scale, int offset) {
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        const float phase = static_cast<float>((index * 17 + static_cast<std::size_t>(offset)) % 251);
        values[index] = std::sin(phase * 0.071F) * scale;
    }
    return values;
}

std::vector<float> cpu_linear(const std::vector<float>& input,
                              const std::vector<float>& weights,
                              int rows, int input_features, int output_features) {
    std::vector<float> output(static_cast<std::size_t>(rows) * output_features, 0.0F);
    for (int row = 0; row < rows; ++row) {
        for (int output_column = 0; output_column < output_features; ++output_column) {
            float sum = 0.0F;
            for (int input_column = 0; input_column < input_features; ++input_column) {
                sum += input[static_cast<std::size_t>(row) * input_features + input_column] *
                       weights[static_cast<std::size_t>(input_column) * output_features +
                               output_column];
            }
            output[static_cast<std::size_t>(row) * output_features + output_column] = sum;
        }
    }
    return output;
}

std::vector<float> cpu_reference(const BenchmarkConfig& config,
                                 const std::vector<float>& input,
                                 const std::vector<float>& first_weights,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& beta,
                                 const std::vector<float>& second_weights) {
    auto hidden = cpu_linear(input, first_weights, config.rows,
                             config.input_features, config.hidden_features);
    std::vector<float> normalized(hidden.size());
    for (int row = 0; row < config.rows; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * config.hidden_features;
        float sum = 0.0F;
        float square_sum = 0.0F;
        for (int column = 0; column < config.hidden_features; ++column) {
            const float value = hidden[offset + column];
            sum += value;
            square_sum += value * value;
        }
        const float mean = sum / static_cast<float>(config.hidden_features);
        const float variance = std::max(
            square_sum / static_cast<float>(config.hidden_features) - mean * mean, 0.0F);
        const float inverse_stddev = 1.0F / std::sqrt(variance + config.epsilon);
        for (int column = 0; column < config.hidden_features; ++column) {
            normalized[offset + column] = (hidden[offset + column] - mean) * inverse_stddev *
                                          gamma[column] + beta[column];
        }
    }
    return cpu_linear(normalized, second_weights, config.rows,
                      config.hidden_features, config.output_features);
}

void validate_config(const BenchmarkConfig& config,
                     const std::vector<LayerNormVariant>& variants) {
    if (config.rows <= 0 || config.input_features <= 0 || config.hidden_features <= 0 ||
        config.output_features <= 0 || config.warmup_iterations < 0 ||
        config.measured_iterations <= 0 || !std::isfinite(config.epsilon) ||
        config.epsilon <= 0.0F || variants.empty()) {
        throw std::invalid_argument("network dimensions, epsilon, iterations, and variants must be positive");
    }
    const auto checked_product = [](int left, int right) {
        const auto product = static_cast<std::size_t>(left) * static_cast<std::size_t>(right);
        if (product > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::invalid_argument("network tensor size overflows addressable memory");
        }
        return product;
    };
    (void)checked_product(config.rows, config.input_features);
    (void)checked_product(config.rows, config.hidden_features);
    (void)checked_product(config.rows, config.output_features);
    (void)checked_product(config.input_features, config.hidden_features);
    (void)checked_product(config.hidden_features, config.output_features);
}

}  // namespace

const char* variant_name(LayerNormVariant variant) noexcept {
    switch (variant) {
        case LayerNormVariant::Baseline: return "baseline";
        case LayerNormVariant::Fused: return "fused";
    }
    return "unknown";
}

LayerNormVariant parse_variant(const std::string& name) {
    for (const auto variant : all_variants()) {
        if (name == variant_name(variant)) return variant;
    }
    throw std::invalid_argument("unknown LayerNorm variant: " + name);
}

std::vector<LayerNormVariant> all_variants() {
    return {LayerNormVariant::Baseline, LayerNormVariant::Fused};
}

std::vector<BenchmarkResult> benchmark_network(
    const BenchmarkConfig& config, const std::vector<LayerNormVariant>& variants) {
    validate_config(config, variants);
    const std::size_t input_count = static_cast<std::size_t>(config.rows) * config.input_features;
    const std::size_t hidden_count = static_cast<std::size_t>(config.rows) * config.hidden_features;
    const std::size_t output_count = static_cast<std::size_t>(config.rows) * config.output_features;
    const auto input = make_values(input_count, 0.75F, 3);
    const auto first_weights = make_values(
        static_cast<std::size_t>(config.input_features) * config.hidden_features,
        0.25F / std::sqrt(static_cast<float>(config.input_features)), 11);
    auto gamma = make_values(config.hidden_features, 0.15F, 19);
    auto beta = make_values(config.hidden_features, 0.05F, 29);
    for (float& value : gamma) value += 1.0F;
    const auto second_weights = make_values(
        static_cast<std::size_t>(config.hidden_features) * config.output_features,
        0.25F / std::sqrt(static_cast<float>(config.hidden_features)), 37);
    const auto reference = cpu_reference(
        config, input, first_weights, gamma, beta, second_weights);

    DeviceBuffer device_input(input_count * sizeof(float));
    DeviceBuffer device_first_weights(first_weights.size() * sizeof(float));
    DeviceBuffer device_gamma(gamma.size() * sizeof(float));
    DeviceBuffer device_beta(beta.size() * sizeof(float));
    DeviceBuffer device_second_weights(second_weights.size() * sizeof(float));
    DeviceBuffer device_hidden(hidden_count * sizeof(float));
    DeviceBuffer device_normalized(hidden_count * sizeof(float));
    DeviceBuffer device_output(output_count * sizeof(float));
    DeviceBuffer device_means(static_cast<std::size_t>(config.rows) * sizeof(float));
    DeviceBuffer device_inverse_stddevs(static_cast<std::size_t>(config.rows) * sizeof(float));
    Stream stream;
    CublasHandle handle(stream.get());

    check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), input_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream.get()), "upload input");
    check_cuda(cudaMemcpyAsync(device_first_weights.get(), first_weights.data(),
                               first_weights.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream.get()), "upload first weights");
    check_cuda(cudaMemcpyAsync(device_gamma.get(), gamma.data(), gamma.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.get()), "upload gamma");
    check_cuda(cudaMemcpyAsync(device_beta.get(), beta.data(), beta.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream.get()), "upload beta");
    check_cuda(cudaMemcpyAsync(device_second_weights.get(), second_weights.data(),
                               second_weights.size() * sizeof(float), cudaMemcpyHostToDevice,
                               stream.get()), "upload second weights");
    check_cuda(cudaStreamSynchronize(stream.get()), "wait for model upload");

    std::vector<BenchmarkResult> results;
    for (const auto variant : variants) {
        const LaunchConfig launch = make_launch_config(
            config.rows, config.hidden_features, variant);
        // Populate the LayerNorm input once before measuring the operator in isolation.
        linear(handle.get(), static_cast<const float*>(device_input.get()),
               static_cast<const float*>(device_first_weights.get()),
               static_cast<float*>(device_hidden.get()), config.rows,
               config.input_features, config.hidden_features);
        check_cuda(cudaStreamSynchronize(stream.get()), "prepare LayerNorm input");

        TimingSummary layernorm_timing;
        if (config.scope != MeasurementScope::Network) {
            layernorm_timing = measure([&] {
                NvtxRange range(variant == LayerNormVariant::Baseline
                                    ? "layernorm_baseline_only" : "layernorm_fused_only");
                launch_layer_norm(
                    variant, launch, static_cast<const float*>(device_hidden.get()),
                    static_cast<const float*>(device_gamma.get()),
                    static_cast<const float*>(device_beta.get()),
                    static_cast<float*>(device_normalized.get()),
                    static_cast<float*>(device_means.get()),
                    static_cast<float*>(device_inverse_stddevs.get()),
                    config.rows, config.hidden_features, config.epsilon, stream.get());
            }, config.warmup_iterations, config.measured_iterations, stream.get());
        }

        const auto network_operation = [&] {
            run_network(
                variant, launch, handle.get(), stream.get(), config,
                static_cast<const float*>(device_input.get()),
                static_cast<const float*>(device_first_weights.get()),
                static_cast<const float*>(device_gamma.get()),
                static_cast<const float*>(device_beta.get()),
                static_cast<const float*>(device_second_weights.get()),
                static_cast<float*>(device_hidden.get()),
                static_cast<float*>(device_normalized.get()),
                static_cast<float*>(device_output.get()),
                static_cast<float*>(device_means.get()),
                static_cast<float*>(device_inverse_stddevs.get()));
        };
        TimingSummary network_timing;
        if (config.scope != MeasurementScope::LayerNorm) {
            network_timing = measure(network_operation, config.warmup_iterations,
                                     config.measured_iterations, stream.get());
        } else {
            // Produce a complete output for correctness after the isolated measurement.
            network_operation();
            check_cuda(cudaStreamSynchronize(stream.get()), "wait for validation network");
        }

        std::vector<float> output(output_count);
        check_cuda(cudaMemcpyAsync(output.data(), device_output.get(), output_count * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream.get()), "download network output");
        check_cuda(cudaStreamSynchronize(stream.get()), "wait for output download");
        float maximum_error = 0.0F;
        double error_sum = 0.0;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float error = std::abs(output[index] - reference[index]);
            maximum_error = std::max(maximum_error, error);
            error_sum += error;
        }
        results.push_back({
            variant,
            layernorm_timing,
            network_timing,
            maximum_error,
            static_cast<float>(error_sum / static_cast<double>(output.size())),
            variant == LayerNormVariant::Baseline ? 2 : 1,
            launch.reduction_block_size,
            variant == LayerNormVariant::Baseline ? launch.apply_block_size : 0,
        });
    }
    return results;
}

}  // namespace lesson31
