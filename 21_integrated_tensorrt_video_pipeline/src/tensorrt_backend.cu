#include "tensorrt_backend.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lesson21 {
namespace {

class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

template <class T> struct TrtDelete { void operator()(T* value) const { delete value; } };

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_npp(NppStatus status, const char* operation) {
    if (status != NPP_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": NPP status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

std::vector<char> read_engine(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open engine: " + path);
    const std::streamsize size = input.tellg();
    if (size <= 0) throw std::runtime_error("engine is empty: " + path);
    input.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!input.read(bytes.data(), size)) throw std::runtime_error("cannot read engine: " + path);
    return bytes;
}

NppStreamContext npp_context(cudaStream_t stream) {
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "get CUDA device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
    NppStreamContext context{};
    context.hStream = stream;
    context.nCudaDeviceId = device;
    context.nMultiProcessorCount = properties.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
    context.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
    context.nSharedMemPerBlock = properties.sharedMemPerBlock;
    context.nCudaDevAttrComputeCapabilityMajor = properties.major;
    context.nCudaDevAttrComputeCapabilityMinor = properties.minor;
    context.nStreamFlags = cudaStreamNonBlocking;
    return context;
}

__global__ void normalize(const unsigned char* source, float* destination,
                          int width, int height, int batch_index) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    const std::size_t output = static_cast<std::size_t>(batch_index) * 3 * plane;
    destination[output + pixel] = source[pixel * 3 + 2] / 255.0F;
    destination[output + plane + pixel] = source[pixel * 3 + 1] / 255.0F;
    destination[output + 2 * plane + pixel] = source[pixel * 3] / 255.0F;
}

void replace_device_buffer(void*& buffer, std::size_t& capacity, std::size_t required) {
    if (capacity >= required) return;
    void* replacement = nullptr;
    check_cuda(cudaMalloc(&replacement, required), "allocate reusable device buffer");
    if (buffer != nullptr) check_cuda(cudaFree(buffer), "release old device buffer");
    buffer = replacement;
    capacity = required;
}

void replace_pinned_buffer(void*& buffer, std::size_t& capacity, std::size_t required) {
    if (capacity >= required) return;
    void* replacement = nullptr;
    check_cuda(cudaMallocHost(&replacement, required), "allocate reusable pinned buffer");
    if (buffer != nullptr) check_cuda(cudaFreeHost(buffer), "release old pinned buffer");
    buffer = replacement;
    capacity = required;
}

struct Slot {
    cudaStream_t stream{};
    cudaEvent_t done{};
    cudaEvent_t h2d_start{};
    cudaEvent_t h2d_end{};
    cudaEvent_t preprocess_start{};
    cudaEvent_t preprocess_end{};
    cudaEvent_t inference_start{};
    cudaEvent_t inference_end{};
    cudaEvent_t d2h_start{};
    cudaEvent_t d2h_end{};
    NppStreamContext npp{};
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context;
    void* input{};
    void* output{};
    void* device_source{};
    void* device_letterbox{};
    void* pinned_source{};
    void* pinned_output{};
    std::size_t input_capacity{};
    std::size_t output_capacity{};
    std::size_t source_capacity{};
    std::size_t letterbox_capacity{};
    std::size_t pinned_source_capacity{};
    std::size_t pinned_output_capacity{};
    std::size_t source_stride{};
    std::size_t output_elements{};
    BatchMetadata metadata;
    double host_staging_ms{};
    double capacity_growth_ms{};

    ~Slot() {
        if (stream != nullptr) cudaStreamSynchronize(stream);
        if (input != nullptr) cudaFree(input);
        if (output != nullptr) cudaFree(output);
        if (device_source != nullptr) cudaFree(device_source);
        if (device_letterbox != nullptr) cudaFree(device_letterbox);
        if (pinned_source != nullptr) cudaFreeHost(pinned_source);
        if (pinned_output != nullptr) cudaFreeHost(pinned_output);
        for (cudaEvent_t event : {done, h2d_start, h2d_end, preprocess_start, preprocess_end,
                                  inference_start, inference_end, d2h_start, d2h_end}) {
            if (event != nullptr) cudaEventDestroy(event);
        }
        if (stream != nullptr) cudaStreamDestroy(stream);
    }
};

}  // namespace

struct TensorRtBackend::Impl {
    explicit Impl(const std::string& path, std::size_t slot_count, cv::Size size)
        : input_size(size), slot_pool(slot_count) {
        if (slot_count == 0 || size.width <= 0 || size.height <= 0) {
            throw std::invalid_argument("invalid backend capacity");
        }
        const std::vector<char> bytes = read_engine(path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("create TensorRT runtime failed");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("deserialize TensorRT engine failed");
        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }
        if (input_name.empty() || output_name.empty()) {
            throw std::runtime_error("expected exactly one input and one output tensor");
        }
        slots.reserve(slot_count);
        for (std::size_t index = 0; index < slot_count; ++index) {
            auto slot = std::make_unique<Slot>();
            check_cuda(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking),
                       "create slot stream");
            for (cudaEvent_t* event : {&slot->done, &slot->h2d_start, &slot->h2d_end,
                                       &slot->preprocess_start, &slot->preprocess_end,
                                       &slot->inference_start, &slot->inference_end,
                                       &slot->d2h_start, &slot->d2h_end}) {
                check_cuda(cudaEventCreate(event), "create slot event");
            }
            slot->npp = npp_context(slot->stream);
            slot->context.reset(engine->createExecutionContext());
            if (!slot->context) throw std::runtime_error("create execution context failed");
            slots.push_back(std::move(slot));
        }
    }

    void ensure_capacity(Slot& slot, const std::vector<cv::Mat>& images,
                         std::size_t input_bytes, std::size_t output_bytes) {
        const auto started = Clock::now();
        std::size_t source_stride = 0;
        for (const cv::Mat& image : images) {
            source_stride = std::max(source_stride, image.total() * image.elemSize());
        }
        slot.source_stride = source_stride;
        const std::size_t source_bytes = source_stride * images.size();
        const std::size_t letterbox_bytes =
            static_cast<std::size_t>(input_size.area()) * 3 * images.size();
        const bool growth_required = slot.input_capacity < input_bytes ||
            slot.output_capacity < output_bytes || slot.source_capacity < source_bytes ||
            slot.letterbox_capacity < letterbox_bytes ||
            slot.pinned_source_capacity < source_bytes ||
            slot.pinned_output_capacity < output_bytes;
        if (!growth_required) {
            slot.capacity_growth_ms = 0.0;
            return;
        }
        replace_device_buffer(slot.input, slot.input_capacity, input_bytes);
        replace_device_buffer(slot.output, slot.output_capacity, output_bytes);
        replace_device_buffer(slot.device_source, slot.source_capacity, source_bytes);
        replace_device_buffer(slot.device_letterbox, slot.letterbox_capacity, letterbox_bytes);
        replace_pinned_buffer(slot.pinned_source, slot.pinned_source_capacity, source_bytes);
        replace_pinned_buffer(slot.pinned_output, slot.pinned_output_capacity, output_bytes);
        slot.capacity_growth_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    }

    void submit(std::size_t index, const std::vector<cv::Mat>& images, BatchMetadata metadata) {
        if (index >= slots.size() || images.empty() || images.size() > 4 ||
            metadata.frames.size() != images.size()) {
            throw std::invalid_argument("invalid slot, batch, or metadata");
        }
        if (slot_pool.state(index) != SlotState::Reserved) {
            throw std::logic_error("backend submit requires a reserved slot");
        }
        Slot& slot = *slots[index];
        try {
            if (const char* failure = std::getenv("LESSON21_FAIL_SUBMIT_BATCH");
                failure != nullptr && metadata.batch_id == std::stoull(failure)) {
                throw std::runtime_error("injected submit failure");
            }
            for (const cv::Mat& image : images) {
                if (image.empty() || image.type() != CV_8UC3) {
                    throw std::invalid_argument("input must be CV_8UC3");
                }
            }
            const nvinfer1::Dims4 shape(static_cast<int>(images.size()), 3,
                                       input_size.height, input_size.width);
            if (!slot.context->setInputShape(input_name.c_str(), shape)) {
                throw std::invalid_argument("batch is outside the TensorRT profile");
            }
            const nvinfer1::Dims output_shape = slot.context->getTensorShape(output_name.c_str());
            std::size_t output_elements = 1;
            for (int dimension = 0; dimension < output_shape.nbDims; ++dimension) {
                if (output_shape.d[dimension] <= 0) {
                    throw std::invalid_argument("output shape is unresolved or invalid");
                }
                output_elements *= static_cast<std::size_t>(output_shape.d[dimension]);
            }
            const std::size_t input_elements =
                images.size() * 3ULL * static_cast<std::size_t>(input_size.area());
            if (std::getenv("LESSON21_FAIL_INSUFFICIENT_CAPACITY") != nullptr) {
                throw std::invalid_argument("injected insufficient slot capacity");
            }
            ensure_capacity(slot, images, input_elements * sizeof(float),
                            output_elements * sizeof(float));
            slot.output_elements = output_elements;
            slot.metadata = metadata;

            const auto staging_started = Clock::now();
            auto* staging = static_cast<unsigned char*>(slot.pinned_source);
            for (std::size_t batch = 0; batch < images.size(); ++batch) {
                const cv::Mat& image = images[batch];
                unsigned char* destination = staging + batch * slot.source_stride;
                const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * image.elemSize();
                for (int row = 0; row < image.rows; ++row) {
                    std::memcpy(destination + static_cast<std::size_t>(row) * row_bytes,
                                image.ptr(row), row_bytes);
                }
            }
            slot.host_staging_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - staging_started).count();

            check_cuda(cudaEventRecord(slot.h2d_start, slot.stream), "record H2D start");
            for (std::size_t batch = 0; batch < images.size(); ++batch) {
                const std::size_t bytes = images[batch].total() * images[batch].elemSize();
                check_cuda(cudaMemcpyAsync(
                    static_cast<unsigned char*>(slot.device_source) + batch * slot.source_stride,
                    staging + batch * slot.source_stride, bytes,
                    cudaMemcpyHostToDevice, slot.stream), "upload pinned frame");
            }
            check_cuda(cudaEventRecord(slot.h2d_end, slot.stream), "record H2D end");
            check_cuda(cudaEventRecord(slot.preprocess_start, slot.stream),
                       "record preprocessing start");
            const std::size_t letterbox_stride = static_cast<std::size_t>(input_size.area()) * 3;
            for (std::size_t batch = 0; batch < images.size(); ++batch) {
                const cv::Mat& image = images[batch];
                auto* source = static_cast<Npp8u*>(slot.device_source) + batch * slot.source_stride;
                auto* letterbox = static_cast<Npp8u*>(slot.device_letterbox) +
                    batch * letterbox_stride;
                check_cuda(cudaMemsetAsync(letterbox, 114, letterbox_stride, slot.stream),
                           "fill letterbox");
                const float scale = std::min(
                    static_cast<float>(input_size.width) / image.cols,
                    static_cast<float>(input_size.height) / image.rows);
                const int resized_width = std::max(1, static_cast<int>(std::round(image.cols * scale)));
                const int resized_height = std::max(1, static_cast<int>(std::round(image.rows * scale)));
                const int pad_x = (input_size.width - resized_width) / 2;
                const int pad_y = (input_size.height - resized_height) / 2;
                slot.metadata.frames[batch].transform =
                    {scale, static_cast<float>(pad_x), static_cast<float>(pad_y),
                     image.cols, image.rows};
                Npp8u* destination = letterbox +
                    (static_cast<std::size_t>(pad_y) * input_size.width + pad_x) * 3;
                check_npp(nppiResize_8u_C3R_Ctx(
                    source, image.cols * 3, {image.cols, image.rows},
                    {0, 0, image.cols, image.rows}, destination, input_size.width * 3,
                    {input_size.width, input_size.height},
                    {0, 0, resized_width, resized_height}, NPPI_INTER_LINEAR, slot.npp),
                    "NPP resize");
                normalize<<<dim3((input_size.width + 15) / 16, (input_size.height + 15) / 16),
                            dim3(16, 16), 0, slot.stream>>>(
                    letterbox, static_cast<float*>(slot.input),
                    input_size.width, input_size.height, static_cast<int>(batch));
                check_cuda(cudaGetLastError(), "launch normalize kernel");
            }
            check_cuda(cudaEventRecord(slot.preprocess_end, slot.stream),
                       "record preprocessing end");
            if (std::getenv("LESSON21_FAIL_TENSOR_ADDRESS") != nullptr ||
                !slot.context->setTensorAddress(input_name.c_str(), slot.input) ||
                !slot.context->setTensorAddress(output_name.c_str(), slot.output)) {
                throw std::runtime_error("bind TensorRT tensors failed");
            }
            check_cuda(cudaEventRecord(slot.inference_start, slot.stream),
                       "record inference start");
            if (std::getenv("LESSON21_FAIL_ENQUEUE") != nullptr ||
                !slot.context->enqueueV3(slot.stream)) {
                throw std::runtime_error("enqueueV3 failed");
            }
            check_cuda(cudaEventRecord(slot.inference_end, slot.stream),
                       "record inference end");
            check_cuda(cudaEventRecord(slot.d2h_start, slot.stream), "record D2H start");
            check_cuda(cudaMemcpyAsync(slot.pinned_output, slot.output,
                                       output_elements * sizeof(float),
                                       cudaMemcpyDeviceToHost, slot.stream),
                       "download output to pinned memory");
            check_cuda(cudaEventRecord(slot.d2h_end, slot.stream), "record D2H end");
            check_cuda(cudaEventRecord(slot.done, slot.stream), "record completion event");
            slot_pool.mark_submitted(index, std::move(metadata));
        } catch (...) {
            cudaStreamSynchronize(slot.stream);
            slot_pool.fail(index);
            throw;
        }
    }

    bool ready(std::size_t index) const {
        if (index >= slots.size()) throw std::out_of_range("slot index is out of range");
        if (slot_pool.state(index) != SlotState::Submitted) return false;
        const cudaError_t status = cudaEventQuery(slots[index]->done);
        if (status == cudaErrorNotReady) return false;
        check_cuda(status, "query slot completion");
        return true;
    }

    GpuBatchResult collect(std::size_t index) {
        Slot& slot = *slots.at(index);
        slot_pool.begin_collection(index);
        try {
            check_cuda(cudaEventSynchronize(slot.done), "wait for slot completion");
            GpuBatchResult result;
            result.metadata = slot.metadata;
            result.output.assign(static_cast<const float*>(slot.pinned_output),
                                 static_cast<const float*>(slot.pinned_output) + slot.output_elements);
            const nvinfer1::Dims shape = slot.context->getTensorShape(output_name.c_str());
            for (int dimension = 0; dimension < shape.nbDims; ++dimension) {
                result.output_shape.push_back(shape.d[dimension]);
            }
            result.host_staging_ms = slot.host_staging_ms;
            result.capacity_growth_ms = slot.capacity_growth_ms;
            check_cuda(cudaEventElapsedTime(&result.h2d_ms, slot.h2d_start, slot.h2d_end),
                       "measure H2D");
            check_cuda(cudaEventElapsedTime(&result.preprocess_ms,
                                            slot.preprocess_start, slot.preprocess_end),
                       "measure preprocessing");
            check_cuda(cudaEventElapsedTime(&result.inference_ms,
                                            slot.inference_start, slot.inference_end),
                       "measure inference");
            check_cuda(cudaEventElapsedTime(&result.d2h_ms, slot.d2h_start, slot.d2h_end),
                       "measure D2H");
            slot_pool.release(index);
            return result;
        } catch (...) {
            slot_pool.fail(index);
            throw;
        }
    }

    cv::Size input_size;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine;
    std::string input_name;
    std::string output_name;
    SlotPool slot_pool;
    std::vector<std::unique_ptr<Slot>> slots;
};

TensorRtBackend::TensorRtBackend(const std::string& engine, std::size_t slot_count, cv::Size size)
    : impl_(std::make_unique<Impl>(engine, slot_count, size)) {}
TensorRtBackend::~TensorRtBackend() = default;
std::optional<std::size_t> TensorRtBackend::try_reserve() { return impl_->slot_pool.try_reserve(); }
std::size_t TensorRtBackend::reserve() { return impl_->slot_pool.reserve(); }
void TensorRtBackend::submit(std::size_t slot, const std::vector<cv::Mat>& images,
                             BatchMetadata metadata) {
    impl_->submit(slot, images, std::move(metadata));
}
bool TensorRtBackend::ready(std::size_t slot) const { return impl_->ready(slot); }
GpuBatchResult TensorRtBackend::collect(std::size_t slot) { return impl_->collect(slot); }
std::size_t TensorRtBackend::available_slots() const { return impl_->slot_pool.available(); }

RuntimeIdentity TensorRtBackend::identity() const {
    int device = 0;
    int runtime = 0;
    int driver = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "get identity device");
    check_cuda(cudaGetDeviceProperties(&properties, device), "get identity properties");
    check_cuda(cudaRuntimeGetVersion(&runtime), "get CUDA runtime version");
    check_cuda(cudaDriverGetVersion(&driver), "get CUDA driver version");
    return {properties.name, properties.major, properties.minor,
            NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
            runtime, driver};
}

}  // namespace lesson21
