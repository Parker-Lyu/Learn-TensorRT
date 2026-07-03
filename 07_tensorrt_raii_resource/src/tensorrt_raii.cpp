#include "tensorrt_raii.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lesson07 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

template <typename T>
struct TensorRtDeleter {
    void operator()(T* ptr) const noexcept {
        delete ptr;
    }
};

template <typename T>
using TensorRtPtr = std::unique_ptr<T, TensorRtDeleter<T>>;

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

    CudaStream(CudaStream&& other) noexcept {
        *this = std::move(other);
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                (void)cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

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

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t byte_count) : byte_count_(byte_count) {
        if (byte_count_ == 0) {
            throw std::runtime_error("DeviceBuffer cannot allocate zero bytes.");
        }
        check_cuda(cudaMalloc(&ptr_, byte_count_), "cudaMalloc");
    }

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        *this = std::move(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            byte_count_ = other.byte_count_;
            other.ptr_ = nullptr;
            other.byte_count_ = 0;
        }
        return *this;
    }

    void* get() const {
        return ptr_;
    }

    std::size_t byte_count() const {
        return byte_count_;
    }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            byte_count_ = 0;
        }
    }

    void* ptr_ = nullptr;
    std::size_t byte_count_ = 0;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t byte_count) : byte_count_(byte_count) {
        if (byte_count_ == 0) {
            throw std::runtime_error("PinnedHostBuffer cannot allocate zero bytes.");
        }
        check_cuda(cudaMallocHost(&ptr_, byte_count_), "cudaMallocHost");
        std::fill_n(static_cast<std::uint8_t*>(ptr_), byte_count_, std::uint8_t{0});
    }

    ~PinnedHostBuffer() {
        reset();
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept {
        *this = std::move(other);
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            byte_count_ = other.byte_count_;
            other.ptr_ = nullptr;
            other.byte_count_ = 0;
        }
        return *this;
    }

    void* get() const {
        return ptr_;
    }

private:
    void reset() noexcept {
        if (ptr_ != nullptr) {
            (void)cudaFreeHost(ptr_);
            ptr_ = nullptr;
            byte_count_ = 0;
        }
    }

    void* ptr_ = nullptr;
    std::size_t byte_count_ = 0;
};

struct TensorBuffer {
    TensorReport report;
    std::unique_ptr<DeviceBuffer> device_buffer;
    std::unique_ptr<PinnedHostBuffer> host_buffer;

    void* address() const {
        if (device_buffer) {
            return device_buffer->get();
        }
        return host_buffer->get();
    }
};

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    const std::ifstream::pos_type end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error("Engine file is empty: " + path);
    }

    std::vector<char> bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return bytes;
}

std::string to_string(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT8:
            return "int8";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        case nvinfer1::DataType::kUINT8:
            return "uint8";
        case nvinfer1::DataType::kFP8:
            return "fp8";
    }
    return "unknown";
}

std::string to_string(nvinfer1::TensorIOMode mode) {
    switch (mode) {
        case nvinfer1::TensorIOMode::kINPUT:
            return "input";
        case nvinfer1::TensorIOMode::kOUTPUT:
            return "output";
        case nvinfer1::TensorIOMode::kNONE:
            return "none";
    }
    return "unknown";
}

std::string to_string(nvinfer1::TensorLocation location) {
    switch (location) {
        case nvinfer1::TensorLocation::kDEVICE:
            return "device";
        case nvinfer1::TensorLocation::kHOST:
            return "host";
    }
    return "unknown";
}

std::size_t byte_size(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
    }
    throw std::runtime_error("Unsupported TensorRT data type.");
}

nvinfer1::Dims to_dims(const std::vector<int32_t>& dimensions) {
    if (dimensions.empty() ||
        dimensions.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        throw std::runtime_error("Input shape rank is outside TensorRT Dims limits.");
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(dimensions.size());
    std::copy(dimensions.begin(), dimensions.end(), dims.d);
    return dims;
}

bool has_dynamic_dimension(const nvinfer1::Dims& dims) {
    return std::any_of(dims.d, dims.d + dims.nbDims, [](int64_t value) { return value < 0; });
}

std::vector<int64_t> to_vector(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        throw std::runtime_error("TensorRT returned an invalid tensor shape.");
    }

    std::vector<int64_t> values;
    values.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        values.push_back(dims.d[i]);
    }
    return values;
}

std::size_t checked_volume(const nvinfer1::Dims& dims, const std::string& tensor_name) {
    if (dims.nbDims <= 0) {
        throw std::runtime_error("Tensor " + tensor_name + " has an invalid rank.");
    }

    std::size_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Tensor " + tensor_name +
                                     " still has a dynamic or invalid dimension.");
        }

        const auto dim = static_cast<std::size_t>(dims.d[i]);
        if (volume > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("Tensor " + tensor_name + " has too many elements.");
        }
        volume *= dim;
    }
    return volume;
}

std::size_t checked_byte_count(const nvinfer1::ICudaEngine& engine,
                               const std::string& tensor_name,
                               const nvinfer1::Dims& dims) {
    const std::size_t volume = checked_volume(dims, tensor_name);
    const std::size_t bytes = byte_size(engine.getTensorDataType(tensor_name.c_str()));

    if (volume > std::numeric_limits<std::size_t>::max() / bytes) {
        throw std::runtime_error("Tensor " + tensor_name + " byte count overflowed.");
    }
    return volume * bytes;
}

void apply_input_shapes(nvinfer1::IExecutionContext& context,
                        const std::vector<InputShape>& input_shapes) {
    for (const InputShape& shape : input_shapes) {
        if (!context.setInputShape(shape.tensor_name.c_str(), to_dims(shape.dimensions))) {
            throw std::runtime_error("Failed to set input shape for tensor: " +
                                     shape.tensor_name);
        }
    }
}

TensorBuffer make_tensor_buffer(const nvinfer1::ICudaEngine& engine,
                                const nvinfer1::IExecutionContext& context,
                                const std::string& tensor_name) {
    const nvinfer1::Dims dims = context.getTensorShape(tensor_name.c_str());
    if (has_dynamic_dimension(dims)) {
        throw std::runtime_error("Tensor " + tensor_name +
                                 " has unresolved dynamic dimensions. Pass --input-shape.");
    }

    const nvinfer1::TensorLocation location = engine.getTensorLocation(tensor_name.c_str());
    TensorBuffer buffer;
    buffer.report.name = tensor_name;
    buffer.report.mode = to_string(engine.getTensorIOMode(tensor_name.c_str()));
    buffer.report.location = to_string(location);
    buffer.report.data_type = to_string(engine.getTensorDataType(tensor_name.c_str()));
    buffer.report.dimensions = to_vector(dims);
    buffer.report.byte_count = checked_byte_count(engine, tensor_name, dims);

    if (location == nvinfer1::TensorLocation::kDEVICE) {
        buffer.device_buffer = std::make_unique<DeviceBuffer>(buffer.report.byte_count);
    } else {
        buffer.host_buffer = std::make_unique<PinnedHostBuffer>(buffer.report.byte_count);
    }
    return buffer;
}

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

InferenceReport run_smoke_inference(const RunConfig& config) {
    if (config.engine_path.empty()) {
        throw std::runtime_error("engine_path must not be empty.");
    }
    if (config.warmup_iterations < 0 || config.measured_iterations <= 0) {
        throw std::runtime_error("Invalid warmup or measured iteration count.");
    }

    TensorRtLogger logger;
    const std::vector<char> engine_bytes = read_binary_file(config.engine_path);

    TensorRtPtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    TensorRtPtr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size())};
    if (!engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine: " + config.engine_path);
    }

    TensorRtPtr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    apply_input_shapes(*context, config.input_shapes);

    std::vector<TensorBuffer> buffers;
    buffers.reserve(static_cast<std::size_t>(engine->getNbIOTensors()));

    InferenceReport report;
    report.engine_path = config.engine_path;

    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        if (name == nullptr) {
            throw std::runtime_error("TensorRT returned a null IO tensor name.");
        }

        TensorBuffer buffer = make_tensor_buffer(*engine, *context, name);
        if (!context->setTensorAddress(name, buffer.address())) {
            throw std::runtime_error("Failed to bind tensor address for: " + std::string(name));
        }

        if (buffer.device_buffer) {
            report.total_device_bytes += buffer.device_buffer->byte_count();
        }
        report.tensors.push_back(buffer.report);
        buffers.push_back(std::move(buffer));
    }

    CudaStream stream;
    for (const TensorBuffer& buffer : buffers) {
        if (buffer.device_buffer && buffer.report.mode == "input") {
            check_cuda(cudaMemsetAsync(buffer.device_buffer->get(), 0,
                                       buffer.device_buffer->byte_count(), stream.get()),
                       "cudaMemsetAsync(input)");
        }
    }

    for (int i = 0; i < config.warmup_iterations; ++i) {
        if (!context->enqueueV3(stream.get())) {
            throw std::runtime_error("TensorRT warmup enqueueV3 failed.");
        }
    }
    stream.synchronize();

    CudaEvent start;
    CudaEvent stop;
    check_cuda(cudaEventRecord(start.get(), stream.get()), "cudaEventRecord(start)");
    for (int i = 0; i < config.measured_iterations; ++i) {
        if (!context->enqueueV3(stream.get())) {
            throw std::runtime_error("TensorRT measured enqueueV3 failed.");
        }
    }
    check_cuda(cudaEventRecord(stop.get(), stream.get()), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize(stop)");

    report.average_enqueue_ms =
        elapsed_ms(start, stop) / static_cast<float>(config.measured_iterations);
    return report;
}

}  // namespace lesson07
