#include "tensorrt_runner.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lesson10 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(status));
    }
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

template <typename T>
using TensorRtPtr = std::unique_ptr<T>;

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

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() {
        check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDefault),
                   "cudaEventCreateWithFlags");
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
        if (byte_count == 0) {
            throw std::runtime_error("DeviceBuffer cannot allocate zero bytes.");
        }
        check_cuda(cudaMalloc(&ptr_, byte_count), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const {
        return ptr_;
    }

    std::size_t byte_count() const {
        return byte_count_;
    }

private:
    void* ptr_ = nullptr;
    std::size_t byte_count_ = 0;
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
    std::vector<char> data(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(data.data(), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return data;
}

std::size_t data_type_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kFP8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
    }
    throw std::runtime_error("Unsupported TensorRT tensor data type.");
}

std::vector<int64_t> dims_to_vector(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        throw std::runtime_error("TensorRT returned invalid dimensions.");
    }
    std::vector<int64_t> result;
    result.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("Only fully static tensor shapes are supported in lesson 10.");
        }
        result.push_back(dims.d[i]);
    }
    return result;
}

std::size_t volume(const std::vector<int64_t>& shape) {
    std::size_t result = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || result > std::numeric_limits<std::size_t>::max() /
                                  static_cast<std::size_t>(dim)) {
            throw std::runtime_error("Tensor shape volume overflowed.");
        }
        result *= static_cast<std::size_t>(dim);
    }
    return result;
}

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&ms, start.get(), stop.get()), "cudaEventElapsedTime");
    return ms;
}

}  // namespace

struct TensorRtRunner::Impl {
    explicit Impl(const std::string& path) {
        const std::vector<char> bytes = read_binary_file(path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) {
            throw std::runtime_error("Failed to deserialize TensorRT engine: " + path);
        }
        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tensor_name = engine->getIOTensorName(i);
            if (tensor_name == nullptr) {
                throw std::runtime_error("TensorRT returned a null IO tensor name.");
            }
            const std::string name = tensor_name;
            const std::vector<int64_t> shape =
                dims_to_vector(context->getTensorShape(tensor_name));
            const std::size_t bytes_per_tensor =
                volume(shape) * data_type_size(engine->getTensorDataType(tensor_name));
            if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
                input.name = name;
                input.shape = shape;
                input.byte_count = bytes_per_tensor;
            } else if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT) {
                output.name = name;
                output.shape = shape;
                output.byte_count = bytes_per_tensor;
            }
        }
        if (input.name.empty() || output.name.empty()) {
            throw std::runtime_error("Lesson 10 expects one input tensor and one output tensor.");
        }
        if (engine->getTensorDataType(input.name.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output.name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Lesson 10 expects float32 input and output tensors.");
        }

        input_device = std::make_unique<DeviceBuffer>(input.byte_count);
        output_device = std::make_unique<DeviceBuffer>(output.byte_count);
        if (!context->setTensorAddress(input.name.c_str(), input_device->get()) ||
            !context->setTensorAddress(output.name.c_str(), output_device->get())) {
            throw std::runtime_error("Failed to bind TensorRT tensor addresses.");
        }
    }

    Logger logger;
    TensorRtPtr<nvinfer1::IRuntime> runtime{nullptr};
    TensorRtPtr<nvinfer1::ICudaEngine> engine{nullptr};
    TensorRtPtr<nvinfer1::IExecutionContext> context{nullptr};
    TensorInfo input;
    TensorInfo output;
    std::unique_ptr<DeviceBuffer> input_device;
    std::unique_ptr<DeviceBuffer> output_device;
};

TensorRtRunner::TensorRtRunner(const std::string& engine_path)
    : impl_(std::make_unique<Impl>(engine_path)) {}

TensorRtRunner::~TensorRtRunner() = default;

std::vector<int64_t> TensorRtRunner::input_shape() const {
    return impl_->input.shape;
}

std::string TensorRtRunner::input_name() const {
    return impl_->input.name;
}

std::string TensorRtRunner::output_name() const {
    return impl_->output.name;
}

InferenceOutput TensorRtRunner::infer(const std::vector<float>& input_tensor) {
    if (input_tensor.size() * sizeof(float) != impl_->input.byte_count) {
        throw std::runtime_error("Input tensor size does not match TensorRT engine input.");
    }

    CudaStream stream;
    CudaEvent h2d_start;
    CudaEvent h2d_stop;
    CudaEvent infer_start;
    CudaEvent infer_stop;
    CudaEvent d2h_start;
    CudaEvent d2h_stop;

    check_cuda(cudaEventRecord(h2d_start.get(), stream.get()), "cudaEventRecord(h2d_start)");
    check_cuda(cudaMemcpyAsync(impl_->input_device->get(), input_tensor.data(), impl_->input.byte_count,
                               cudaMemcpyHostToDevice, stream.get()),
               "cudaMemcpyAsync(input)");
    check_cuda(cudaEventRecord(h2d_stop.get(), stream.get()), "cudaEventRecord(h2d_stop)");

    check_cuda(cudaEventRecord(infer_start.get(), stream.get()), "cudaEventRecord(infer_start)");
    if (!impl_->context->enqueueV3(stream.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    }
    check_cuda(cudaEventRecord(infer_stop.get(), stream.get()), "cudaEventRecord(infer_stop)");

    InferenceOutput output;
    output.output_name = impl_->output.name;
    output.output_shape = impl_->output.shape;
    output.values.resize(impl_->output.byte_count / sizeof(float));
    check_cuda(cudaEventRecord(d2h_start.get(), stream.get()), "cudaEventRecord(d2h_start)");
    check_cuda(cudaMemcpyAsync(output.values.data(), impl_->output_device->get(), impl_->output.byte_count,
                               cudaMemcpyDeviceToHost, stream.get()),
               "cudaMemcpyAsync(output)");
    check_cuda(cudaEventRecord(d2h_stop.get(), stream.get()), "cudaEventRecord(d2h_stop)");
    check_cuda(cudaEventSynchronize(d2h_stop.get()), "cudaEventSynchronize(d2h_stop)");

    output.h2d_ms = elapsed_ms(h2d_start, h2d_stop);
    output.enqueue_ms = elapsed_ms(infer_start, infer_stop);
    output.d2h_ms = elapsed_ms(d2h_start, d2h_stop);
    return output;
}

}  // namespace lesson10
