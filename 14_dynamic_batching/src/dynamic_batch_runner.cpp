#include "dynamic_batch_runner.hpp"

#include "batch_layout.hpp"

#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lesson14 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
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

std::vector<char> read_engine(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) {
        throw std::runtime_error("failed to open non-empty engine: " + path);
    }
    std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
    input.seekg(0);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("failed to read engine: " + path);
    }
    return bytes;
}

std::vector<int64_t> to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> shape;
    for (int index = 0; index < dims.nbDims; ++index) {
        shape.push_back(dims.d[index]);
    }
    return shape;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes == 0) {
            throw std::invalid_argument("device allocation must not be empty");
        }
        check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (pointer_) (void)cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const { return pointer_; }
private:
    void* pointer_{nullptr};
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&value_), "cudaStreamCreate"); }
    ~Stream() { if (value_) (void)cudaStreamDestroy(value_); }
    cudaStream_t get() const { return value_; }
private:
    cudaStream_t value_{nullptr};
};

class Event {
public:
    Event() { check_cuda(cudaEventCreate(&value_), "cudaEventCreate"); }
    ~Event() { if (value_) (void)cudaEventDestroy(value_); }
    cudaEvent_t get() const { return value_; }
private:
    cudaEvent_t value_{nullptr};
};

float elapsed(const Event& start, const Event& stop) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return milliseconds;
}

}  // namespace

struct DynamicBatchRunner::Impl {
    explicit Impl(const std::string& path) {
        const auto bytes = read_engine(path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("failed to create TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("failed to deserialize engine: " + path);
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("failed to create execution context");

        for (int index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input = name;
            else output = name;
        }
        if (input.empty() || output.empty()) {
            throw std::runtime_error("expected one input and one output tensor");
        }
        if (engine->getTensorDataType(input.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("lesson 14 expects float32 input and output tensors");
        }
    }

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input;
    std::string output;
    Stream stream;
};

DynamicBatchRunner::DynamicBatchRunner(const std::string& engine_path)
    : impl_(std::make_unique<Impl>(engine_path)) {}
DynamicBatchRunner::~DynamicBatchRunner() = default;

InferenceTiming DynamicBatchRunner::infer(const std::vector<float>& input,
                                          std::size_t batch_size) {
    if (batch_size == 0 || batch_size > 4) {
        throw std::invalid_argument("batch size must be in [1, 4]");
    }
    const nvinfer1::Dims4 input_dims(static_cast<int>(batch_size), 3, 640, 640);
    if (!impl_->context->setInputShape(impl_->input.c_str(), input_dims)) {
        throw std::runtime_error("batch shape is outside the engine optimization profile");
    }
    const auto input_shape = to_shape(impl_->context->getTensorShape(impl_->input.c_str()));
    const auto output_shape = to_shape(impl_->context->getTensorShape(impl_->output.c_str()));
    const std::size_t input_elements = checked_volume(input_shape);
    const std::size_t output_elements = checked_volume(output_shape);
    if (input.size() != input_elements) {
        throw std::invalid_argument("input element count does not match the runtime batch shape");
    }

    DeviceBuffer device_input(input_elements * sizeof(float));
    DeviceBuffer device_output(output_elements * sizeof(float));
    if (!impl_->context->setTensorAddress(impl_->input.c_str(), device_input.get()) ||
        !impl_->context->setTensorAddress(impl_->output.c_str(), device_output.get())) {
        throw std::runtime_error("failed to bind TensorRT tensor addresses");
    }
    Event h2d_start, h2d_stop, compute_start, compute_stop, d2h_start, d2h_stop;
    check_cuda(cudaEventRecord(h2d_start.get(), impl_->stream.get()), "record h2d start");
    check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), input_elements * sizeof(float),
                               cudaMemcpyHostToDevice, impl_->stream.get()), "copy input");
    check_cuda(cudaEventRecord(h2d_stop.get(), impl_->stream.get()), "record h2d stop");
    check_cuda(cudaEventRecord(compute_start.get(), impl_->stream.get()), "record compute start");
    if (!impl_->context->enqueueV3(impl_->stream.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    check_cuda(cudaEventRecord(compute_stop.get(), impl_->stream.get()), "record compute stop");
    std::vector<float> output(output_elements);
    check_cuda(cudaEventRecord(d2h_start.get(), impl_->stream.get()), "record d2h start");
    check_cuda(cudaMemcpyAsync(output.data(), device_output.get(), output_elements * sizeof(float),
                               cudaMemcpyDeviceToHost, impl_->stream.get()), "copy output");
    check_cuda(cudaEventRecord(d2h_stop.get(), impl_->stream.get()), "record d2h stop");
    check_cuda(cudaEventSynchronize(d2h_stop.get()), "wait for inference");

    return InferenceTiming{batch_size, elapsed(h2d_start, h2d_stop),
                           elapsed(compute_start, compute_stop), elapsed(d2h_start, d2h_stop),
                           output_shape, std::accumulate(output.begin(), output.end(), 0.0)};
}

std::string DynamicBatchRunner::input_name() const { return impl_->input; }
std::string DynamicBatchRunner::output_name() const { return impl_->output; }

RuntimeIdentity DynamicBatchRunner::runtime_identity() const {
    int device = 0;
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    return RuntimeIdentity{properties.name, properties.major, properties.minor,
                           NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
                           runtime_version, driver_version};
}

}  // namespace lesson14
