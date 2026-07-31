#include "scale_shift_plugin.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) throw std::runtime_error("failed to read engine: " + path);
    std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
    input.seekg(0);
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("failed while reading engine: " + path);
    }
    return bytes;
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) { check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc"); }
    ~DeviceBuffer() { if (pointer_) (void)cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const noexcept { return pointer_; }
private:
    void* pointer_{nullptr};
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~Stream() { if (stream_) (void)cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }
private:
    cudaStream_t stream_{nullptr};
};
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: validate_plugin ENGINE");
        if (!initScaleShiftPlugin()) throw std::runtime_error("plugin library initialization failed");
        Logger logger;
        const auto bytes = read_file(argv[1]);
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("failed to create TensorRT runtime");
        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("engine deserialization failed");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context || engine->getNbIOTensors() != 2) {
            throw std::runtime_error("expected one input and one output tensor");
        }

        const char* input_name = nullptr;
        const char* output_name = nullptr;
        for (int32_t index = 0; index < engine->getNbIOTensors(); ++index) {
            const char* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) input_name = name;
            else output_name = name;
        }
        if (!input_name || !output_name) throw std::runtime_error("failed to identify engine I/O tensors");

        const std::vector<float> input{-2.0F, -0.5F, 0.5F, 3.0F};
        std::vector<float> output(input.size());
        DeviceBuffer device_input(input.size() * sizeof(float));
        DeviceBuffer device_output(output.size() * sizeof(float));
        Stream stream;
        if (!context->setTensorAddress(input_name, device_input.get()) ||
            !context->setTensorAddress(output_name, device_output.get())) {
            throw std::runtime_error("failed to set TensorRT tensor addresses");
        }
        check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), input.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream.get()), "copy input");
        if (!context->enqueueV3(stream.get())) throw std::runtime_error("enqueueV3 failed");
        check_cuda(cudaMemcpyAsync(output.data(), device_output.get(), output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream.get()), "copy output");
        check_cuda(cudaStreamSynchronize(stream.get()), "wait for output");

        float max_error = 0.0F;
        for (std::size_t index = 0; index < input.size(); ++index) {
            max_error = std::max(max_error,
                std::abs(output[index] - (input[index] * 2.0F - 1.0F)));
        }
        if (max_error > 1e-6F) throw std::runtime_error("plugin numerical validation failed");
        std::cout << "ScaleShift IPluginV3 max_abs=" << max_error << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
