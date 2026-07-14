#include "scale_shift_plugin.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

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
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
class Logger final : public nvinfer1::ILogger {
public: void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
}};
std::vector<char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) throw std::runtime_error("failed to read engine: " + path);
    std::vector<char> bytes(static_cast<std::size_t>(input.tellg()));
    input.seekg(0); input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: validate_plugin ENGINE");
        if (!initScaleShiftPlugin()) throw std::runtime_error("plugin library initialization failed");
        Logger logger;
        const auto bytes = read_file(argv[1]);
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
        std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("engine deserialization failed");
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
        if (!context || engine->getNbBindings() != 2) throw std::runtime_error("unexpected engine bindings");
        const int input_index = engine->bindingIsInput(0) ? 0 : 1;
        const int output_index = 1 - input_index;
        std::vector<float> input{-2.0F, -0.5F, 0.5F, 3.0F};
        std::vector<float> output(input.size());
        void* bindings[2]{};
        check_cuda(cudaMalloc(&bindings[input_index], input.size() * sizeof(float)), "cudaMalloc input");
        check_cuda(cudaMalloc(&bindings[output_index], output.size() * sizeof(float)), "cudaMalloc output");
        cudaStream_t stream{};
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        check_cuda(cudaMemcpyAsync(bindings[input_index], input.data(), input.size() * sizeof(float),
                                   cudaMemcpyHostToDevice, stream), "copy input");
        if (!context->enqueueV2(bindings, stream, nullptr)) throw std::runtime_error("enqueueV2 failed");
        check_cuda(cudaMemcpyAsync(output.data(), bindings[output_index], output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream), "copy output");
        check_cuda(cudaStreamSynchronize(stream), "wait for output");
        float max_error = 0.0F;
        for (std::size_t index = 0; index < input.size(); ++index)
            max_error = std::max(max_error, std::abs(output[index] - (input[index] * 2.0F - 1.0F)));
        (void)cudaStreamDestroy(stream);
        (void)cudaFree(bindings[input_index]);
        (void)cudaFree(bindings[output_index]);
        if (max_error > 1e-6F) throw std::runtime_error("plugin numerical validation failed");
        std::cout << "ScaleShift plugin max_abs=" << max_error << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
