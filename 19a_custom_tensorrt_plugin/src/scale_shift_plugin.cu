#include "scale_shift_plugin.hpp"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <new>

namespace lesson19a {
namespace {
constexpr const char* kName = "ScaleShift";
constexpr const char* kVersion = "1";

__global__ void scale_shift_kernel(const float* input, float* output,
                                   std::size_t count, float scale, float shift) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale + shift;
}
}

ScaleShiftPlugin::ScaleShiftPlugin(float scale, float shift) : scale_(scale), shift_(shift) {}
ScaleShiftPlugin::ScaleShiftPlugin(const void* data, std::size_t length) {
    if (length == sizeof(float) * 2) {
        std::memcpy(&scale_, data, sizeof(float));
        std::memcpy(&shift_, static_cast<const char*>(data) + sizeof(float), sizeof(float));
    }
}
const char* ScaleShiftPlugin::getPluginType() const noexcept { return kName; }
const char* ScaleShiftPlugin::getPluginVersion() const noexcept { return kVersion; }
int ScaleShiftPlugin::getNbOutputs() const noexcept { return 1; }
int ScaleShiftPlugin::initialize() noexcept { return 0; }
void ScaleShiftPlugin::terminate() noexcept {}
std::size_t ScaleShiftPlugin::getSerializationSize() const noexcept { return sizeof(float) * 2; }
void ScaleShiftPlugin::serialize(void* buffer) const noexcept {
    std::memcpy(buffer, &scale_, sizeof(float));
    std::memcpy(static_cast<char*>(buffer) + sizeof(float), &shift_, sizeof(float));
}
void ScaleShiftPlugin::destroy() noexcept { delete this; }
ScaleShiftPlugin* ScaleShiftPlugin::clone() const noexcept {
    auto* plugin = new (std::nothrow) ScaleShiftPlugin(scale_, shift_);
    if (plugin) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}
void ScaleShiftPlugin::setPluginNamespace(const char* value) noexcept {
    namespace_ = value ? value : "";
}
const char* ScaleShiftPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }
nvinfer1::DataType ScaleShiftPlugin::getOutputDataType(
    int, const nvinfer1::DataType* input_types, int) const noexcept { return input_types[0]; }
nvinfer1::DimsExprs ScaleShiftPlugin::getOutputDimensions(
    int, const nvinfer1::DimsExprs* inputs, int, nvinfer1::IExprBuilder&) noexcept {
    return inputs[0];
}
bool ScaleShiftPlugin::supportsFormatCombination(
    int position, const nvinfer1::PluginTensorDesc* descriptors, int input_count, int output_count) noexcept {
    if (input_count != 1 || output_count != 1 || position < 0 || position >= 2) return false;
    const auto& descriptor = descriptors[position];
    if (descriptor.format != nvinfer1::TensorFormat::kLINEAR) return false;
    return position == 0 ? descriptor.type == nvinfer1::DataType::kFLOAT
                         : descriptor.type == descriptors[0].type;
}
void ScaleShiftPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                                       const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}
std::size_t ScaleShiftPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                               const nvinfer1::PluginTensorDesc*, int) const noexcept { return 0; }
int ScaleShiftPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                              const nvinfer1::PluginTensorDesc*, const void* const* inputs,
                              void* const* outputs, void*, cudaStream_t stream) noexcept {
    std::size_t count = 1;
    for (int index = 0; index < input_desc[0].dims.nbDims; ++index) {
        if (input_desc[0].dims.d[index] <= 0) return 1;
        count *= static_cast<std::size_t>(input_desc[0].dims.d[index]);
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    scale_shift_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]),
        count, scale_, shift_);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}

ScaleShiftPluginCreator::ScaleShiftPluginCreator() {
    fields_.emplace_back(nvinfer1::PluginField{"scale", nullptr,
        nvinfer1::PluginFieldType::kFLOAT32, 1});
    fields_.emplace_back(nvinfer1::PluginField{"shift", nullptr,
        nvinfer1::PluginFieldType::kFLOAT32, 1});
    collection_.nbFields = static_cast<int>(fields_.size());
    collection_.fields = fields_.data();
}
const char* ScaleShiftPluginCreator::getPluginName() const noexcept { return kName; }
const char* ScaleShiftPluginCreator::getPluginVersion() const noexcept { return kVersion; }
const nvinfer1::PluginFieldCollection* ScaleShiftPluginCreator::getFieldNames() noexcept {
    return &collection_;
}
nvinfer1::IPluginV2* ScaleShiftPluginCreator::createPlugin(
    const char*, const nvinfer1::PluginFieldCollection* fields) noexcept {
    float scale = 1.0F, shift = 0.0F;
    if (fields) {
        for (int index = 0; index < fields->nbFields; ++index) {
            const auto& field = fields->fields[index];
            if (!field.name || !field.data || field.type != nvinfer1::PluginFieldType::kFLOAT32) continue;
            if (std::strcmp(field.name, "scale") == 0) scale = *static_cast<const float*>(field.data);
            if (std::strcmp(field.name, "shift") == 0) shift = *static_cast<const float*>(field.data);
        }
    }
    auto* plugin = new (std::nothrow) ScaleShiftPlugin(scale, shift);
    if (plugin) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}
nvinfer1::IPluginV2* ScaleShiftPluginCreator::deserializePlugin(
    const char*, const void* data, std::size_t length) noexcept {
    auto* plugin = new (std::nothrow) ScaleShiftPlugin(data, length);
    if (plugin) plugin->setPluginNamespace(namespace_.c_str());
    return plugin;
}
void ScaleShiftPluginCreator::setPluginNamespace(const char* value) noexcept {
    namespace_ = value ? value : "";
}
const char* ScaleShiftPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }

REGISTER_TENSORRT_PLUGIN(ScaleShiftPluginCreator);

}  // namespace lesson19a

extern "C" bool initScaleShiftPlugin() { return true; }
