#include "scale_shift_plugin.hpp"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <new>

namespace lesson19a {
namespace {
constexpr const char* kName = "ScaleShift";
constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "";

__global__ void scale_shift_kernel(const float* input, float* output,
                                   std::size_t count, float scale, float shift) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = input[index] * scale + shift;
}

void read_fields(const nvinfer1::PluginFieldCollection* fields, float& scale, float& shift) {
    if (!fields) return;
    for (int32_t index = 0; index < fields->nbFields; ++index) {
        const auto& field = fields->fields[index];
        if (!field.name || !field.data || field.type != nvinfer1::PluginFieldType::kFLOAT32 ||
            field.length != 1) {
            continue;
        }
        if (std::strcmp(field.name, "scale") == 0) {
            scale = *static_cast<const float*>(field.data);
        } else if (std::strcmp(field.name, "shift") == 0) {
            shift = *static_cast<const float*>(field.data);
        }
    }
}
}  // namespace

ScaleShiftPlugin::ScaleShiftPlugin(float scale, float shift) : scale_(scale), shift_(shift) {
    initialize_serialization_fields();
}

void ScaleShiftPlugin::initialize_serialization_fields() {
    serialization_fields_.clear();
    serialization_fields_.emplace_back(
        nvinfer1::PluginField{"scale", &scale_, nvinfer1::PluginFieldType::kFLOAT32, 1});
    serialization_fields_.emplace_back(
        nvinfer1::PluginField{"shift", &shift_, nvinfer1::PluginFieldType::kFLOAT32, 1});
    serialization_collection_.nbFields = static_cast<int32_t>(serialization_fields_.size());
    serialization_collection_.fields = serialization_fields_.data();
}

nvinfer1::IPluginCapability* ScaleShiftPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    if (type == nvinfer1::PluginCapabilityType::kCORE) {
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kBUILD) {
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    }
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

ScaleShiftPlugin* ScaleShiftPlugin::clone() noexcept {
    return new (std::nothrow) ScaleShiftPlugin(scale_, shift_);
}

const char* ScaleShiftPlugin::getPluginName() const noexcept { return kName; }
const char* ScaleShiftPlugin::getPluginVersion() const noexcept { return kVersion; }
const char* ScaleShiftPlugin::getPluginNamespace() const noexcept { return kNamespace; }

int32_t ScaleShiftPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t input_count,
                                          const nvinfer1::DynamicPluginTensorDesc*,
                                          int32_t output_count) noexcept {
    return input_count == 1 && output_count == 1 ? 0 : 1;
}

int32_t ScaleShiftPlugin::getOutputDataTypes(nvinfer1::DataType* output_types, int32_t output_count,
                                             const nvinfer1::DataType* input_types,
                                             int32_t input_count) const noexcept {
    if (!output_types || !input_types || input_count != 1 || output_count != 1) return 1;
    output_types[0] = input_types[0];
    return 0;
}

int32_t ScaleShiftPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t input_count,
                                          const nvinfer1::DimsExprs*, int32_t shape_input_count,
                                          nvinfer1::DimsExprs* outputs, int32_t output_count,
                                          nvinfer1::IExprBuilder&) noexcept {
    if (!inputs || !outputs || input_count != 1 || shape_input_count != 0 || output_count != 1) return 1;
    outputs[0] = inputs[0];
    return 0;
}

bool ScaleShiftPlugin::supportsFormatCombination(
    int32_t position, const nvinfer1::DynamicPluginTensorDesc* descriptors,
    int32_t input_count, int32_t output_count) noexcept {
    if (!descriptors || input_count != 1 || output_count != 1 || position < 0 || position >= 2) {
        return false;
    }
    const auto& descriptor = descriptors[position].desc;
    if (descriptor.format != nvinfer1::TensorFormat::kLINEAR) return false;
    return position == 0 ? descriptor.type == nvinfer1::DataType::kFLOAT
                         : descriptor.type == descriptors[0].desc.type;
}

int32_t ScaleShiftPlugin::getNbOutputs() const noexcept { return 1; }
std::size_t ScaleShiftPlugin::getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
                                               const nvinfer1::DynamicPluginTensorDesc*,
                                               int32_t) const noexcept { return 0; }

int32_t ScaleShiftPlugin::onShapeChange(const nvinfer1::PluginTensorDesc*, int32_t input_count,
                                        const nvinfer1::PluginTensorDesc*,
                                        int32_t output_count) noexcept {
    return input_count == 1 && output_count == 1 ? 0 : 1;
}

int32_t ScaleShiftPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                                  const nvinfer1::PluginTensorDesc*, const void* const* inputs,
                                  void* const* outputs, void*, cudaStream_t stream) noexcept {
    if (!input_desc || !inputs || !outputs || !inputs[0] || !outputs[0]) return 1;
    std::size_t count = 1;
    for (int32_t index = 0; index < input_desc[0].dims.nbDims; ++index) {
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

nvinfer1::IPluginV3* ScaleShiftPlugin::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept {
    return clone();
}

const nvinfer1::PluginFieldCollection* ScaleShiftPlugin::getFieldsToSerialize() noexcept {
    return &serialization_collection_;
}

ScaleShiftPluginCreator::ScaleShiftPluginCreator() {
    fields_.emplace_back(
        nvinfer1::PluginField{"scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    fields_.emplace_back(
        nvinfer1::PluginField{"shift", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1});
    collection_.nbFields = static_cast<int32_t>(fields_.size());
    collection_.fields = fields_.data();
}

const char* ScaleShiftPluginCreator::getPluginName() const noexcept { return kName; }
const char* ScaleShiftPluginCreator::getPluginVersion() const noexcept { return kVersion; }
const char* ScaleShiftPluginCreator::getPluginNamespace() const noexcept { return kNamespace; }
const nvinfer1::PluginFieldCollection* ScaleShiftPluginCreator::getFieldNames() noexcept {
    return &collection_;
}

nvinfer1::IPluginV3* ScaleShiftPluginCreator::createPlugin(
    const char*, const nvinfer1::PluginFieldCollection* fields,
    nvinfer1::TensorRTPhase) noexcept {
    float scale = 1.0F;
    float shift = 0.0F;
    read_fields(fields, scale, shift);
    return new (std::nothrow) ScaleShiftPlugin(scale, shift);
}

REGISTER_TENSORRT_PLUGIN(ScaleShiftPluginCreator);

}  // namespace lesson19a

extern "C" bool initScaleShiftPlugin() { return true; }
