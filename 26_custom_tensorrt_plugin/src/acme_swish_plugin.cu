#include "acme_swish_plugin.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <new>

namespace lesson26 {
namespace {
constexpr const char* kName = "AcmeSwish";
constexpr const char* kVersion = "1";
constexpr const char* kNamespace = "com.acme";

__global__ void acme_swish_kernel(const float* input, float* output, std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = input[index];
        output[index] = value / (1.0F + expf(-value));
    }
}
}  // namespace

nvinfer1::IPluginCapability* AcmeSwishPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
    if (type == nvinfer1::PluginCapabilityType::kCORE) return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    if (type == nvinfer1::PluginCapabilityType::kBUILD) return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME) return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    return nullptr;
}
AcmeSwishPlugin* AcmeSwishPlugin::clone() noexcept { return new (std::nothrow) AcmeSwishPlugin(); }
const char* AcmeSwishPlugin::getPluginName() const noexcept { return kName; }
const char* AcmeSwishPlugin::getPluginVersion() const noexcept { return kVersion; }
const char* AcmeSwishPlugin::getPluginNamespace() const noexcept { return kNamespace; }

int32_t AcmeSwishPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t inputs,
    const nvinfer1::DynamicPluginTensorDesc*, int32_t outputs) noexcept {
    return inputs == 1 && outputs == 1 ? 0 : 1;
}
int32_t AcmeSwishPlugin::getOutputDataTypes(nvinfer1::DataType* output_types, int32_t outputs,
    const nvinfer1::DataType* input_types, int32_t inputs) const noexcept {
    if (!output_types || !input_types || inputs != 1 || outputs != 1 || input_types[0] != nvinfer1::DataType::kFLOAT) return 1;
    output_types[0] = input_types[0];
    return 0;
}
int32_t AcmeSwishPlugin::getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t input_count,
    const nvinfer1::DimsExprs*, int32_t shape_inputs, nvinfer1::DimsExprs* outputs,
    int32_t output_count, nvinfer1::IExprBuilder&) noexcept {
    if (!inputs || !outputs || input_count != 1 || shape_inputs != 0 || output_count != 1) return 1;
    outputs[0] = inputs[0];
    return 0;
}
bool AcmeSwishPlugin::supportsFormatCombination(int32_t position,
    const nvinfer1::DynamicPluginTensorDesc* descriptors, int32_t inputs, int32_t outputs) noexcept {
    if (!descriptors || inputs != 1 || outputs != 1 || position < 0 || position >= 2) return false;
    return descriptors[position].desc.format == nvinfer1::TensorFormat::kLINEAR &&
           descriptors[position].desc.type == nvinfer1::DataType::kFLOAT;
}
int32_t AcmeSwishPlugin::getNbOutputs() const noexcept { return 1; }
std::size_t AcmeSwishPlugin::getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
    const nvinfer1::DynamicPluginTensorDesc*, int32_t) const noexcept { return 0; }
int32_t AcmeSwishPlugin::onShapeChange(const nvinfer1::PluginTensorDesc*, int32_t inputs,
    const nvinfer1::PluginTensorDesc*, int32_t outputs) noexcept { return inputs == 1 && outputs == 1 ? 0 : 1; }

int32_t AcmeSwishPlugin::enqueue(const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc*, const void* const* inputs, void* const* outputs,
    void*, cudaStream_t stream) noexcept {
    if (!input_desc || !inputs || !outputs || !inputs[0] || !outputs[0]) return 1;
    std::size_t count = 1;
    for (int32_t index = 0; index < input_desc[0].dims.nbDims; ++index) {
        if (input_desc[0].dims.d[index] <= 0) return 1;
        count *= static_cast<std::size_t>(input_desc[0].dims.d[index]);
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    acme_swish_kernel<<<blocks, threads, 0, stream>>>(static_cast<const float*>(inputs[0]),
        static_cast<float*>(outputs[0]), count);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
}
nvinfer1::IPluginV3* AcmeSwishPlugin::attachToContext(nvinfer1::IPluginResourceContext*) noexcept { return clone(); }
const nvinfer1::PluginFieldCollection* AcmeSwishPlugin::getFieldsToSerialize() noexcept {
    static nvinfer1::PluginFieldCollection empty{0, nullptr};
    return &empty;
}

const char* AcmeSwishPluginCreator::getPluginName() const noexcept { return kName; }
const char* AcmeSwishPluginCreator::getPluginVersion() const noexcept { return kVersion; }
const char* AcmeSwishPluginCreator::getPluginNamespace() const noexcept { return kNamespace; }
const nvinfer1::PluginFieldCollection* AcmeSwishPluginCreator::getFieldNames() noexcept {
    static nvinfer1::PluginFieldCollection empty{0, nullptr};
    return &empty;
}
nvinfer1::IPluginV3* AcmeSwishPluginCreator::createPlugin(const char*,
    const nvinfer1::PluginFieldCollection*, nvinfer1::TensorRTPhase) noexcept {
    return new (std::nothrow) AcmeSwishPlugin();
}

namespace {
AcmeSwishPluginCreator plugin_creator;
struct PluginRegistrar {
    PluginRegistrar() {
        auto* registry = getPluginRegistry();
        registry->registerCreator(plugin_creator, "");
        registry->registerCreator(plugin_creator, "com.acme");
    }
};
PluginRegistrar plugin_registrar;
}  // namespace
}  // namespace lesson26

extern "C" bool initAcmeSwishPlugin() { return true; }
