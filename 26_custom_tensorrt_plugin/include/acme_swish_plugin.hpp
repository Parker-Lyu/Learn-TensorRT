#pragma once

#include <NvInfer.h>

namespace lesson26 {

class AcmeSwishPlugin final : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
public:
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    AcmeSwishPlugin* clone() noexcept override;
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
        const nvinfer1::DynamicPluginTensorDesc*, int32_t) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType*, int32_t, const nvinfer1::DataType*, int32_t) const noexcept override;
    int32_t getOutputShapes(const nvinfer1::DimsExprs*, int32_t, const nvinfer1::DimsExprs*, int32_t,
        nvinfer1::DimsExprs*, int32_t, nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int32_t, const nvinfer1::DynamicPluginTensorDesc*, int32_t, int32_t) noexcept override;
    int32_t getNbOutputs() const noexcept override;
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc*, int32_t,
        const nvinfer1::DynamicPluginTensorDesc*, int32_t) const noexcept override;
    int32_t onShapeChange(const nvinfer1::PluginTensorDesc*, int32_t,
        const nvinfer1::PluginTensorDesc*, int32_t) noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc*, const nvinfer1::PluginTensorDesc*,
        const void* const*, void* const*, void*, cudaStream_t) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override;
};

class AcmeSwishPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(const char*, const nvinfer1::PluginFieldCollection*,
        nvinfer1::TensorRTPhase) noexcept override;
};

}  // namespace lesson26

extern "C" bool initAcmeSwishPlugin();
