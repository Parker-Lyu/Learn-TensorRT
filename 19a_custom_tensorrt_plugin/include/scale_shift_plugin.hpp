#pragma once

#include <NvInfer.h>

#include <string>
#include <vector>

namespace lesson19a {

class ScaleShiftPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    ScaleShiftPlugin(float scale, float shift);
    ScaleShiftPlugin(const void* data, std::size_t length);

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    std::size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    ScaleShiftPlugin* clone() const noexcept override;
    void setPluginNamespace(const char* plugin_namespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int index, const nvinfer1::DataType* input_types, int input_count) const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int output_index,
        const nvinfer1::DimsExprs* inputs, int input_count,
        nvinfer1::IExprBuilder& expression_builder) noexcept override;
    bool supportsFormatCombination(int position, const nvinfer1::PluginTensorDesc* descriptors,
        int input_count, int output_count) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                         const nvinfer1::DynamicPluginTensorDesc*, int) noexcept override;
    std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                 const nvinfer1::PluginTensorDesc*, int) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                const nvinfer1::PluginTensorDesc* output_desc,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override;

private:
    float scale_{1.0F};
    float shift_{0.0F};
    std::string namespace_;
};

class ScaleShiftPluginCreator final : public nvinfer1::IPluginCreator {
public:
    ScaleShiftPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fields) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        const char* name, const void* data, std::size_t length) noexcept override;
    void setPluginNamespace(const char* plugin_namespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection collection_{};
    std::string namespace_;
};

}  // namespace lesson19a

extern "C" bool initScaleShiftPlugin();
