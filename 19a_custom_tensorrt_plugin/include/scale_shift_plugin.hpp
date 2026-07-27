#pragma once

#include <NvInfer.h>

#include <string>
#include <vector>

namespace lesson19a {

class ScaleShiftPlugin final : public nvinfer1::IPluginV3,
                               public nvinfer1::IPluginV3OneCore,
                               public nvinfer1::IPluginV3OneBuild,
                               public nvinfer1::IPluginV3OneRuntime {
public:
    ScaleShiftPlugin(float scale, float shift);

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override;
    ScaleShiftPlugin* clone() noexcept override;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;

    int32_t configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int32_t input_count,
                            const nvinfer1::DynamicPluginTensorDesc* outputs,
                            int32_t output_count) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* output_types, int32_t output_count,
                               const nvinfer1::DataType* input_types,
                               int32_t input_count) const noexcept override;
    int32_t getOutputShapes(const nvinfer1::DimsExprs* inputs, int32_t input_count,
                            const nvinfer1::DimsExprs* shape_inputs, int32_t shape_input_count,
                            nvinfer1::DimsExprs* outputs, int32_t output_count,
                            nvinfer1::IExprBuilder& expression_builder) noexcept override;
    bool supportsFormatCombination(int32_t position,
        const nvinfer1::DynamicPluginTensorDesc* descriptors,
        int32_t input_count, int32_t output_count) noexcept override;
    int32_t getNbOutputs() const noexcept override;
    std::size_t getWorkspaceSize(const nvinfer1::DynamicPluginTensorDesc* inputs,
        int32_t input_count, const nvinfer1::DynamicPluginTensorDesc* outputs,
        int32_t output_count) const noexcept override;

    int32_t onShapeChange(const nvinfer1::PluginTensorDesc* inputs, int32_t input_count,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t output_count) noexcept override;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                    const nvinfer1::PluginTensorDesc* output_desc,
                    const void* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext* context) noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override;

private:
    void initialize_serialization_fields();

    float scale_{1.0F};
    float shift_{0.0F};
    std::vector<nvinfer1::PluginField> serialization_fields_;
    nvinfer1::PluginFieldCollection serialization_collection_{};
};

class ScaleShiftPluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    ScaleShiftPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(const char* name,
        const nvinfer1::PluginFieldCollection* fields,
        nvinfer1::TensorRTPhase phase) noexcept override;

private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection collection_{};
};

}  // namespace lesson19a

extern "C" bool initScaleShiftPlugin();
