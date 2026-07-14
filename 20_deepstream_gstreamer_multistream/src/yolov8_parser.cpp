#include <nvdsinfer_custom_impl.h>

#include <algorithm>
#include <cstddef>
#include <vector>

extern "C" bool NvDsInferParseYoloV8(
    const std::vector<NvDsInferLayerInfo>& output_layers,
    const NvDsInferNetworkInfo& network,
    const NvDsInferParseDetectionParams& detection,
    std::vector<NvDsInferObjectDetectionInfo>& objects) {
    if (output_layers.size() != 1 || output_layers[0].buffer == nullptr) return false;
    constexpr int attributes = 84;
    const std::size_t elements = output_layers[0].inferDims.numElements;
    if (elements % attributes != 0) return false;
    const std::size_t candidates = elements / attributes;
    const auto* output = static_cast<const float*>(output_layers[0].buffer);
    const std::size_t class_count = std::min<std::size_t>(80, detection.numClassesConfigured);

    for (std::size_t candidate = 0; candidate < candidates; ++candidate) {
        int best_class = -1;
        float best_score = 0.0F;
        for (std::size_t class_id = 0; class_id < class_count; ++class_id) {
            const float score = output[(4 + class_id) * candidates + candidate];
            const float threshold = detection.perClassPreclusterThreshold[class_id];
            if (score >= threshold && score > best_score) {
                best_score = score;
                best_class = static_cast<int>(class_id);
            }
        }
        if (best_class < 0) continue;
        const float center_x = output[candidate];
        const float center_y = output[candidates + candidate];
        const float width = output[2 * candidates + candidate];
        const float height = output[3 * candidates + candidate];
        const float left = std::clamp(center_x - width * 0.5F, 0.0F,
                                      static_cast<float>(network.width));
        const float top = std::clamp(center_y - height * 0.5F, 0.0F,
                                     static_cast<float>(network.height));
        const float right = std::clamp(center_x + width * 0.5F, 0.0F,
                                       static_cast<float>(network.width));
        const float bottom = std::clamp(center_y + height * 0.5F, 0.0F,
                                        static_cast<float>(network.height));
        if (right <= left || bottom <= top) continue;
        NvDsInferObjectDetectionInfo object{};
        object.classId = best_class;
        object.detectionConfidence = best_score;
        object.left = left;
        object.top = top;
        object.width = right - left;
        object.height = bottom - top;
        objects.push_back(object);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
