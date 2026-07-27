#include "katas.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace lesson23 {
namespace {

std::size_t checked_image_element_count(const Image& image) {
    if (image.width <= 0 || image.height <= 0 || image.channels <= 0) {
        throw std::invalid_argument("image dimensions and channel count must be positive");
    }

    const auto width = static_cast<std::size_t>(image.width);
    const auto height = static_cast<std::size_t>(image.height);
    const auto channels = static_cast<std::size_t>(image.channels);
    if (width > std::numeric_limits<std::size_t>::max() / height ||
        width * height > std::numeric_limits<std::size_t>::max() / channels) {
        throw std::invalid_argument("image dimensions overflow the addressable element count");
    }
    return width * height * channels;
}

void validate_image(const Image& image) {
    if (image.data_hwc.size() != checked_image_element_count(image)) {
        throw std::invalid_argument("HWC data size does not match the image dimensions");
    }
}

}  // namespace

float iou(const Box& a, const Box& b) {
    const float intersection_width =
        std::max(0.0F, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float intersection_height =
        std::max(0.0F, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float intersection = intersection_width * intersection_height;
    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float denominator = area_a + area_b - intersection;
    return denominator > 0.0F ? intersection / denominator : 0.0F;
}

std::vector<Box> nms(std::vector<Box> boxes, float threshold) {
    if (threshold < 0.0F || threshold > 1.0F) {
        throw std::invalid_argument("NMS threshold must be in [0, 1]");
    }

    std::stable_sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.score > b.score;
    });

    std::vector<Box> kept;
    kept.reserve(boxes.size());
    for (const auto& candidate : boxes) {
        const bool suppressed =
            std::any_of(kept.begin(), kept.end(), [&](const Box& selected) {
                return selected.class_id == candidate.class_id &&
                       iou(selected, candidate) > threshold;
            });
        if (!suppressed) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

std::vector<std::size_t> top_k_indices(const std::vector<float>& scores, std::size_t k) {
    k = std::min(k, scores.size());
    std::vector<std::size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k), indices.end(),
        [&](std::size_t a, std::size_t b) {
            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
        });
    indices.resize(k);
    return indices;
}

float bilinear_sample(const Image& image, float x, float y, int channel) {
    validate_image(image);
    if (channel < 0 || channel >= image.channels) {
        throw std::invalid_argument("sample channel is outside the image channel range");
    }

    x = std::clamp(x, 0.0F, static_cast<float>(image.width - 1));
    y = std::clamp(y, 0.0F, static_cast<float>(image.height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const auto at = [&](int px, int py) {
        const auto pixel = static_cast<std::size_t>(py) * static_cast<std::size_t>(image.width) +
                           static_cast<std::size_t>(px);
        const auto index = pixel * static_cast<std::size_t>(image.channels) +
                           static_cast<std::size_t>(channel);
        return image.data_hwc[index];
    };
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);
    return (1.0F - dy) * ((1.0F - dx) * at(x0, y0) + dx * at(x1, y0)) +
           dy * ((1.0F - dx) * at(x0, y1) + dx * at(x1, y1));
}

std::vector<float> hwc_to_chw(const Image& image) {
    validate_image(image);
    const auto width = static_cast<std::size_t>(image.width);
    const auto height = static_cast<std::size_t>(image.height);
    const auto channels = static_cast<std::size_t>(image.channels);
    const std::size_t plane = width * height;
    std::vector<float> output(image.data_hwc.size());
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            for (std::size_t channel = 0; channel < channels; ++channel) {
                output[channel * plane + y * width + x] =
                    image.data_hwc[(y * width + x) * channels + channel];
            }
        }
    }
    return output;
}

Box map_from_letterbox(const Box& box, const Letterbox& transform) {
    if (transform.scale <= 0.0F || transform.original_width <= 0 ||
        transform.original_height <= 0) {
        throw std::invalid_argument("invalid letterbox transform");
    }

    Box output = box;
    output.x1 = std::clamp((box.x1 - transform.pad_x) / transform.scale, 0.0F,
                           static_cast<float>(transform.original_width));
    output.x2 = std::clamp((box.x2 - transform.pad_x) / transform.scale, 0.0F,
                           static_cast<float>(transform.original_width));
    output.y1 = std::clamp((box.y1 - transform.pad_y) / transform.scale, 0.0F,
                           static_cast<float>(transform.original_height));
    output.y2 = std::clamp((box.y2 - transform.pad_y) / transform.scale, 0.0F,
                           static_cast<float>(transform.original_height));
    return output;
}

}  // namespace lesson23
