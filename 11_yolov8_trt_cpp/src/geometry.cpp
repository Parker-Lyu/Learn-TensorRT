#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lesson11 {

void validate_letterbox_info(const LetterboxInfo& info) {
    if (!std::isfinite(info.scale) || info.scale <= 0.0F ||
        info.original_width <= 0 || info.original_height <= 0 ||
        info.pad_left < 0 || info.pad_top < 0) {
        throw std::runtime_error("Invalid letterbox scale, padding, or original image size.");
    }
}

Box map_box_to_original(const Box& box_in_letterbox, const LetterboxInfo& info) {
    validate_letterbox_info(info);
    if (!std::isfinite(box_in_letterbox.x1) || !std::isfinite(box_in_letterbox.y1) ||
        !std::isfinite(box_in_letterbox.x2) || !std::isfinite(box_in_letterbox.y2)) {
        throw std::runtime_error("Box coordinates must be finite.");
    }
    // Promote before subtraction/division so finite float coordinates do not overflow.
    const auto map_x = [&](float x) {
        return static_cast<float>(std::clamp(
            (static_cast<double>(x) - info.pad_left) / info.scale,
            0.0, static_cast<double>(info.original_width)));
    };
    const auto map_y = [&](float y) {
        return static_cast<float>(std::clamp(
            (static_cast<double>(y) - info.pad_top) / info.scale,
            0.0, static_cast<double>(info.original_height)));
    };
    Box mapped{map_x(box_in_letterbox.x1), map_y(box_in_letterbox.y1),
               map_x(box_in_letterbox.x2), map_y(box_in_letterbox.y2)};
    return mapped;
}

}  // namespace lesson11
