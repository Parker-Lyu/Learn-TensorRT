#pragma once

#include "yolo_types.hpp"

namespace lesson11 {

// Validate the scale, original dimensions, and leading padding used by inverse mapping.
void validate_letterbox_info(const LetterboxInfo& info);
Box map_box_to_original(const Box& box_in_letterbox, const LetterboxInfo& info);

}  // namespace lesson11
