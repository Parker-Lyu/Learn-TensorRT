#include "visualize.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace lesson11 {

cv::Mat draw_detections(const cv::Mat& bgr_image, const std::vector<Detection>& detections) {
    if (bgr_image.empty()) {
        throw std::runtime_error("draw_detections received an empty image.");
    }

    cv::Mat output = bgr_image.clone();
    for (const Detection& detection : detections) {
        const cv::Scalar color((37 * (detection.class_id + 3)) % 255,
                              (17 * (detection.class_id + 7)) % 255,
                              (29 * (detection.class_id + 11)) % 255);
        const int x1 = static_cast<int>(std::round(detection.box.x1));
        const int y1 = static_cast<int>(std::round(detection.box.y1));
        const int x2 = static_cast<int>(std::round(detection.box.x2));
        const int y2 = static_cast<int>(std::round(detection.box.y2));
        cv::rectangle(output, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        const std::string label =
            detection.class_name + " " + std::to_string(detection.confidence).substr(0, 4);
        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int label_y = std::max(0, y1 - text_size.height - baseline - 4);
        cv::rectangle(output, cv::Rect(x1, label_y, text_size.width + 4,
                                       text_size.height + baseline + 4),
                      color, cv::FILLED);
        cv::putText(output, label, cv::Point(x1 + 2, label_y + text_size.height + 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    }
    return output;
}

}  // namespace lesson11
