#include "result_writer.hpp"

#include "postprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace lesson21 {

ResultWriter::ResultWriter(const std::filesystem::path& output_directory,
                           std::size_t maximum_detection_records)
    : output_directory_(output_directory),
      maximum_detection_records_(maximum_detection_records) {
    std::filesystem::create_directories(output_directory_);
    detections_.open(output_directory_ / "detections.jsonl");
    if (!detections_) throw std::runtime_error("cannot open detections output");
}

double ResultWriter::write(const GpuBatchResult& result,
                           const std::vector<cv::Mat>& source_images,
                           std::vector<double>& frame_latencies_ms) {
    const auto started = Clock::now();
    const std::size_t count = result.metadata.frames.size();
    if (count == 0 || source_images.size() != count || result.output.size() % count != 0) {
        throw std::invalid_argument("result output does not match batch metadata");
    }
    const std::size_t elements_per_image = result.output.size() / count;
    for (std::size_t index = 0; index < count; ++index) {
        const auto begin = result.output.begin() + static_cast<std::ptrdiff_t>(index * elements_per_image);
        std::vector<float> slice(begin, begin + static_cast<std::ptrdiff_t>(elements_per_image));
        const Transform& transform = result.metadata.frames[index].transform;
        lesson11::LetterboxInfo letterbox{
            transform.source_width, transform.source_height, 640, 640,
            static_cast<int>(std::round(transform.source_width * transform.scale)),
            static_cast<int>(std::round(transform.source_height * transform.scale)),
            static_cast<int>(transform.pad_x), static_cast<int>(transform.pad_y),
            0, 0, transform.scale};
        const auto detections = lesson11::decode_yolov8_output(
            slice, {1, result.output_shape.at(1), result.output_shape.at(2)}, letterbox, {});
        if (!annotation_written_) {
            cv::Mat annotated = source_images[index].clone();
            for (const auto& detection : detections) {
                cv::rectangle(annotated,
                              {static_cast<int>(detection.box.x1), static_cast<int>(detection.box.y1)},
                              {static_cast<int>(detection.box.x2), static_cast<int>(detection.box.y2)},
                              {0, 255, 0}, 2);
            }
            if (!cv::imwrite((output_directory_ / "annotated_0.jpg").string(), annotated)) {
                throw std::runtime_error("cannot save annotation");
            }
            annotation_written_ = true;
        }
        if (detection_records_written_ >= maximum_detection_records_) {
            frame_latencies_ms.push_back(std::chrono::duration<double, std::milli>(
                Clock::now() - result.metadata.frames[index].captured_at).count());
            continue;
        }
        detections_ << "{\"stream_id\":" << result.metadata.frames[index].stream_id
                    << ",\"frame_id\":" << result.metadata.frames[index].frame_id
                    << ",\"batch_id\":" << result.metadata.batch_id << ",\"detections\":[";
        for (std::size_t detection = 0; detection < detections.size(); ++detection) {
            if (detection != 0) detections_ << ',';
            const auto& value = detections[detection];
            detections_ << "{\"class_id\":" << value.class_id
                        << ",\"confidence\":" << value.confidence << ",\"box\":["
                        << value.box.x1 << ',' << value.box.y1 << ','
                        << value.box.x2 << ',' << value.box.y2 << "]}";
        }
        detections_ << "]}\n";
        ++detection_records_written_;
        frame_latencies_ms.push_back(std::chrono::duration<double, std::milli>(
            Clock::now() - result.metadata.frames[index].captured_at).count());
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

}  // namespace lesson21
