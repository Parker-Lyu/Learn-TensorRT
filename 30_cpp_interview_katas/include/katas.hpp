#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace lesson30 {

struct Box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

float iou(const Box& left, const Box& right);
std::vector<Box> nms(std::vector<Box> boxes, float threshold);
std::vector<std::size_t> top_k_indices(const std::vector<float>& scores, std::size_t k);

struct Image {
    int width;
    int height;
    int channels;
    std::vector<float> data_hwc;
};

float bilinear_sample(const Image& image, float x, float y, int channel);
std::vector<float> hwc_to_chw(const Image& image);

struct Letterbox {
    float scale;
    float pad_x;
    float pad_y;
    int original_width;
    int original_height;
};

Box map_from_letterbox(const Box& box, const Letterbox& transform);

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t capacity) : values_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("ring capacity must be positive");
        }
    }

    bool push(T value) {
        if (size_ == values_.size()) {
            return false;
        }
        values_[(head_ + size_) % values_.size()] = std::move(value);
        ++size_;
        return true;
    }

    std::optional<T> pop() {
        if (size_ == 0) {
            return std::nullopt;
        }
        T value = std::move(values_[head_]);
        head_ = (head_ + 1) % values_.size();
        --size_;
        return value;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::vector<T> values_;
    std::size_t head_{0};
    std::size_t size_{0};
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("queue capacity must be positive");
        }
    }

    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        writable_.wait(lock, [&] { return closed_ || values_.size() < capacity_; });
        if (closed_) {
            return false;
        }
        values_.push_back(std::move(value));
        readable_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        readable_.wait(lock, [&] { return closed_ || !values_.empty(); });
        if (values_.empty()) {
            return std::nullopt;
        }
        T value = std::move(values_.front());
        values_.pop_front();
        writable_.notify_one();
        return value;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        readable_.notify_all();
        writable_.notify_all();
    }

private:
    const std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable readable_;
    std::condition_variable writable_;
    std::deque<T> values_;
    bool closed_{false};
};

}  // namespace lesson30
