#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace lesson16 {

enum class OverloadPolicy {
    Block,
    DropNewest,
    DropOldest,
};

enum class CloseMode {
    Drain,
    Discard,
};

enum class PushResult {
    Pushed,
    DroppedNewest,
    DroppedOldest,
    Closed,
};

struct QueueStats {
    std::size_t size{0};
    std::size_t high_watermark{0};
    std::size_t pushed{0};
    std::size_t popped{0};
    std::size_t dropped{0};
    bool closed{false};
};

// A bounded multi-producer/multi-consumer queue with explicit overload and shutdown behavior.
// All state, including statistics, is guarded by the same mutex as the payload queue.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity, OverloadPolicy policy = OverloadPolicy::Block)
        : capacity_(capacity), policy_(policy) {
        if (capacity_ == 0) {
            throw std::invalid_argument("queue capacity must be greater than zero");
        }
    }

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    PushResult push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (policy_ == OverloadPolicy::Block) {
            not_full_.wait(lock, [this] { return closed_ || queue_.size() < capacity_; });
        }

        if (closed_) {
            return PushResult::Closed;
        }

        if (queue_.size() == capacity_) {
            if (policy_ == OverloadPolicy::DropNewest) {
                ++dropped_;
                return PushResult::DroppedNewest;
            }

            // DropOldest keeps the newest camera frame and bounds end-to-end latency.
            queue_.pop_front();
            ++dropped_;
            queue_.push_back(std::move(value));
            ++pushed_;
            lock.unlock();
            not_empty_.notify_one();
            return PushResult::DroppedOldest;
        }

        queue_.push_back(std::move(value));
        ++pushed_;
        if (queue_.size() > high_watermark_) {
            high_watermark_ = queue_.size();
        }
        lock.unlock();
        not_empty_.notify_one();
        return PushResult::Pushed;
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !queue_.empty(); });

        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        ++popped_;
        lock.unlock();
        not_full_.notify_one();
        return value;
    }

    void close(CloseMode mode = CloseMode::Drain) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) {
                return;
            }
            closed_ = true;
            if (mode == CloseMode::Discard) {
                dropped_ += queue_.size();
                queue_.clear();
            }
        }
        // Both sides may be blocked, so close must wake both condition variables.
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    QueueStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return QueueStats{queue_.size(), high_watermark_, pushed_, popped_, dropped_, closed_};
    }

private:
    const std::size_t capacity_;
    const OverloadPolicy policy_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    std::size_t high_watermark_{0};
    std::size_t pushed_{0};
    std::size_t popped_{0};
    std::size_t dropped_{0};
    bool closed_{false};
};

}  // namespace lesson16
