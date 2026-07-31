#pragma once

#include <algorithm>
#include <chrono>
#include <utility>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace lesson21 {

using Clock = std::chrono::steady_clock;

enum class SlotState { Free, Reserved, Submitted, Completing, Failed };

struct Transform {
    float scale{1.0F};
    float pad_x{0.0F};
    float pad_y{0.0F};
    int source_width{0};
    int source_height{0};
};

struct FrameMetadata {
    std::size_t stream_id{0};
    std::uint64_t frame_id{0};
    std::size_t batch_index{0};
    Clock::time_point captured_at{};
    Transform transform{};
};

struct BatchMetadata {
    std::uint64_t batch_id{0};
    std::vector<FrameMetadata> frames;
};

struct Accounting {
    std::uint64_t captured{0};
    std::uint64_t admitted{0};
    std::uint64_t rejected_on_admission{0};
    std::uint64_t evicted{0};
    std::uint64_t submitted{0};
    std::uint64_t completed{0};
    std::uint64_t failed_in_flight{0};
    std::uint64_t discarded_on_abort{0};

    void validate_terminal(bool normal_drain) const;
};

class SlotPool {
public:
    explicit SlotPool(std::size_t slot_count);
    std::size_t reserve();
    void mark_submitted(std::size_t slot, BatchMetadata batch);
    BatchMetadata begin_collection(std::size_t slot);
    void release(std::size_t slot);
    void fail(std::size_t slot);
    SlotState state(std::size_t slot) const;
    std::size_t size() const noexcept;
private:
    struct Slot { SlotState state{SlotState::Free}; std::optional<BatchMetadata> batch; };
    void require(std::size_t slot, SlotState expected) const;
    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Slot> slots_;
};

// Dispatch deliberately accepts arbitrary completion order. Identity is copied from immutable batch
// metadata rather than inferred from the slot number or completion position.
class IdentityDispatcher {
public:
    void dispatch(const BatchMetadata& batch);
    const std::vector<FrameMetadata>& results() const noexcept { return results_; }
private:
    std::vector<FrameMetadata> results_;
};

}  // namespace lesson21

namespace lesson21 {
enum class OverloadPolicy { Block, DropOldest };
template <class T> class BoundedQueue {
public:
    BoundedQueue(std::size_t capacity, OverloadPolicy policy)
        : capacity_(capacity), policy_(policy) {
        if (capacity == 0) throw std::invalid_argument("queue capacity must be positive");
    }
    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (policy_ == OverloadPolicy::Block) {
            not_full_.wait(lock, [this] { return closed_ || values_.size() < capacity_; });
            if (closed_) return false;
        } else if (values_.size() == capacity_) {
            values_.pop_front(); ++evicted_;
        }
        if (closed_) return false;
        values_.push_back(std::move(value));
        peak_ = std::max(peak_, values_.size());
        not_empty_.notify_one();
        return true;
    }
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !values_.empty(); });
        if (values_.empty()) return std::nullopt;
        T value = std::move(values_.front()); values_.pop_front();
        not_full_.notify_one(); return value;
    }
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (values_.empty()) return std::nullopt;
        T value = std::move(values_.front()); values_.pop_front();
        not_full_.notify_one(); return value;
    }
    void close(bool discard) {
        std::lock_guard<std::mutex> lock(mutex_); closed_ = true;
        if (discard) { discarded_ += values_.size(); values_.clear(); }
        not_empty_.notify_all(); not_full_.notify_all();
    }
    bool empty() const { std::lock_guard<std::mutex> l(mutex_); return values_.empty(); }
    std::size_t evicted() const { std::lock_guard<std::mutex> l(mutex_); return evicted_; }
    std::size_t discarded() const { std::lock_guard<std::mutex> l(mutex_); return discarded_; }
    std::size_t peak() const { std::lock_guard<std::mutex> l(mutex_); return peak_; }
private:
    const std::size_t capacity_; const OverloadPolicy policy_;
    mutable std::mutex mutex_; std::condition_variable not_empty_, not_full_;
    std::deque<T> values_; bool closed_{false}; std::size_t evicted_{0}, discarded_{0}, peak_{0};
};
}  // namespace lesson21
