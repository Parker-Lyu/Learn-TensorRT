#include "pipeline_core.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace lesson21 {

void Accounting::validate_terminal(bool normal_drain) const {
    if (captured != admitted + rejected_on_admission)
        throw std::logic_error("captured must equal admitted plus admission rejections");
    if (admitted != evicted + submitted + discarded_on_abort)
        throw std::logic_error("admitted terminal accounting is inconsistent");
    if (submitted != completed + failed_in_flight)
        throw std::logic_error("submitted terminal accounting is inconsistent");
    if (normal_drain && (discarded_on_abort != 0 || failed_in_flight != 0))
        throw std::logic_error("normal drain cannot contain aborted or failed work");
}

SlotPool::SlotPool(std::size_t slot_count) : slots_(slot_count) {
    if (slot_count == 0) throw std::invalid_argument("slot count must be positive");
}

std::size_t SlotPool::reserve() {
    std::unique_lock<std::mutex> lock(mutex_);
    available_.wait(lock, [this] {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& s) {
            return s.state == SlotState::Free;
        });
    });
    const auto it = std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) {
        return s.state == SlotState::Free;
    });
    it->state = SlotState::Reserved;
    return static_cast<std::size_t>(std::distance(slots_.begin(), it));
}

void SlotPool::require(std::size_t slot, SlotState expected) const {
    if (slot >= slots_.size()) throw std::out_of_range("slot index is out of range");
    if (slots_[slot].state != expected) throw std::logic_error("invalid slot state transition");
}

void SlotPool::mark_submitted(std::size_t slot, BatchMetadata batch) {
    if (batch.frames.empty()) throw std::invalid_argument("cannot submit an empty batch");
    std::lock_guard<std::mutex> lock(mutex_);
    require(slot, SlotState::Reserved);
    slots_[slot].batch = std::move(batch);
    slots_[slot].state = SlotState::Submitted;
}

BatchMetadata SlotPool::begin_collection(std::size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    require(slot, SlotState::Submitted);
    slots_[slot].state = SlotState::Completing;
    return *slots_[slot].batch;
}

void SlotPool::release(std::size_t slot) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        require(slot, SlotState::Completing);
        slots_[slot].batch.reset();
        slots_[slot].state = SlotState::Free;
    }
    available_.notify_one();
}

void SlotPool::fail(std::size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slots_.size() || slots_[slot].state == SlotState::Free)
        throw std::logic_error("only owned slots can fail");
    slots_[slot].state = SlotState::Failed;
}

SlotState SlotPool::state(std::size_t slot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slots_.size()) throw std::out_of_range("slot index is out of range");
    return slots_[slot].state;
}
std::size_t SlotPool::size() const noexcept { return slots_.size(); }

void IdentityDispatcher::dispatch(const BatchMetadata& batch) {
    for (std::size_t index = 0; index < batch.frames.size(); ++index) {
        if (batch.frames[index].batch_index != index)
            throw std::invalid_argument("batch index does not match the immutable batch layout");
        results_.push_back(batch.frames[index]);
    }
}

}  // namespace lesson21
