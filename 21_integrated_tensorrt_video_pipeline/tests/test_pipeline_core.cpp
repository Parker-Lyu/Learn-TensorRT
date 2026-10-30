#include "pipeline_core.hpp"

#include <chrono>
#include <future>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)

namespace {
void closed_drop_queue_preserves_drain_contents() {
    lesson21::BoundedQueue<int> queue(2, lesson21::OverloadPolicy::DropOldest);
    queue.push(1);
    queue.push(2);
    queue.close(false);
    CHECK(!queue.push(3));
    CHECK(queue.evicted() == 0);
    CHECK(queue.pop().value() == 1);
    CHECK(queue.pop().value() == 2);
    CHECK(!queue.pop());
}

void close_wakes_blocked_producer_and_consumer() {
    using namespace std::chrono_literals;
    lesson21::BoundedQueue<int> full(1, lesson21::OverloadPolicy::Block);
    full.push(1);
    auto producer = std::async(std::launch::async, [&] { return full.push(2); });
    const bool blocked = producer.wait_for(20ms) == std::future_status::timeout;
    full.close(true);
    CHECK(producer.wait_for(2s) == std::future_status::ready);
    CHECK(!producer.get());
    CHECK(blocked);
    CHECK(full.discarded() == 1);

    lesson21::BoundedQueue<int> empty(1, lesson21::OverloadPolicy::Block);
    auto consumer = std::async(std::launch::async, [&] { return empty.pop(); });
    empty.close(false);
    CHECK(consumer.wait_for(2s) == std::future_status::ready);
    CHECK(!consumer.get());
}

void abort_accounting_distinguishes_rejections_and_in_flight_failures() {
    // 12 captured = 2 admission rejections + 10 admitted;
    // 10 admitted = 1 evicted + 6 submitted + 3 unsubmitted abort discards.
    lesson21::Accounting accounting{12, 10, 2, 1, 6, 4, 2, 3};
    accounting.validate_terminal(false);
    bool rejected = false;
    try { accounting.validate_terminal(true); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
}
}  // namespace

int main() {
    try {
        closed_drop_queue_preserves_drain_contents();
        close_wakes_blocked_producer_and_consumer();
        abort_accounting_distinguishes_rejections_and_in_flight_failures();
        lesson21::SlotPool pool(2);
        const auto a = pool.reserve();
        const auto b = pool.reserve();
        CHECK(a != b);
        CHECK(!pool.try_reserve().has_value());
        CHECK(pool.available() == 0);
        pool.mark_submitted(a, {10, {{7, 100, 0, lesson21::Clock::now(), {0.5F, 1, 2, 10, 20}}}});
        pool.mark_submitted(b, {11, {{8, 200, 0, lesson21::Clock::now(), {}}}});
        lesson21::IdentityDispatcher dispatch;
        dispatch.dispatch(pool.begin_collection(b)); pool.release(b);
        dispatch.dispatch(pool.begin_collection(a)); pool.release(a);
        CHECK(dispatch.results().size() == 2);
        CHECK(dispatch.results()[0].stream_id == 8 && dispatch.results()[0].frame_id == 200);
        CHECK(dispatch.results()[1].stream_id == 7 && dispatch.results()[1].frame_id == 100);
        CHECK(pool.available() == 2);
        lesson21::SlotPool failed_pool(1);
        const auto failed_slot = failed_pool.reserve();
        failed_pool.fail(failed_slot);
        CHECK(failed_pool.state(failed_slot) == lesson21::SlotState::Failed);
        bool invalid_release = false;
        try { failed_pool.release(failed_slot); }
        catch (const std::logic_error&) { invalid_release = true; }
        CHECK(invalid_release);
        lesson21::BoundedQueue<int> queue(2, lesson21::OverloadPolicy::DropOldest);
        CHECK(queue.push(1)); CHECK(queue.push(2)); CHECK(queue.push(3));
        CHECK(queue.evicted() == 1 && queue.peak() == 2);
        CHECK(queue.pop().value() == 2); queue.close(false); CHECK(queue.pop().value() == 3);
        CHECK(!queue.pop().has_value() && !queue.push(4));
        lesson21::BoundedQueue<int> aborted(2, lesson21::OverloadPolicy::Block);
        aborted.push(1); aborted.push(2); aborted.close(true);
        CHECK(aborted.discarded() == 2 && !aborted.pop().has_value());
        lesson21::Accounting ok{10, 9, 1, 2, 7, 7, 0, 0};
        ok.validate_terminal(true);
        bool rejected = false;
        try { lesson21::Accounting{1, 1, 0, 0, 1, 0, 0, 0}.validate_terminal(true); }
        catch (const std::logic_error&) { rejected = true; }
        CHECK(rejected);
        std::cout << "pipeline core tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
