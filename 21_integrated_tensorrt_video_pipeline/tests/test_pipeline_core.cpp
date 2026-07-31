#include "pipeline_core.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)

int main() {
    try {
        lesson21::SlotPool pool(2);
        const auto a = pool.reserve();
        const auto b = pool.reserve();
        CHECK(a != b);
        pool.mark_submitted(a, {10, {{7, 100, 0, lesson21::Clock::now(), {0.5F, 1, 2, 10, 20}}}});
        pool.mark_submitted(b, {11, {{8, 200, 0, lesson21::Clock::now(), {}}}});
        lesson21::IdentityDispatcher dispatch;
        dispatch.dispatch(pool.begin_collection(b)); pool.release(b);
        dispatch.dispatch(pool.begin_collection(a)); pool.release(a);
        CHECK(dispatch.results().size() == 2);
        CHECK(dispatch.results()[0].stream_id == 8 && dispatch.results()[0].frame_id == 200);
        CHECK(dispatch.results()[1].stream_id == 7 && dispatch.results()[1].frame_id == 100);
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
