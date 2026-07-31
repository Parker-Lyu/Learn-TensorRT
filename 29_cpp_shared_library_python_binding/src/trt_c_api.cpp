#include "trt_c_api.h"

#include "batch_layout.hpp"
#include "dynamic_batch_runner.hpp"

#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct TrtSession {
    explicit TrtSession(const char* path) : runner(path) {}
    lesson17::DynamicBatchRunner runner;
};

namespace {
thread_local std::string last_error;
template <typename Function>
int boundary(Function&& function) noexcept {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    } catch (...) {
        last_error = "unknown C++ exception";
        return 2;
    }
}
}

extern "C" TRT_API int trt_session_create(const char* engine_path, TrtSession** session) {
    return boundary([&] {
        if (!engine_path || !session) throw std::invalid_argument("engine_path and session are required");
        *session = nullptr;
        *session = new TrtSession(engine_path);
    });
}
extern "C" TRT_API void trt_session_destroy(TrtSession* session) { delete session; }
extern "C" TRT_API size_t trt_input_elements(size_t batch_size) {
    constexpr size_t sample = 3ULL * 640ULL * 640ULL;
    return batch_size >= 1 && batch_size <= 4 ? batch_size * sample : 0;
}
extern "C" TRT_API int trt_session_infer(TrtSession* session, const float* input,
                                           size_t input_elements, size_t batch_size,
                                           TrtInferenceResult* result) {
    return boundary([&] {
        if (!session || !input || !result) throw std::invalid_argument("session, input, and result are required");
        const size_t expected = trt_input_elements(batch_size);
        if (!expected || input_elements != expected) throw std::invalid_argument("invalid batch or input element count");
        std::vector<float> values(input, input + input_elements);
        const auto timing = session->runner.infer(values, batch_size);
        *result = {timing.batch_size, lesson17::checked_volume(timing.output_shape),
                   timing.h2d_ms, timing.compute_ms, timing.d2h_ms, timing.output_checksum};
    });
}
extern "C" TRT_API const char* trt_last_error(void) { return last_error.c_str(); }
