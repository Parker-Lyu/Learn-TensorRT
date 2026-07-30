#pragma once

#include <stddef.h>

#if defined(_WIN32)
#define TRT_API __declspec(dllexport)
#else
#define TRT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtSession TrtSession;

typedef struct TrtInferenceResult {
    size_t batch_size;
    size_t output_elements;
    float h2d_ms;
    float compute_ms;
    float d2h_ms;
    double output_checksum;
} TrtInferenceResult;

TRT_API int trt_session_create(const char* engine_path, TrtSession** session);
TRT_API void trt_session_destroy(TrtSession* session);
TRT_API size_t trt_input_elements(size_t batch_size);
TRT_API int trt_session_infer(TrtSession* session, const float* input, size_t input_elements,
                              size_t batch_size, TrtInferenceResult* result);
TRT_API const char* trt_last_error(void);

#ifdef __cplusplus
}
#endif
