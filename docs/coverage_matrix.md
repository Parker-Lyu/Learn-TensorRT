# Production Deployment Coverage Matrix

This document is not a second roadmap.

Its only purpose is to answer one question: if a job description asks for a production deployment skill, which lesson in this repository covers it?

Use it as a checklist when reading job descriptions or preparing interview talking points. Use `docs/learning_roadmap.md` as the actual learning plan.

| Requirement | Coverage | Repository targets |
| --- | --- | --- |
| Modern C++ engineering and memory safety | Covered as core path | `01_hello_world`, `03_opencv_preprocess`, `07_tensorrt_raii_resource`, `23_cpp_interview_katas` |
| TensorRT C++ inference lifecycle | Covered as core path | `06_trtexec_engine`, `08_tensorrt_cpp_basic`, `10_yolov8_trt_cpp` |
| Pinned memory, CUDA streams, async copies, overlap | Covered as core path | `04_cuda_memory_stream`, `11_nsight_performance_diagnosis`, `15_async_video_pipeline` |
| Zero-copy-style memory paths and Unified Memory trade-offs | Covered as advanced transfer topic | `04_cuda_memory_stream`, `17_cuda_preprocess_npp` |
| PTQ, entropy calibration, INT8 accuracy regression | Covered as core path | `12_yolov8_int8_calibration`, `24_benchmark_report` |
| Mixed precision fallback and sensitive layer rollback | Covered as advanced quantization topic | `12_yolov8_int8_calibration` |
| Unsupported operators, ONNX graph surgery, plugins | Covered as advanced path | `19_onnx_graph_surgery_plugin` |
| Performance diagnosis with `trtexec` and Nsight Systems | Covered as advanced path | `06_trtexec_engine`, `11_nsight_performance_diagnosis` |
| CUDA or NPP preprocessing optimization | Covered as advanced path | `17_cuda_preprocess_npp` |
| Single-stream video pipeline | Covered as core path | `13_cpp_producer_consumer`, `15_async_video_pipeline` |
| Multi-stream video pipeline | Covered as core/advanced bridge | `14_dynamic_batching`, `16_multistream_video_pipeline` |
| DeepStream and GStreamer | Covered as advanced path | `20_deepstream_gstreamer_multistream` |
| C++ shared library and Python binding | Covered as advanced path | `21_cpp_shared_library_python_binding` |
| Production Docker runtime packaging | Covered as delivery path | `24_benchmark_report` |
| C++ hand-written deployment interview code | Covered as advanced path | `23_cpp_interview_katas` |
| LLM inference entry point | Covered as extension | `22_llm_inference_intro`, `21_cpp_shared_library_python_binding` |

## Notes

- The core path is enough for TensorRT deployment engineer interviews.
- The advanced path is aimed at senior industrial vision, medical imaging, edge AI, and high-end manufacturing deployment roles.
- DeepStream and TensorRT plugins are valuable, but they should come after the C++ YOLO TensorRT pipeline is already stable.
