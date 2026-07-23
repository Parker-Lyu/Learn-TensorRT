# Production Deployment Coverage Matrix

This document is not a second roadmap.

Its only purpose is to answer one question: if a job description asks for a production deployment skill, which lesson in this repository covers it?

Use it as a checklist when reading job descriptions or preparing interview talking points. Use `docs/learning_roadmap.md` as the actual learning plan.

## Baseline

All core-path coverage targets the single `nvcr.io/nvidia/pytorch:25.11-py3` development image,
TensorRT 10.14, CUDA Toolkit 13.0, and ISO C++17. ModelOpt workflows run in that same environment.

| Requirement | Coverage | Repository targets |
| --- | --- | --- |
| ISO C++17 engineering and memory safety | Covered as core path | `01_hello_world`, `03_opencv_preprocess`, `07_tensorrt_raii_resource`, `23_cpp_interview_katas` |
| TensorRT 10.14 C++ inference lifecycle | Covered as core path | `06_trtexec_engine`, `08_tensorrt_cpp_basic`, `10_yolov8_trt_cpp` |
| Precision alignment and backend output debugging | Covered as core deployment-debug path | `05_torch_to_onnx`, `06a_polygraphy_precision_alignment`, `12_yolov8_int8_quantization_engineering`, `12a_precision_performance_report` |
| Pinned memory, CUDA streams, async copies, overlap | Covered as core path | `04_cuda_memory_stream`, `11_nsight_performance_diagnosis`, `15_async_video_pipeline` |
| Zero-copy-style memory paths and Unified Memory trade-offs | Covered as advanced transfer topic | `04_cuda_memory_stream`, `17_cuda_preprocess_npp` |
| Explicit-Q/DQ PTQ and INT8 accuracy regression | Covered as core path | `12_yolov8_int8_quantization_engineering`, `12a_precision_performance_report` |
| Dataset-level detection metrics and regression gates | Covered as core validation path | `12_yolov8_int8_quantization_engineering`, `12a_precision_performance_report` |
| Mixed precision fallback and sensitive layer rollback | Covered as advanced quantization topic | `12_yolov8_int8_quantization_engineering` |
| Unsupported operators, ONNX graph surgery, plugins | Covered as advanced path | `19_onnx_graph_surgery_plugin`, `19a_custom_tensorrt_plugin` |
| TensorRT custom plugin implementation | Covered as portfolio differentiator | `19a_custom_tensorrt_plugin` |
| TensorRT 10.14 plugin API, lifecycle, and validation | Covered as advanced path | `19_onnx_graph_surgery_plugin`, `19a_custom_tensorrt_plugin` |
| Performance diagnosis with `trtexec` and Nsight Systems | Covered as advanced path | `06_trtexec_engine`, `11_nsight_performance_diagnosis` |
| CUDA or NPP preprocessing optimization | Covered as advanced path | `17_cuda_preprocess_npp` |
| Single-stream video pipeline | Covered as core path | `13_cpp_producer_consumer`, `15_async_video_pipeline` |
| Multi-stream video pipeline | Covered as core/advanced bridge | `14_dynamic_batching`, `16_multistream_video_pipeline` |
| Pipeline latency, overload, and stability evidence | Covered as core reporting path | `17a_pipeline_performance_report` |
| Failure injection, sanitizer checks, and soak testing | Covered across the core reliability path | `07_tensorrt_raii_resource`, `13_cpp_producer_consumer`, `15_async_video_pipeline`, `16_multistream_video_pipeline`, `17a_pipeline_performance_report` |
| Triton model serving, dynamic batching, and metrics | Covered as server inference elective | `18a_triton_inference_server` |
| DeepStream and GStreamer | Covered as advanced path | `20_deepstream_gstreamer_multistream` |
| Jetson Orin/Xavier, cross compilation, and DLA | Covered as edge deployment extension | `20a_jetson_orin_xavier_dla_deployment`, `20_deepstream_gstreamer_multistream` |
| C++ shared library and Python binding | Covered as advanced path | `21_cpp_shared_library_python_binding` |
| Production Docker runtime packaging | Covered as delivery path | `24_final_portfolio_case_study` |
| C++ hand-written deployment interview code | Covered as advanced path | `23_cpp_interview_katas` |
| LLM inference entry point and controlled benchmark matrix | Covered as extension | `22_llm_inference_intro` |
| Incremental portfolio evidence | Covered across core checkpoints | `10a_end_to_end_validation_report`, `12a_precision_performance_report`, `17a_pipeline_performance_report`, `24_final_portfolio_case_study` |

## Notes

- The core path is enough for TensorRT deployment engineer interviews.
- Core-path artifacts and measurements are accepted only when reproduced on the TensorRT 10.14,
  CUDA 13.0, ISO C++17 baseline.
- The advanced path is aimed at senior industrial vision, medical imaging, edge AI, and high-end manufacturing deployment roles.
- Polygraphy belongs in the core path because precision alignment is part of real deployment debugging.
- DeepStream, Jetson/DLA, and TensorRT plugins are valuable, but they should come after the C++ YOLO TensorRT pipeline is already stable.
- Electives are selected from target job descriptions; the roadmap does not require completing every
  elective in numerical order.
