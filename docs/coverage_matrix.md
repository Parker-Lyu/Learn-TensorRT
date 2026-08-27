# Production Deployment Coverage Matrix

This document is not a second roadmap.

Its only purpose is to answer one question: if a job description asks for a production deployment skill, which lesson in this repository covers it?

Use it as a checklist when reading job descriptions or preparing interview talking points. Use `docs/learning_roadmap.md` as the actual learning plan.

## Baseline

All core-path coverage targets the single `nvcr.io/nvidia/pytorch:25.11-py3` development image,
TensorRT 10.14, CUDA Toolkit 13.0, and ISO C++17. ModelOpt workflows run in that same environment.

| Requirement | Coverage | Repository targets |
| --- | --- | --- |
| ISO C++17 engineering and memory safety | Covered as core path | `01_hello_world`, `03_opencv_preprocess`, `08_tensorrt_raii_resource`, `16_cpp_producer_consumer` |
| TensorRT 10.14 C++ inference lifecycle | Covered as core path | `06_trtexec_engine`, `09_tensorrt_cpp_basic`, `11_yolov8_trt_cpp` |
| Precision alignment and backend output debugging | Covered as core deployment-debug path | `05_torch_to_onnx`, `07_polygraphy_precision_alignment`, `14_yolov8_int8_quantization_engineering`, `15_precision_performance_report` |
| Pinned memory, CUDA streams, async copies, overlap | Covered as core path | `04_cuda_memory_stream`, `13_nsight_performance_diagnosis`, `18_async_video_pipeline`, `21_integrated_tensorrt_video_pipeline` |
| Zero-copy-style memory paths and Unified Memory trade-offs | Covered as advanced transfer topic | `04_cuda_memory_stream`, `20_cuda_preprocess_npp` |
| Explicit-Q/DQ PTQ and INT8 accuracy regression | Covered as core path | `14_yolov8_int8_quantization_engineering`, `15_precision_performance_report` |
| Dataset-level detection metrics and regression gates | Covered as core validation path | `14_yolov8_int8_quantization_engineering`, `15_precision_performance_report` |
| Mixed-precision layer audit and fallback diagnosis | Covered as advanced quantization topic | `14_yolov8_int8_quantization_engineering` |
| Unsupported operators, ONNX graph surgery, plugins | Covered as advanced path; the same `AcmeSwish` model is solved by graph surgery and then a plugin | `25_onnx_graph_surgery_plugin`, `26_custom_tensorrt_plugin` |
| TensorRT custom plugin implementation | Covered as portfolio differentiator | `26_custom_tensorrt_plugin` |
| TensorRT 10.14 plugin API, lifecycle, and validation | Covered as advanced path | `26_custom_tensorrt_plugin` |
| Performance diagnosis with `trtexec` and Nsight Systems | Covered as advanced path | `06_trtexec_engine`, `13_nsight_performance_diagnosis` |
| CUDA kernel diagnosis and optimization with Nsight Compute | Covered as advanced path; controlled fusion is measured separately from rejected candidates and deployment-value decisions | `20_cuda_preprocess_npp`, `31_nsight_compute_kernel_analysis` |
| CUDA or NPP preprocessing optimization | Covered as advanced path | `20_cuda_preprocess_npp` |
| Single-stream video pipeline | Integrated TensorRT implementation covered | `16_cpp_producer_consumer`, `18_async_video_pipeline`, `21_integrated_tensorrt_video_pipeline` |
| Multi-stream video pipeline | Integrated dynamic-batch TensorRT implementation covered | `17_dynamic_batching`, `19_multistream_video_pipeline`, `21_integrated_tensorrt_video_pipeline` |
| Pipeline latency, overload, and stability evidence | Covered; formal gate status is regenerated for each environment | `22_pipeline_performance_report` |
| Failure injection, sanitizer checks, and soak testing | Covered; see the environment-specific lesson 22 report gates | `08_tensorrt_raii_resource`, `16_cpp_producer_consumer`, `18_async_video_pipeline`, `19_multistream_video_pipeline`, `22_pipeline_performance_report` |
| Triton model serving, dynamic batching, and metrics | Implementation scaffold; runtime comparison report pending | `24_triton_inference_server` |
| DeepStream and GStreamer | Implementation scaffold; two-stream runtime acceptance pending | `27_deepstream_gstreamer_multistream` |
| Jetson Orin/Xavier, cross compilation, and DLA | Target-native scaffold; cross-compilation path and target acceptance pending | `28_jetson_orin_xavier_dla_deployment`, `27_deepstream_gstreamer_multistream` |
| C++ shared library and Python binding | Covered as advanced path | `29_cpp_shared_library_python_binding` |
| Production Docker runtime packaging | Covered as delivery path | `32_final_portfolio_case_study` |
| LLM inference entry point and controlled benchmark matrix | Teaching implementation covered; benchmark evidence pending | `30_llm_inference_intro` |
| Incremental portfolio evidence | Covered across core checkpoints | `12_end_to_end_validation_report`, `15_precision_performance_report`, `22_pipeline_performance_report`, `32_final_portfolio_case_study` |

## Notes

- The core path is enough for TensorRT deployment engineer interviews.
- Core-path artifacts and measurements are accepted only when reproduced on the TensorRT 10.14,
  CUDA 13.0, ISO C++17 baseline.
- The advanced path is aimed at senior industrial vision, medical imaging, edge AI, and high-end manufacturing deployment roles.
- Polygraphy belongs in the core path because precision alignment is part of real deployment debugging.
- DeepStream, Jetson/DLA, and TensorRT plugins are valuable, but they should come after the C++ YOLO TensorRT pipeline is already stable.
- Electives are selected from target job descriptions; the roadmap does not require completing every
  elective in numerical order.
