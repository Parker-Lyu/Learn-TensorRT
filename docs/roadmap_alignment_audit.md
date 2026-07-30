# Roadmap, README, and Implementation Alignment Audit

Audit date: 2026-07-28

## Scope And Method

This audit compares three layers of the repository:

1. The purpose, topics, deliverables, and acceptance criteria in
   [`learning_roadmap.md`](learning_roadmap.md).
2. The learner-facing claims and reproduction commands in every lesson `README.md`.
3. The tracked implementation, tests, and generated-report logic. Files under `reports/` are local,
   ignored evidence rather than versioned proof.

`Aligned` means the tracked lesson provides the runnable teaching content promised by the roadmap.
It does not mean that every GPU- or platform-dependent command was rerun during this documentation
audit. `Partial` identifies a concrete missing implementation, evidence gate, or learner deliverable.
`Runtime pending` means the implementation is present but the required platform run is not preserved
or has not passed.

## Course-By-Course Result

| Lesson | Status | Roadmap/README/implementation comparison |
| --- | --- | --- |
| `00_environment_check` | Partial | The script inventories the required stack, but it does not strictly reject the wrong CUDA Toolkit version or container identity, and it does not compile an ISO C++17 probe. It can therefore print `PASS` without proving the complete pinned baseline. |
| `01_hello_world` | Aligned | The CMake target builds and runs a minimal ISO C++17 executable with extensions disabled, matching the README and roadmap. |
| `02_opencv_read_image_info` | Aligned | The implementation loads the shared image, reports `cv::Mat` metadata, handles errors, and links OpenCV through target-based CMake. |
| `03_opencv_preprocess` | Aligned | Letterbox, color conversion, normalization, HWC-to-CHW, batch layout, reverse coordinate mapping, saved outputs, and focused defensive tests are implemented in reusable files. |
| `04_cuda_memory_stream` | Aligned | Pageable, pinned, mapped-pinned, and managed-memory paths, streams, events, validation, and the documented transfer trade-offs are present. |
| `05_torch_to_onnx` | Aligned | Static/dynamic simplified export, graph inspection, and same-input PyTorch-versus-ONNX Runtime comparison are implemented. Netron remains an appropriate manual inspection checkpoint. |
| `06_trtexec_engine` | Aligned | Strict FP32, FP16, dynamic profiles, timing cache, serialized engines, layer/timing exports, and environment manifests match the roadmap. |
| `06a_polygraphy_precision_alignment` | Partial | The lesson performs controlled final-output comparison and records drift, but it does not implement a genuine layerwise or tensorwise first-divergence workflow capable of isolating the first failing layer. |
| `07_tensorrt_raii_resource` | Aligned | TensorRT/CUDA RAII ownership, move-only wrappers, staged failure injection, explicit errors, repeated lifecycle runs, and host/device memory gates are implemented. The substantive lifecycle gate still requires a GPU run. |
| `08_tensorrt_cpp_basic` | Aligned | Strongly typed building, strict timing-cache reuse, TensorRT 10 name-based I/O, type/format validation, buffer binding, and `enqueueV3` are present. |
| `09_yolov8_trt_python` | Partial | The full TensorRT Python pipeline is present, but the optional Ultralytics reference records only summary information; it does not calculate a tolerance, comparison metric, or pass/fail result for the roadmap's “output is close” criterion. |
| `10_yolov8_trt_cpp` | Aligned | Modular preprocessing, runtime, postprocessing, visualization, reusable CUDA buffers, NVTX, CLI output, stage timing, and focused edge-case tests match the roadmap. |
| `10a_end_to_end_validation_report` | Partial | The generator enforces consistent artifacts and produces a strong controlled-input report. The reproduction sequence builds its ONNX and engine prerequisites, but the generated evidence and report are intentionally ignored, and the promised one-page summary and 3–5-minute walkthrough are only a short summary and speaking outline. |
| `11_nsight_performance_diagnosis` | Partial | Strict Nsight capture/export/statistics tooling and baseline diagnosis are implemented. No committed before-and-after timeline demonstrates the roadmap's required evidence-backed optimization. |
| `12_yolov8_int8_quantization_engineering` | Aligned | Immutable data identities, preprocessing parity, common evaluation, ModelOpt Q/DQ, quality gates, Engine Inspector audit, matched performance, and a committed TensorRT 10.14 decision are present. |
| `12a_precision_performance_report` | Partial | Identity-linked latency, drift, detection-quality, layer-audit, and decision evidence is strong. The required one-page English summary and actual 3–5-minute explanation are still only a short paragraph and prompt. |
| `13_cpp_producer_consumer` | Partial | Queue ownership, backpressure, shutdown, failure propagation, and stress behavior are implemented. The required ThreadSanitizer acceptance is not complete; the latest locally generated pipeline report records a TSAN startup failure but is not versioned evidence. |
| `14_dynamic_batching` | Aligned | One engine runs batches 1, 2, and 4 with runtime shapes, checked input/output offsets, and compute-latency/throughput evidence. The README correctly scopes its throughput to model compute. |
| `15_async_video_pipeline` | Materially partial | Video capture, bounded queues, timestamps, micro-batching, failures, and metrics are real, but inference is a CPU `std::async` simulation. There is no TensorRT/CUDA backend, CPU/GPU overlap, or program-generated GPU-utilization evidence. |
| `16_multistream_video_pipeline` | Materially partial | Per-stream queues, scheduling, identity, out-of-order completion, failures, and metrics are implemented. The worker only sleeps and returns frame identities; it does not batch TensorRT inference or route detection results. |
| `17_cuda_preprocess_npp` | Partial | NPP resize, fused CUDA conversion, CPU comparison, transfer modes, and separate CUDA timings are present. The final pinned/mapped host-output copy into the result vector is not timed, so the README/report currently undercount host-side output-copy cost. |
| `17a_pipeline_performance_report` | Incomplete | The latest locally generated report fails the 30-minute soak and TSAN gates, but reports are intentionally not versioned. The collector also lacks ASAN, GPU-utilization collection, shutdown-during-load injection, and memory sampling across soak/restart cycles; its pipeline timings use the simulated workers from lessons 15 and 16. |
| `18_openvino_yolov8` | Runtime pending | Synchronous and asynchronous OpenVINO CPU paths, percentiles, environment isolation, and TensorRT comparison generation are implemented. No durable runtime comparison is committed, and raw-output drift has no acceptance tolerance. |
| `18a_triton_inference_server` | Partial / runtime pending | The model repository, HTTP client, dynamic batching, and concurrency driver are present. There is only one batching configuration, output correctness is not compared with a reference, Prometheus/GPU evidence is not collected into a report, and the two documented concurrency runs overwrite the same default output file. |
| `19_onnx_graph_surgery_plugin` | Mostly aligned | The escalation strategy, runnable unsupported node, GraphSurgeon rewrite, and numerical validation are present. Constant folding and node splitting are roadmap labels rather than implemented exercises, and runtime evidence is not preserved. |
| `19a_custom_tensorrt_plugin` | Aligned; runtime evidence local | A real TensorRT 10.14 `IPluginV3` library implements fields, registration, serialization, CUDA `enqueue`, engine building, deserialization, and CPU-reference validation. Dynamic shape handling is not exercised by the fixed demo shape, and no durable run log is committed. |
| `20_deepstream_gstreamer_multistream` | Partial / runtime pending | Real DeepStream configuration generation, a YOLO parser, and runtime-asset build steps exist. No accepted two-stream DeepStream run, model-load evidence, per-stream FPS, or GPU monitoring is preserved, and the roadmap's tracker topic is not configured. |
| `20a_jetson_orin_xavier_dla_deployment` | Partial / runtime pending | Target-native GPU/DLA build, fallback analysis, target benchmark, and honest x86 boundaries are present. The roadmap's aarch64 cross-compilation checklist/toolchain path is absent, and Jetson power/clocks/runtime acceptance remains pending. |
| `21_cpp_shared_library_python_binding` | Mostly aligned | The real lesson 14 runner is exposed through a C ABI and Python `ctypes` with explicit ownership and exception boundaries. The result struct exposes timing, element count, and checksum rather than output tensors or decoded detections, limiting its usefulness to Python business logic. |
| `22_llm_inference_intro` | Partial / evidence pending | The deterministic teaching Transformer implements tokenization, causal attention, prefill/decode steps, KV-cache growth, and a 2-by-2 benchmark matrix. Generated reports are local and ignored; prefill is token-by-token rather than backend-style parallel prefill, and quantization/backend coverage remains conceptual. |
| `23_cpp_interview_katas` | Mostly aligned | Every named kata and the move-only CUDA wrapper are implemented and tested. The defensive matrix should still add true extreme-coordinate cases, zero-capacity queue validation, and blocked-consumer wakeup coverage. |
| `24_final_portfolio_case_study` | Partial | The report generator, local check matrix, evidence links, and measured multi-stage image exist. The precision decision now correctly retains FP16 because matched INT8 is slower, and the elective section no longer claims completion. Final acceptance remains incomplete while the 17a soak/TSAN gates and Triton/DeepStream/Jetson runtime validation are pending. |

## README Language Result

All lesson `README.md` files are now written in English. During this audit, the Chinese appendix in
lesson 06 and the Chinese lesson 12 README were translated. The scan checks lesson README prose; it
does not prohibit Chinese in private notes, Git history, or user-facing discussion outside the
learner documentation.

## Highest-Priority Alignment Work

1. Replace the simulated workers in lessons 15 and 16 with a reusable TensorRT backend while
   retaining a CPU-testable fake for deterministic unit tests.
2. Complete lesson 17a evidence: monitored 30-minute soak, monitored restart cycles, ASAN, working
   TSAN, CUDA memcheck, GPU utilization, and shutdown-during-load injection.
3. Add a real layerwise/tensorwise divergence path to lesson 06a and a quantitative reference gate
   to lesson 09.
4. Complete Triton comparison/reporting, DeepStream execution, and Jetson target evidence; add the
   missing Jetson cross-compilation path.
5. Expand checkpoint English summaries and walkthroughs into the deliverables promised by the
   roadmap rather than leaving speaker prompts.
6. Make lesson 21 return useful inference output or decoded detections across the C ABI.

Until those items are complete, the repository should describe them as implemented scaffolds,
partial lessons, or pending platform validation rather than completed production coverage.
