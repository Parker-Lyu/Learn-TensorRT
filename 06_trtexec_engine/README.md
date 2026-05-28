# 06 - trtexec Engine

Goal: build and benchmark TensorRT engines from ONNX before writing TensorRT C++ code.

Topics:

- FP32 engine
- FP16 engine
- Dynamic shape profile
- Layer profiling
- Engine serialization

Acceptance criteria:

- FP32 and FP16 `.engine` files are generated.
- A benchmark table records latency, throughput, and GPU memory.
