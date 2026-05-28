# 06 - trtexec Engine

Goal: build and benchmark TensorRT engines from ONNX before writing TensorRT C++ code.

Topics:

- `trtexec --onnx`
- FP32 engine
- FP16 engine
- Static shape
- Dynamic shape profile
- Workspace memory
- Layer profiling
- Engine serialization

Acceptance criteria:

- FP32 and FP16 `.engine` files are generated.
- A benchmark table records latency, throughput, and GPU memory.
- You can compare FP32 and FP16 results.
