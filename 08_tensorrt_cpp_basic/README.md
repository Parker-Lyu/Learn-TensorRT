# 08 - TensorRT C++ Basic

Goal: write the minimal TensorRT C++ runtime flow.

Topics:

- Logger
- Builder
- ONNX parser
- Runtime
- Engine
- Engine deserialization
- Execution context
- Tensor names and shapes
- CUDA buffers
- Inference enqueue

Acceptance criteria:

- A C++ program loads an engine and runs one inference.
- Builder, parser, engine, runtime, context, buffer, and stream lifetimes are clear.
