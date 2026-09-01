# 29 - C++ Shared Library and Python Binding

## Purpose

This lesson packages the real lesson 17 TensorRT dynamic-batch runner as `libtrt_inference.so` and
exposes a narrow C ABI consumed by Python `ctypes`.

## Prerequisites

- Generate lesson 05's dynamic ONNX model and build lesson 17's dynamic TensorRT engine.
- Use the pinned development container with an accessible NVIDIA GPU.

## Deliverables

- `libtrt_inference.so` with a documented C ABI
- `python/trt_ctypes.py` Python client
- ABI, error-boundary, ownership, and integration tests

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 29_cpp_shared_library_python_binding -B 29_cpp_shared_library_python_binding/build
cmake --build 29_cpp_shared_library_python_binding/build --parallel
```

The generated build directory is ignored.

## Run

Export and validate the dynamic ONNX model, build its TensorRT engine, then call the C ABI from Python:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
./17_dynamic_batching/build_dynamic_engine.sh
python3 29_cpp_shared_library_python_binding/python/trt_ctypes.py --batch 2
```

Example output (local run):

```text
batch=2 output_elements=1411200 compute_ms=30.512 checksum=13901846.764
```

The ABI uses opaque session ownership, plain pointers, sizes, error codes, and one result struct.
C++ exceptions never cross the language boundary; they become a nonzero code plus thread-local
`trt_last_error()`. Python owns input memory until the synchronous call returns, while C++ owns the
TensorRT runtime, engine, context, stream, and device buffers through RAII.

`pybind11` can provide more idiomatic Python classes and automatic NumPy conversion, but it couples
the binary to Python/pybind11 ABI details. A C ABI is useful for stable integration with Python,
Rust, Go, or another service language; a later product can layer pybind11 on top without changing
the inference core.

The default engine is lesson 17's generated batch-1-to-4 plan. Rebuild it on the deployment
TensorRT/CUDA/GPU environment; do not copy a serialized plan between platforms. One `TrtSession`
owns one execution context and is not a concurrent request queue. Give each worker its own session
or serialize access at the caller.

## Outputs

- `build/libtrt_inference.so` and other native files are ignored build artifacts.
- The Python client prints structured inference results or an explicit C-ABI error.

## Tests

Run the Python tests from the repository root:

```bash
python3 -m unittest discover -s 29_cpp_shared_library_python_binding/tests -v
```

## Checkpoints

1. Expose the lesson 17 TensorRT runner through a narrow, stable C ABI.
2. Manage opaque session ownership, input/output memory, error codes, and exception boundaries safely.
3. Call the shared library from Python `ctypes` and validate structured results.
