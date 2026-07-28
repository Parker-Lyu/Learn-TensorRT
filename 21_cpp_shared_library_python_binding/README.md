# 21 - C++ Shared Library and Python Binding

This lesson packages the real lesson 14 TensorRT dynamic-batch runner as `libtrt_inference.so` and
exposes a narrow C ABI consumed by Python `ctypes`.

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
./14_dynamic_batching/setup_autocast_deps.sh
./14_dynamic_batching/build_dynamic_engine.sh
cmake -S 21_cpp_shared_library_python_binding -B 21_cpp_shared_library_python_binding/build
cmake --build 21_cpp_shared_library_python_binding/build -j
python3 21_cpp_shared_library_python_binding/python/trt_ctypes.py --batch 2
python3 -m unittest discover -s 21_cpp_shared_library_python_binding/tests -v
```

The ABI uses opaque session ownership, plain pointers, sizes, error codes, and one result struct.
C++ exceptions never cross the language boundary; they become a nonzero code plus thread-local
`trt_last_error()`. Python owns input memory until the synchronous call returns, while C++ owns the
TensorRT runtime, engine, context, stream, and device buffers through RAII.

`pybind11` can provide more idiomatic Python classes and automatic NumPy conversion, but it couples
the binary to Python/pybind11 ABI details. A C ABI is useful for stable integration with Python,
Rust, Go, or another service language; a later product can layer pybind11 on top without changing
the inference core.

The default engine is lesson 14's generated batch-1-to-4 plan. Rebuild it on the deployment
TensorRT/CUDA/GPU environment; do not copy a serialized plan between platforms. One `TrtSession`
owns one execution context and is not a concurrent request queue. Give each worker its own session
or serialize access at the caller.
