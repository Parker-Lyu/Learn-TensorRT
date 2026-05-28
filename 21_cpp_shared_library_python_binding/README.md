# 21 - C++ Shared Library And Python Binding

Goal: package C++ inference code as a shared library and call it from Python.

Topics:

- C ABI wrapper
- `.so` dynamic library
- Header design
- Struct-based input and output
- `ctypes`
- `pybind11`
- Ownership across language boundaries
- Error code versus exception boundary

Acceptance criteria:

- The TensorRT C++ inference class is compiled into a shared library.
- A Python script loads the library and runs inference.
- The exposed API uses simple inputs and structured outputs.
