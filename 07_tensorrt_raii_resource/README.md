# 07 - TensorRT RAII Resource

Goal: make TensorRT and CUDA C++ code safe for long-running services.

Topics:

- RAII
- `std::unique_ptr`
- Custom deleter
- TensorRT runtime, engine, and context ownership
- CUDA buffer wrapper
- CUDA stream wrapper
- Move-only resource classes
- Exception-safe initialization

Acceptance criteria:

- TensorRT objects are managed by smart pointers or small RAII wrappers.
- CUDA buffers and streams are released automatically.
- The code remains safe if initialization fails halfway.
- You can explain why this matters for 24x7 camera inference services.
