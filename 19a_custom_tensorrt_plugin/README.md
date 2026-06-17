# 19a - Custom TensorRT Plugin

Goal: build, load, and validate one small custom TensorRT plugin.

Why it matters:

- Real CV models sometimes contain operators that TensorRT cannot parse or optimize directly.
- A runnable plugin demonstrates TensorRT lifecycle knowledge beyond command-line engine building.

Suggested plugin scope:

- Start with a compact operator such as `ScaleShift`, `Clip`, or `CustomNormalize`.
- Keep the math simple so the lesson focuses on plugin registration, serialization, runtime loading, and validation.

Topics:

- `IPluginV2DynamicExt`
- Plugin creator registration
- Plugin field collection and parameters
- Dynamic shape and data type handling
- CUDA kernel launch from `enqueue`
- Serialization and deserialization
- Building a plugin shared library with CMake
- Loading plugins with `trtexec --plugins`
- Loading plugins from TensorRT C++ runtime code
- ONNX GraphSurgeon replacement with a plugin node
- Reference output comparison with Python, ONNX Runtime, or Polygraphy

Acceptance criteria:

- A plugin shared library builds successfully.
- `trtexec` can load the plugin library and build an engine containing the plugin layer.
- A runtime example loads the plugin-backed engine and runs inference.
- Plugin output is numerically checked against a simple reference implementation.
- You can explain registration, serialization, deserialization, and `enqueue`.
