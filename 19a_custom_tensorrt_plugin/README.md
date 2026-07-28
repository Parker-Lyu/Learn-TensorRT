# 19a - Custom TensorRT ScaleShift Plugin

This lesson implements `output = input * scale + shift` as a complete custom TensorRT plugin: CUDA
kernel, creator fields, registration, format negotiation, shape propagation, serialization,
deserialization, engine build, and C++ runtime validation.

## TensorRT 10.14 API

The implementation uses `IPluginV3` with separate core, build, and runtime capability interfaces.
`IPluginCreatorV3One` receives ONNX attributes during the build phase and the serialized field
collection during engine deserialization. The build uses a strongly typed network.

## Build and Validate

```bash
./build_and_validate.sh
```

The script builds `libscale_shift_plugin.so`, creates an ONNX graph containing the custom
`ScaleShift` node, loads the library with `trtexec --staticPlugins`, builds a strongly typed
engine, then deserializes it in C++ and compares four outputs with the CPU formula. Generated
engines remain ignored and must be rebuilt for the deployment TensorRT/CUDA/GPU environment.

Run CUDA memory checking:

```bash
compute-sanitizer --tool memcheck \
  ./build/validate_plugin outputs/scale_shift.engine
```

## Lifecycle

- The creator receives ONNX attributes `scale` and `shift` as plugin fields.
- TensorRT clones the V3 plugin and attaches a runtime capability to each execution context.
- `supportsFormatCombination` accepts only linear FP32 input/output.
- `getOutputShapes` preserves the input shape.
- `enqueue` launches on the TensorRT-provided CUDA stream without synchronizing it.
- The `scale` and `shift` fields are serialized into the engine; deserialization reconstructs the V3
  plugin.
- The plugin library must be loaded before deserializing an engine that contains the plugin.
