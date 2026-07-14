# 19a - Custom TensorRT ScaleShift Plugin

This lesson implements `output = input * scale + shift` as a complete custom TensorRT plugin: CUDA
kernel, creator fields, registration, format negotiation, shape propagation, serialization,
deserialization, engine build, and C++ runtime validation.

## Pinned API Boundary

The course container uses TensorRT 8.6.1. The runnable implementation therefore uses
`IPluginV2DynamicExt`, the correct dynamic-shape interface for this pinned legacy environment.
Current TensorRT releases use `IPluginV3` capability interfaces; the concepts map, but the class
surface does not. Porting requires separate core/build/runtime capabilities and current creator
registration—not a class rename.

## Build and Validate

```bash
./build_and_validate.sh
```

The script builds `libscale_shift_plugin.so`, creates an ONNX graph containing the custom
`ScaleShift` node, loads the library with `trtexec --staticPlugins`, builds an engine, then deserializes it
in C++ and compares four outputs with the CPU formula. Generated engines remain ignored and must be
rebuilt for the deployment TensorRT/CUDA/GPU environment.

Older examples use `--plugins`; TensorRT 8.6 accepts it but marks it deprecated. This lesson uses
the current flag exposed by the pinned `trtexec` binary.

Run CUDA memory checking:

```bash
compute-sanitizer --tool memcheck \
  ./build/validate_plugin outputs/scale_shift.engine
```

## Lifecycle

- The creator receives ONNX attributes `scale` and `shift` as plugin fields.
- TensorRT clones the plugin while building execution contexts.
- `supportsFormatCombination` accepts only linear FP32 input/output.
- `getOutputDimensions` preserves the input shape.
- `enqueue` launches on the TensorRT-provided CUDA stream without synchronizing it.
- Two floats are serialized into the engine; deserialization reconstructs the plugin.
- The plugin library must be loaded before deserializing an engine that contains the plugin.
