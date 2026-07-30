# 19a - Custom TensorRT ScaleShift Plugin

## Purpose

This lesson implements `output = input * scale + shift` as a complete custom TensorRT plugin: CUDA
kernel, creator fields, registration, format negotiation, shape propagation, serialization,
deserialization, engine build, and C++ runtime validation.

## Prerequisites

- Complete lesson 19 and use the pinned TensorRT/CUDA development container.
- An accessible NVIDIA GPU and `trtexec` are required.

## Deliverables

- `ScaleShift` TensorRT plugin shared library
- Plugin ONNX model, engine-build workflow, and C++ validator
- CPU-reference numerical comparison

## TensorRT 10.14 API

The implementation uses `IPluginV3` with separate core, build, and runtime capability interfaces.
`IPluginCreatorV3One` receives ONNX attributes during the build phase and the serialized field
collection during engine deserialization. The build uses a strongly typed network.

## Lifecycle

- The creator receives ONNX attributes `scale` and `shift` as plugin fields.
- TensorRT clones the V3 plugin and attaches a runtime capability to each execution context.
- `supportsFormatCombination` accepts only linear FP32 input/output.
- `getOutputShapes` preserves the input shape.
- `enqueue` launches on the TensorRT-provided CUDA stream without synchronizing it.
- The `scale` and `shift` fields are serialized into the engine; deserialization reconstructs the V3
  plugin.
- The plugin library must be loaded before deserializing an engine that contains the plugin.

## Build

Configure and compile the plugin library and validator from the repository root:

```bash
cmake -S 19a_custom_tensorrt_plugin -B 19a_custom_tensorrt_plugin/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 19a_custom_tensorrt_plugin/build --parallel
```

## Run

Create the custom-node ONNX graph, build the engine, and run the validator:

```bash
python3 19a_custom_tensorrt_plugin/create_plugin_model.py
/opt/tensorrt/bin/trtexec \
  --stronglyTyped \
  --staticPlugins=19a_custom_tensorrt_plugin/build/libscale_shift_plugin.so \
  --onnx=19a_custom_tensorrt_plugin/outputs/scale_shift.onnx \
  --saveEngine=19a_custom_tensorrt_plugin/outputs/scale_shift.engine \
  --skipInference
./19a_custom_tensorrt_plugin/build/validate_plugin \
  19a_custom_tensorrt_plugin/outputs/scale_shift.engine
```

The convenience wrapper `19a_custom_tensorrt_plugin/build_and_validate.sh` performs the same build
and run sequence. The validator deserializes the engine in C++ and compares four outputs with the
CPU formula. Generated engines remain ignored and must be rebuilt for the deployment
TensorRT/CUDA/GPU environment.

Run CUDA memory checking:

```bash
compute-sanitizer --tool memcheck \
  ./19a_custom_tensorrt_plugin/build/validate_plugin \
  19a_custom_tensorrt_plugin/outputs/scale_shift.engine
```

## Outputs

- The plugin library and validator are ignored build artifacts.
- The custom-node ONNX model and TensorRT engine are environment-specific files under ignored `outputs/`.

## Checkpoints

1. Implement, register, build, serialize, deserialize, and execute a TensorRT `IPluginV3` layer.
2. Launch a CUDA kernel from plugin `enqueue` while respecting dynamic shape and data-type contracts.
3. Validate plugin output numerically against a reference implementation.
