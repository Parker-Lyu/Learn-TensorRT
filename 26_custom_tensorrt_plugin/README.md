# 26 - Implementing a TensorRT Plugin for an Unsupported ONNX Operator

## Purpose

Lesson 25 repairs `com.acme::AcmeSwish` by editing the ONNX graph. This lesson keeps that custom
node and implements the alternative TensorRT solution: a CUDA-backed `IPluginV3` that lets
TensorRT parse, build, serialize, deserialize, and execute the original model.

The plugin implements:

```text
AcmeSwish(x) = x * sigmoid(x)
```

The input model is produced by Lesson 25, so both lessons solve the same conversion failure using
different integration boundaries.

## Prerequisites

- Complete Lesson 25, or generate its model with `python3 25_onnx_graph_surgery_plugin/create_demo_model.py`.
- Use the pinned `learn-tensorrt` development container with TensorRT 10.14 and CUDA 13.
- An NVIDIA GPU and `/opt/tensorrt/bin/trtexec` are required for engine conversion and runtime validation.

## Deliverables

- `AcmeSwish` TensorRT plugin shared library
- CMake build and `trtexec` conversion workflow for Lesson 25's original ONNX model
- C++ engine validator that checks the complete `Add -> AcmeSwish -> Mul` graph
- CUDA memory-checking command

## Plugin Contract

The ONNX parser matches the plugin creator using the custom node identity:

```text
ONNX domain       com.acme
ONNX op_type      AcmeSwish
Plugin namespace  com.acme
Plugin name       AcmeSwish
Plugin version    1
Inputs            one FP32 linear tensor
Outputs           one FP32 tensor with the input shape
```

The plugin has no attributes. `supportsFormatCombination` accepts only FP32 and linear layout;
`getOutputShapes` preserves every input dimension; `enqueue` launches the elementwise CUDA kernel
on TensorRT's stream without synchronizing it. The empty serialization field collection makes the
stateless plugin reconstructible when the engine is deserialized.

## Build

Run from the repository root inside the development container:

```bash
cmake -S 26_custom_tensorrt_plugin -B 26_custom_tensorrt_plugin/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 26_custom_tensorrt_plugin/build --parallel
```

## Run

The wrapper generates Lesson 25's source model, builds the plugin, converts the original graph, and
validates the serialized engine:

```bash
./26_custom_tensorrt_plugin/build_and_validate.sh
```

The equivalent explicit commands are:

```bash
python3 25_onnx_graph_surgery_plugin/create_demo_model.py
/opt/tensorrt/bin/trtexec \
  --stronglyTyped \
  --staticPlugins=26_custom_tensorrt_plugin/build/libacme_swish_plugin.so \
  --onnx=25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx \
  --saveEngine=26_custom_tensorrt_plugin/outputs/acme_swish.engine \
  --skipInference
./26_custom_tensorrt_plugin/build/validate_plugin \
  26_custom_tensorrt_plugin/outputs/acme_swish.engine
```

The validator compares the complete graph against:

```text
(input + 0.25) * sigmoid(input + 0.25) * 1.5
```

Run CUDA memory checking with:

```bash
compute-sanitizer --tool memcheck \
  26_custom_tensorrt_plugin/build/validate_plugin \
  26_custom_tensorrt_plugin/outputs/acme_swish.engine
```

## Outputs

Committed deliverables are the plugin source, CMake files, validator, wrapper, and documentation.
Ignored generated artifacts include the plugin library, build directory, and
`outputs/acme_swish.engine`. The source ONNX model remains under Lesson 25's ignored `outputs/`
directory and is regenerated from repository code.

## Tests

Build and run the end-to-end smoke test in the pinned container:

```bash
./26_custom_tensorrt_plugin/build_and_validate.sh
```

This requires a working NVIDIA driver and GPU. Without GPU access, inspect the source and run
`git diff --check`; engine conversion and CUDA execution cannot be claimed as verified.

## Checkpoints

1. Explain how `com.acme::AcmeSwish` maps to the plugin creator's namespace and name.
2. Trace the `IPluginV3` core, build, and runtime capability responsibilities.
3. Explain why the plugin preserves dynamic dimensions but accepts only FP32 linear tensors.
4. Explain why the TensorRT-provided CUDA stream is used directly in `enqueue`.
5. Verify that the original Lesson 25 model converts only when the plugin library is loaded.
6. Compare the graph-surgery path in Lesson 25 with this plugin path in terms of deployment and
   maintenance trade-offs.
