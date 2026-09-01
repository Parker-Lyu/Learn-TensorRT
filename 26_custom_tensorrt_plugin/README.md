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

Run the wrapper to generate Lesson 25's source model, build the plugin, convert the original graph,
and validate the serialized engine:

```bash
./26_custom_tensorrt_plugin/build_and_validate.sh
```

<details><summary>Example output (partial)</summary>

```text
[100%] Built target validate_plugin
wrote /workspace/Learn-TensorRT/25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx
Successfully created plugin: AcmeSwish
Engine built in 0.282282 sec.
&&&& PASSED TensorRT.trtexec [TensorRT v101401]
AcmeSwish IPluginV3 full_graph_max_abs=1.49012e-08
```
</details>

To inspect or reproduce the wrapper step-by-step, run these equivalent commands:

```bash
# Regenerate the unsupported AcmeSwish ONNX model.
python3 25_onnx_graph_surgery_plugin/create_demo_model.py
# Build an engine while loading the custom plugin library.
/opt/tensorrt/bin/trtexec \
  --stronglyTyped \
  --staticPlugins=26_custom_tensorrt_plugin/build/libacme_swish_plugin.so \
  --onnx=25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx \
  --saveEngine=26_custom_tensorrt_plugin/outputs/acme_swish.engine \
  --skipInference
# Execute the plugin validator against the serialized engine.
./26_custom_tensorrt_plugin/build/validate_plugin \
  26_custom_tensorrt_plugin/outputs/acme_swish.engine
```

The validator compares the complete graph against:

```text
(input + 0.25) * sigmoid(input + 0.25) * 1.5
```

To check the plugin for CUDA memory errors, run:

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

## Appendix: TensorRT Plugin Fundamentals

This appendix summarizes the plugin concepts used by this lesson. It is intentionally tied to the
`AcmeSwish` implementation rather than being a complete replacement for the TensorRT API reference.

### Why `IPluginV3`?

TensorRT can import standard ONNX operators directly. An unsupported operator needs an extension
that tells TensorRT three things: what the layer is, what tensor contracts it accepts, and how to
execute it. `IPluginV3` separates those responsibilities into capability interfaces:

- **Core** exposes the stable identity (`name`, `version`, and `namespace`) and cloning behavior.
- **Build** participates in format/type negotiation, output-shape inference, configuration, and
  workspace-size calculation while TensorRT builds the engine.
- **Runtime** receives concrete tensor descriptors and launches the operation during inference.

The separation matters because build-time objects and runtime execution contexts can have different
lifetimes and may be used concurrently.

### Plugin Lifecycle

For this lesson the important sequence is:

1. The shared library is loaded and registers `AcmeSwishPluginCreator`.
2. The ONNX parser finds `domain=com.acme` and `op_type=AcmeSwish`, then asks the creator to make a
   plugin instance.
3. During engine building, TensorRT calls shape, data-type, format, configuration, and workspace
   methods. It may clone the plugin while constructing the engine.
4. TensorRT serializes the plugin state into the engine. `AcmeSwish` is stateless, so its serialized
   field collection is empty.
5. When an execution context runs, TensorRT calls `onShapeChange` as needed and then `enqueue` for
   each inference.
6. When an engine is deserialized, the plugin library must be loaded and its creator registered
   before `deserializeCudaEngine` is called.

### Inputs, Outputs, and `enqueue`

The input and output tensors are device buffers owned by the TensorRT execution context. The
`PluginTensorDesc` values describe their concrete data type, format, and dimensions; the `inputs`
and `outputs` arrays contain their device addresses. `enqueue` is the execution part of the plugin:
it launches the CUDA work that transforms input tensors into output tensors.

This implementation computes one element per CUDA thread and uses the CUDA stream supplied by
TensorRT. It must not call `cudaDeviceSynchronize` or otherwise block the stream, because TensorRT
uses that stream to order copies, kernels, and neighboring layers.

### Shape Inference and Dynamic Shapes

`getOutputShapes` runs during engine construction with symbolic `DimsExprs`, not necessarily with
known integer dimensions. `AcmeSwish` is elementwise, so its output expression is exactly the input
expression. At runtime, `enqueue` receives concrete dimensions and calculates the element count
before launching the kernel.

For a plugin whose output shape depends on a dimension value, shape inference must use TensorRT's
expression builder rather than reading a dimension as a host integer. Dynamic-shape support also
requires validating every concrete shape in `onShapeChange` and `enqueue`; assumptions that only
hold for `[1, 4]` would fail for other optimization-profile shapes.

### Data Types: FP32 and FP16

The current plugin deliberately accepts only `FP32` with `kLINEAR` format. `getOutputDataTypes` and
`supportsFormatCombination` enforce that contract, and the kernel uses `float` pointers.

Supporting FP16 is not just changing one enum. The plugin would need to accept `kHALF`, use half or
vectorized half loads/stores, implement numerically appropriate sigmoid math, and validate both
types in `supportsFormatCombination`. The engine's strongly typed network would then select the
declared type; a plugin must never reinterpret FP16 memory as FP32. FP16 can improve throughput and
memory traffic, but it can also introduce larger numerical error, especially in the sigmoid tails.

### Workspace

Workspace is temporary device memory supplied by TensorRT for one plugin execution. The plugin
reports its required size through `getWorkspaceSize`; TensorRT owns allocation and lifetime, and
the pointer arrives in `enqueue`. This implementation needs no temporary storage, so it returns
zero and ignores `workspace`. A reduction, sort, or multi-stage algorithm would request workspace
instead of allocating with `cudaMalloc` inside `enqueue`.

### Plugin Creator, Registration, and Serialization

The creator is the factory and metadata provider. Its name/version/namespace must match the ONNX
custom-node identity, and its field collection describes attributes that can be passed from ONNX.
The creator in this lesson has no fields because `AcmeSwish` has no parameters.

Registration puts the creator in TensorRT's plugin registry. Engine serialization stores the plugin
identity and serialized fields, but it does not make the plugin's compiled CUDA implementation
appear in a new process. Therefore the process that loads an engine must load
`libacme_swish_plugin.so` first. Without the library, TensorRT cannot find the creator and engine
deserialization fails even if the engine file itself is present.

### Reference Validation

Validate at two boundaries:

1. **Operator level:** compare `AcmeSwish(x)` against a NumPy or PyTorch reference using fixed and
   boundary-valued inputs, including negative, zero, and large positive/negative values.
2. **Graph level:** run the complete ONNX graph in ONNX Runtime (or the explicit CPU formula), run
   the TensorRT engine with the same input, and compare final outputs with stated `rtol`/`atol`.

The validator in this lesson uses the graph-level reference
`(input + 0.25) * sigmoid(input + 0.25) * 1.5`. For a larger model, retain the input manifest,
reference backend/version, maximum absolute error, relative error, and pass/fail threshold so the
comparison can be reproduced. If FP16 is enabled, choose tolerances based on the intended precision
and compare distributions or task metrics as well as elementwise error.

## Checkpoints

1. Explain how `com.acme::AcmeSwish` maps to the plugin creator's namespace and name.
2. Trace the `IPluginV3` core, build, and runtime capability responsibilities.
3. Explain why the plugin preserves dynamic dimensions but accepts only FP32 linear tensors.
4. Explain why the TensorRT-provided CUDA stream is used directly in `enqueue`.
5. Verify that the original Lesson 25 model converts only when the plugin library is loaded.
6. Compare the graph-surgery path in Lesson 25 with this plugin path in terms of deployment and
   maintenance trade-offs.
