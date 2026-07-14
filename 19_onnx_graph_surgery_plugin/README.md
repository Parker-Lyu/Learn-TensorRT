# 19 - ONNX Graph Surgery and Plugin Escalation

This lesson demonstrates the escalation order for unsupported operators with a runnable failure:

1. Rewrite the source model with equivalent supported operators when you control model code.
2. Replace or split nodes in ONNX when the exported graph is the integration boundary.
3. Write a TensorRT plugin only when equivalent supported operators are unavailable or too slow.

## Isolated GraphSurgeon Environment

The pinned TensorRT container does not expose `onnx_graphsurgeon` globally. Install the exact lesson
versions locally without changing the global ONNX/TensorRT environment:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  ./19_onnx_graph_surgery_plugin/setup_local_deps.sh
```

The mirror variable is optional. Packages are installed under ignored `.deps/`.

## Run the Failure and Repair

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=19_onnx_graph_surgery_plugin/.deps

python3 19_onnx_graph_surgery_plugin/create_demo_model.py
python3 19_onnx_graph_surgery_plugin/diagnose_model.py

trtexec --onnx=19_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx
# Expected failure: no importer for com.acme::AcmeSwish.

python3 19_onnx_graph_surgery_plugin/rewrite_with_graphsurgeon.py
python3 19_onnx_graph_surgery_plugin/validate_rewrite.py

trtexec \
  --onnx=19_onnx_graph_surgery_plugin/outputs/rewritten_swish.onnx \
  --saveEngine=19_onnx_graph_surgery_plugin/outputs/rewritten_swish.engine
```

The repaired graph implements `swish(x) = x * sigmoid(x)` with two standard ONNX nodes. Cleanup and
topological sorting remove the disconnected custom node before export.

## Plugin Interface Awareness

Modern TensorRT uses `IPluginV3` capability interfaces to separate core identity, build-time shape
and format negotiation, and runtime execution. A creator constructs plugins from fields, and engine
deserialization must find the registered creator and compatible plugin library.

The pinned TensorRT 8.6 environment predates the current V3 workflow, so legacy production code may
use `IPluginV2DynamicExt`: one class owns format support, output dimensions, serialization, cloning,
and `enqueue`. Lesson 19a implements that legacy path honestly and documents the migration boundary;
do not paste a V2 implementation into a current V3 codebase and rename the class.
