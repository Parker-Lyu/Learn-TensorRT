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

trtexec --stronglyTyped --onnx=19_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx
# Expected failure: no importer for com.acme::AcmeSwish.

python3 19_onnx_graph_surgery_plugin/rewrite_with_graphsurgeon.py
python3 19_onnx_graph_surgery_plugin/validate_rewrite.py

trtexec \
  --stronglyTyped \
  --onnx=19_onnx_graph_surgery_plugin/outputs/rewritten_swish.onnx \
  --saveEngine=19_onnx_graph_surgery_plugin/outputs/rewritten_swish.engine
```

The repaired graph implements `swish(x) = x * sigmoid(x)` with two standard ONNX nodes. Cleanup and
topological sorting remove the disconnected custom node before export.

## Plugin Interface Awareness

TensorRT 10.14 uses `IPluginV3` capability interfaces to separate core identity, build-time shape
and format negotiation, and runtime execution. A creator constructs plugins from fields, and engine
deserialization must find the registered creator and compatible plugin library. Lesson 19a provides
the runnable V3 implementation. The repaired standard-operator graph is built as a strongly typed
network so TensorRT follows the data types declared by ONNX.
