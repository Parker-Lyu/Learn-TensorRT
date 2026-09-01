# 25 - ONNX Graph Surgery for Unsupported TensorRT Operators

## Purpose

This lesson demonstrates how to repair an ONNX model that TensorRT cannot convert because it
contains a custom operator. The demo model contains `com.acme::AcmeSwish`, whose semantics are
`x * sigmoid(x)`.

The lesson uses ONNX GraphSurgeon to replace that unsupported node with standard ONNX operators:

```text
AcmeSwish(x) -> Sigmoid(x) + Mul(x, Sigmoid(x))
```

The broader escalation order is:

1. Rewrite the source model with equivalent supported operators when you control model code.
2. Replace or split nodes in ONNX when the exported graph is the integration boundary.
3. Write a TensorRT plugin only when equivalent supported operators are unavailable or too slow.

This lesson implements the graph-surgery path. Lesson 26 keeps the original custom node and
implements the TensorRT plugin path for the same model.

## Prerequisites

- Use the pinned development container and its ONNX/TensorRT baseline.
- Internet or an available package index is required once to install the isolated GraphSurgeon dependencies.

## Deliverables

- Unsupported-operator demo model and diagnosis tool
- GraphSurgeon rewrite and numerical-validation scripts
- A TensorRT engine built from the rewritten graph
- Diagnosis tests and an isolated dependency setup

The original `outputs/unsupported_swish.onnx` is also the input artifact for Lesson 26.

## Relationship to Lesson 26

This lesson intentionally stops at the graph-surgery solution. The custom node contract is:

```text
domain = com.acme
op_type = AcmeSwish
inputs = 1 FP32 tensor
outputs = 1 tensor with the same shape and type
```

Because `AcmeSwish` can be expressed with standard ONNX operators, this lesson replaces it with
`Sigmoid` and `Mul`, then builds the repaired model with TensorRT. Lesson 26 keeps
`com.acme::AcmeSwish` in the graph and registers a TensorRT `IPluginV3` whose creator matches that
domain and operator name. The two lessons therefore compare graph surgery with plugin integration
on the same source model.

## Setup

The pinned TensorRT container does not expose `onnx_graphsurgeon` globally. Install the exact lesson
versions locally without changing the global ONNX/TensorRT environment:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  ./25_onnx_graph_surgery_plugin/setup_local_deps.sh
```

The mirror variable is optional. Packages are installed under ignored `.deps/`.

## Run

Run every command from the repository root. The inline `#` comments explain what each line does:

```bash
# Disable user-level site-packages so only the container baseline packages are visible.
export PYTHONNOUSERSITE=1
# Add the lesson's isolated .deps directory to the import path so onnx_graphsurgeon is importable.
export PYTHONPATH=25_onnx_graph_surgery_plugin/.deps

# Generate the demo ONNX model whose graph contains the custom com.acme::AcmeSwish node.
python3 25_onnx_graph_surgery_plugin/create_demo_model.py
# Load that model and list its custom-domain nodes to confirm the unsupported operator.
python3 25_onnx_graph_surgery_plugin/diagnose_model.py

# Ask trtexec to build a strongly-typed engine from the unsupported model (expected to fail).
trtexec --stronglyTyped --onnx=25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx
# Expected failure: no importer for com.acme::AcmeSwish.

# Rewrite the graph: replace AcmeSwish with standard Sigmoid + Mul (x * sigmoid(x)), then clean and topologically sort.
python3 25_onnx_graph_surgery_plugin/rewrite_with_graphsurgeon.py
# Run the rewritten model in ONNX Runtime and compare its output against a NumPy reference.
python3 25_onnx_graph_surgery_plugin/validate_rewrite.py

# Build a strongly-typed TensorRT engine from the repaired standard-operator graph and save it.
trtexec \
  --stronglyTyped \
  --onnx=25_onnx_graph_surgery_plugin/outputs/rewritten_swish.onnx \
  --saveEngine=25_onnx_graph_surgery_plugin/outputs/rewritten_swish.engine
```

### Handoff to Lesson 26

Keep `outputs/unsupported_swish.onnx` as the plugin-path input. Do not replace it with
`rewritten_swish.onnx` when starting Lesson 26:

```

text
unsupported_swish.onnx
  -> Lesson 25 GraphSurgeon
  -> rewritten_swish.onnx
  -> TensorRT engine

unsupported_swish.onnx
  -> Lesson 26 AcmeSwish TensorRT plugin
  -> TensorRT engine
```

The repaired graph implements `swish(x) = x * sigmoid(x)` with two standard ONNX nodes. Cleanup and
topological sorting remove the disconnected custom node before export.

## Outputs

- The unsupported model, rewritten model, validation evidence, and rebuilt engine are generated
  under ignored `outputs/`.
- The expected parser failure is evidence only when its command and log are retained.
- `unsupported_swish.onnx` is the handoff input for Lesson 26; this lesson's engine is built from
  `rewritten_swish.onnx`.

## Tests

Run the Python tests from the repository root:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=25_onnx_graph_surgery_plugin/.deps \
python3 -m unittest discover -s 25_onnx_graph_surgery_plugin/tests -v
```

## Checkpoints

1. Identify the domain and operator type of an unsupported ONNX node.
2. Explain why `com.acme::AcmeSwish` cannot be parsed by TensorRT without additional handling.
3. Rewrite and validate a small ONNX graph with ONNX GraphSurgeon.
4. Explain when graph surgery is preferable to a TensorRT plugin.
5. Identify `unsupported_swish.onnx` as the input artifact for Lesson 26.
