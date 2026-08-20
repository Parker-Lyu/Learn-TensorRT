# 25 - ONNX Graph Surgery and Plugin Escalation

## Purpose

This lesson demonstrates the escalation order for unsupported operators with a runnable failure:

1. Rewrite the source model with equivalent supported operators when you control model code.
2. Replace or split nodes in ONNX when the exported graph is the integration boundary.
3. Write a TensorRT plugin only when equivalent supported operators are unavailable or too slow.

## Prerequisites

- Use the pinned development container and its ONNX/TensorRT baseline.
- Internet or an available package index is required once to install the isolated GraphSurgeon dependencies.

## Deliverables

- Unsupported-operator demo model and diagnosis tool
- GraphSurgeon rewrite and numerical-validation scripts
- Diagnosis tests and an isolated dependency setup

## Plugin Interface Awareness

TensorRT 10.14 uses `IPluginV3` capability interfaces to separate core identity, build-time shape
and format negotiation, and runtime execution. A creator constructs plugins from fields, and engine
deserialization must find the registered creator and compatible plugin library. Lesson 26 provides
the runnable V3 implementation. The repaired standard-operator graph is built as a strongly typed
network so TensorRT follows the data types declared by ONNX.

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
# 关闭用户级 site-packages，避免宿主/用户 Python 包污染容器基线环境
export PYTHONNOUSERSITE=1
# 把本课隔离安装的依赖目录 .deps 加入模块搜索路径，以导入 onnx_graphsurgeon
export PYTHONPATH=25_onnx_graph_surgery_plugin/.deps

# 生成一张包含自定义算子 com.acme::AcmeSwish 的演示 ONNX 模型
python3 25_onnx_graph_surgery_plugin/create_demo_model.py
# 加载该模型并列出其中的自定义域节点，确认不支持算子
python3 25_onnx_graph_surgery_plugin/diagnose_model.py

# 用 trtexec 尝试从含不支持算子的模型构建强类型引擎（预期失败）
trtexec --stronglyTyped --onnx=25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx
# Expected failure: no importer for com.acme::AcmeSwish.

# 用 ONNX GraphSurgeon 重写图：把 AcmeSwish 替换为标准 Sigmoid + Mul（即 x * sigmoid(x)），并清理、拓扑排序
python3 25_onnx_graph_surgery_plugin/rewrite_with_graphsurgeon.py
# 用 ONNX Runtime 运行重写后的模型，并与 NumPy 参考结果做数值对比
python3 25_onnx_graph_surgery_plugin/validate_rewrite.py

# 从修复后的标准算子图构建强类型 TensorRT 引擎并保存
trtexec \
  --stronglyTyped \
  --onnx=25_onnx_graph_surgery_plugin/outputs/rewritten_swish.onnx \
  --saveEngine=25_onnx_graph_surgery_plugin/outputs/rewritten_swish.engine
```

The repaired graph implements `swish(x) = x * sigmoid(x)` with two standard ONNX nodes. Cleanup and
topological sorting remove the disconnected custom node before export.

## Outputs

- The unsupported model, rewritten model, validation evidence, and rebuilt engine are generated
  under ignored `outputs/`.
- The expected parser failure is evidence only when its command and log are retained.

## Tests

Run the Python tests from the repository root:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=25_onnx_graph_surgery_plugin/.deps \
python3 -m unittest discover -s 25_onnx_graph_surgery_plugin/tests -v
```

## Checkpoints

1. Diagnose an unsupported ONNX operator and choose among model rewrite, graph surgery, and a TensorRT plugin.
2. Rewrite and validate a small ONNX graph with ONNX GraphSurgeon.
3. Explain the build-time and runtime responsibilities of TensorRT `IPluginV3` capabilities.
