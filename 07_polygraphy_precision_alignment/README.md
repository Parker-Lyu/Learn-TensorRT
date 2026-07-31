# 07 - Polygraphy Precision Alignment

## Purpose

- Learn a repeatable single-input precision-debug workflow when ONNX Runtime and TensorRT outputs
  disagree.
- Real deployment work is not finished when an engine builds successfully.
- Senior candidates should be able to prove where numerical drift starts instead of guessing
  whether preprocessing, export, precision mode, or TensorRT parsing caused the issue.
- A one-image tensor comparison is a debugging gate, not a dataset-level release criterion. Later
  lessons extend it into multi-image drift statistics and decoded detection-quality comparison.

TensorRT deployment is not finished when an engine builds. The engine must still produce outputs
that match the validated ONNX model closely enough for the target task.

This lesson keeps the comparison narrow and honest:

```text
lesson 05 preprocessed tensor
  -> Polygraphy data loader reads .npy
  -> ONNX Runtime output
  -> TensorRT output
  -> error summary and precision note
```

The raw YOLO output is compared before decode, NMS, visualization, or coordinate mapping. That makes
it easier to tell whether drift starts in model execution or in later postprocessing code.

In production work, this single-sample check is only the first gate:

```text
single tensor alignment
  -> multi-image numerical drift statistics
  -> decoded box/class/score comparison
  -> dataset-level detection quality report
```

Lesson 14 extends this idea when comparing FP32, FP16, and INT8 engines. Lesson 32 should include
both this precision-alignment note and later accuracy-regression evidence.

## Prerequisites

Run the lesson 05 export and validation first:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

Build at least one TensorRT engine with lesson 06:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
```

## Deliverables

- `align_precision.py` controlled comparison workflow
- Saved Polygraphy logs and backend outputs
- `precision_report.json` and generated precision-alignment note

## Directory Layout

- `load_npy_input.py`: Polygraphy data loader that feeds the lesson 05 NCHW `.npy` tensor.
- `align_precision.py`: runs Polygraphy inspection and inference commands, saves logs, and writes a
  compact precision report.
- `polygraphy_cli_compat.py`: local launcher that keeps Polygraphy working with the repository's
  NumPy 2.x environment without changing system packages.
- `outputs/`: generated runner outputs, logs, JSON reports, and Markdown notes. This folder is
  ignored by git.
- `../assets/img.jpeg`: canonical image used to generate the controlled lesson 05 input tensor.
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: validated ONNX model from lesson 05.
- `../06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: default serialized TensorRT engine
  from lesson 06.

## Tolerance Notes

Start strict for FP32:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --rtol 1e-3 --atol 1e-3
```

For FP16, expect larger drift:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --rtol 1e-2 \
  --atol 1e-2 \
  --keep-going
```

Do not loosen tolerance just to make a command pass. The default Polygraphy comparison uses
elementwise relative and absolute tolerance, while `precision_report.json` still records max error,
mean error, P99 error, close fraction, and the index of the largest mismatch. This report can show
whether the backend execution is suspicious, but it cannot prove final detector quality by itself.
Decide whether decoded detections remain acceptable on a representative image set before calling
FP16 or INT8 drift acceptable.

## Expected Report Fields

`precision_report.json` records:

- ONNX path
- engine path
- input `.npy` path and input tensor name
- TensorRT mode
- command lines and log paths
- runner output artifacts
- output names and shapes
- max, mean, median, and P99 absolute error
- largest-mismatch index and values
- tolerance settings
- `np.allclose` result
- likely-cause note

`precision_alignment_note.md` is the short human-readable note that should feed the final benchmark
report.

## Polygraphy Command Reference

`align_precision.py` calls Polygraphy through `polygraphy_cli_compat.py`, which applies the local
NumPy compatibility patch and then forwards arguments to Polygraphy. The displayed command in the
lesson logs starts with `polygraphy`, but the actual Python launcher is:

```bash
python3 07_polygraphy_precision_alignment/polygraphy_cli_compat.py ...
```

The default ONNX inspection command is:

```bash
polygraphy inspect model 05_torch_to_onnx/outputs/yolov8n.onnx \
  --model-type onnx \
  --show layers \
  --log-file 07_polygraphy_precision_alignment/outputs/inspect_onnx.log \
  --log-format no-colors
```

This command does not run inference. It asks Polygraphy to parse the ONNX file and print model
structure information:

- `inspect model`: inspect a model artifact instead of executing it.
- `05_torch_to_onnx/outputs/yolov8n.onnx`: the ONNX model exported and validated in lesson 05.
- `--model-type onnx`: tells Polygraphy how to interpret the file.
- `--show layers`: includes layer-level details, which helps confirm tensor names and shapes.

The ONNX Runtime smoke run uses the `.npy` input through the lesson data loader:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 05_torch_to_onnx/outputs/yolov8n.onnx \
  --onnxrt \
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --log-file 07_polygraphy_precision_alignment/outputs/run_onnxrt.log \
  --log-format no-colors
```

This command runs only the ONNX Runtime backend and saves its raw output tensors:

- `POLYGRAPHY_INPUT_NPY`: path consumed by `load_npy_input.py`; this keeps the controlled input in
  NumPy's binary `.npy` format.
- `POLYGRAPHY_INPUT_NAME`: model input tensor name used in the feed dictionary, normally `images`.
- `run 05_torch_to_onnx/outputs/yolov8n.onnx`: execute the ONNX model.
- `--onnxrt`: enables the ONNX Runtime runner.
- `--data-loader-script`: points Polygraphy to the Python function that yields input tensors.
- `--save-outputs`: writes the runner output in Polygraphy's JSON output format for later loading
  and reporting.

When `--trt-mode engine` is used, the serialized TensorRT engine is inspected first:

```bash
polygraphy inspect model 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --show layers \
  --log-file 07_polygraphy_precision_alignment/outputs/inspect_engine.log \
  --log-format no-colors
```

This command checks the serialized TensorRT artifact before comparison:

- `06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: the engine built in lesson 06.
- `--model-type engine`: tells Polygraphy this file is a serialized TensorRT engine, not ONNX.
- `--show layers`: prints engine layer details when available, useful for confirming the expected
  artifact is being compared.

The default engine comparison command reuses the saved ONNX Runtime output as the reference and
feeds the same `.npy` input to TensorRT:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --trt \
  --load-outputs 07_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 07_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

This command runs the serialized TensorRT engine and compares it against the saved ONNX Runtime
reference:

- `run 06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: execute the serialized engine.
- `--model-type engine`: interprets the input artifact as a TensorRT engine.
- `--trt`: enables the TensorRT runner.
- `--load-outputs`: loads the ONNX Runtime outputs saved by the earlier smoke run, so Polygraphy can
  compare TensorRT output against that reference.
- `--data-loader-script`: feeds the exact same `.npy` input tensor to TensorRT.
- `--save-outputs`: saves the TensorRT runner output and comparison artifact.
- `--rtol` and `--atol`: relative and absolute tolerances used by Polygraphy's elementwise
  comparison.

When `--trt-mode build` is used, Polygraphy builds a temporary TensorRT engine from the ONNX model
and compares ONNX Runtime and TensorRT in one run:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 05_torch_to_onnx/outputs/yolov8n.onnx \
  --onnxrt \
  --trt \
  --trt-min-shapes images:[1,3,640,640] \
  --trt-opt-shapes images:[1,3,640,640] \
  --trt-max-shapes images:[1,3,640,640] \
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 07_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

This command is useful when a serialized lesson 06 engine is not available:

- `run 05_torch_to_onnx/outputs/yolov8n.onnx`: starts from the ONNX model instead of an engine file.
- `--onnxrt` and `--trt`: runs both backends in the same Polygraphy invocation.
- `--trt-min-shapes`, `--trt-opt-shapes`, and `--trt-max-shapes`: define the TensorRT optimization
  profile for the static YOLO input shape. The `images` prefix must match the model input name.
- `--data-loader-script`: still feeds the same controlled `.npy` input tensor.
- `--save-outputs`, `--rtol`, and `--atol`: save runner results and apply the same numerical
  tolerance policy as the serialized-engine path.

`--rtol`, `--atol`, `--input-name`, `--input-npy`, `--engine`, and `--output-dir` replace the values
shown above when those options are passed to `align_precision.py`.

Common logging options:

- `--log-file`: stores Polygraphy's detailed console output in `outputs/` so the lesson can keep a
  reproducible debug record.
- `--log-format no-colors`: removes terminal color codes from logs, making them easier to search and
  include in reports.

## Run

### Input Tensor

This lesson directly reuses the controlled input tensor saved by lesson 05 from `assets/img.jpeg`:

```text
05_torch_to_onnx/outputs/input_nchw_float32.npy
```

`align_precision.py` passes this `.npy` file to Polygraphy through `load_npy_input.py` and
`--data-loader-script`. The tensor remains in NumPy's binary format; no intermediate input JSON is
generated.

Use a different input tensor when experimenting with another preprocessed sample:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
  --input-npy path/to/input_nchw_float32.npy \
  --skip-trt
```

Override the input tensor name if the ONNX inspection report shows a different name:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --input-name images --skip-trt
```

### Smoke Test ONNX Runtime

Run only the ONNX Runtime side when you want to verify Polygraphy setup before using TensorRT:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --skip-trt
```

This writes:

- `outputs/inspect_onnx.log`
- `outputs/run_onnxrt.log`
- `outputs/onnxrt_outputs.json`
- `outputs/precision_report.json`
- `outputs/precision_alignment_note.md`

Compare the lesson 05 ONNX model against the lesson 06 FP32 engine:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py
```

The default comparison uses:

```text
ONNX:   05_torch_to_onnx/outputs/yolov8n.onnx
Engine: 06_trtexec_engine/outputs/yolov8n_static_fp32.engine
Input:  05_torch_to_onnx/outputs/input_nchw_float32.npy
```

The command writes:

- `outputs/inspect_onnx.log`
- `outputs/inspect_engine.log`
- `outputs/run_onnxrt.log`
- `outputs/compare_onnxrt_trt.log`
- `outputs/onnxrt_outputs.json`
- `outputs/trt_compare_outputs.json`
- `outputs/precision_report.json`
- `outputs/precision_alignment_note.md`

Use a different engine, for example the FP16 engine from lesson 06:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --rtol 1e-2 \
  --atol 1e-2 \
  --keep-going
```

`--keep-going` is useful for FP16 or INT8 experiments because Polygraphy may return a nonzero status
when tolerance fails, but the mismatch evidence is still valuable.

Let Polygraphy build a temporary TensorRT engine from ONNX when a serialized engine is not available:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --trt-mode build
```

The serialized lesson 06 engine is preferred for normal course work because it compares the exact
artifact that later C++ lessons will load.

## Outputs

- The runnable commands above produce the files and console evidence described in `Deliverables`.
- Generated build and runtime artifacts remain in the lesson's ignored build or output directory.

## Checkpoints

- Run `--skip-trt` and confirm Polygraphy can execute the ONNX model with the saved input tensor.
- Compare FP32 ONNX Runtime and FP32 TensorRT output, then explain the largest mismatch.
- Repeat with the FP16 engine and explain why a different tolerance may be reasonable.
- Intentionally set `--rtol 1e-8 --atol 1e-8` and inspect the generated mismatch evidence.
- Confirm the ONNX model and TensorRT engine came from the same export before debugging
  postprocessing.
- Save the final `precision_alignment_note.md` as evidence for lesson 31.
- Explain why a single-input allclose result is useful for debugging but insufficient for release
  approval.

## Appendix: ONNX → TensorRT 精度对齐与排查笔记

模型从 ONNX 转换为 TensorRT 后，原始输出可能与 ONNX Runtime（ORT）产生数值偏差。
排查的核心原则是：**固定输入和模型版本，先建立 FP32 基线，再检查 FP16 或 INT8；先定位
误差从哪里开始，再决定如何修复。**

这份笔记是故障排查清单，不替代本课生成的单输入报告，也不替代多样本、解码后检测结果和
数据集级任务指标。单个张量的 `allclose` 结果只能回答“这个样本的原始输出是否在给定容差内”，
不能单独证明模型精度满足上线要求。

### 1. 核心工具

| 工具 | 主要用途 | 使用边界 |
| --- | --- | --- |
| **Polygraphy** | 运行 ORT/TRT、保存输入输出、逐张量比较、缩减失败子图和搜索精度约束 | 逐层标记输出可能改变 TensorRT 的融合和内存占用，结果应作为定位线索 |
| **Netron** | 查看节点、张量名称、属性及图结构 | 只负责可视化，不验证实际运行语义 |
| **trtexec** | 构建和分析引擎，记录解析日志、层信息、精度、tactic 与性能 | 性能分析不能代替数值验证 |
| **Polygraphy Surgeon / ONNX GraphSurgeon** | 提取子图、常量折叠和有依据的图修改 | 修改后必须重新运行 ONNX Checker、ORT 基线和 TRT 对比 |

### 2. 开始前先固定比较条件

很多“精度问题”实际上是比较条件不同。至少确认以下项目一致：

1. ONNX 模型与 TensorRT 引擎来自同一次导出；序列化引擎不能脱离其构建环境讨论。
2. 两个后端读取完全相同的预处理后输入张量，而不是分别读取并预处理同一张图片。
3. 输入名称、形状、数据类型、布局和动态 Shape profile 一致。
4. 比较的是同一语义阶段的输出，例如都在 decode 和 NMS 之前。
5. 随机种子、插件版本、TensorRT/CUDA/GPU/驱动以及构建选项已记录。
6. 同时查看绝对误差、相对误差、误差分位数、`NaN`/`Inf` 和任务级指标；不要只看一个
   最大相对误差，因为参考值接近零时该指标会被放大。

推荐时间线：

```text
确认比较条件
  → 验证 ORT 自身基线
  → 对齐 ORT 与 TRT FP32
  → 若 FP32 失败，定位解析、图语义或 tactic 问题
  → 若 FP32 通过，再测试 FP16 或显式 Q/DQ INT8
  → 对定位出的局部问题做最小修复
  → 回归单输入、代表性样本集和任务级指标
```

### 3. 第一阶段：FP32 基线

#### 3.1 从本课的受控比较开始

先执行本课已有工作流，它会复用课程 05 保存的 `.npy` 输入，并比较 ORT 与课程 06 的
FP32 引擎：

```bash
python3 07_polygraphy_precision_alignment/align_precision.py
```

不要为了让结果变成 `PASSED` 而直接放宽容差。先检查报告中的最大、均值和 P99 绝对误差、
最大误差位置、输出形状及异常值，再根据业务指标制定容差。

#### 3.2 单独验证 TF32 的影响

在 Ampere 及更新架构上，TF32 可能使 FP32 网络选择吞吐更高但数值行为不同的实现。
不过，**本课程固定的 Polygraphy 0.49.26 使用 `--tf32` 显式启用 TF32，并不存在
`--no-tf32` 参数**。因此，用 Polygraphy 从 ONNX 临时构建基线时先不传 `--tf32`，再单独
增加该参数做 A/B 对比：

```bash
# 基线：不显式启用 TF32
polygraphy run model.onnx --onnxrt --trt \
  --data-loader-script your_data_loader.py \
  --rtol 1e-4 --atol 1e-4

# 对照实验：显式启用 TF32
polygraphy run model.onnx --onnxrt --trt --tf32 \
  --data-loader-script your_data_loader.py \
  --rtol 1e-4 --atol 1e-4
```

不同前端或 TensorRT API 对构建标志的默认处理可能不同，不能把上述命令的默认行为直接推广到
所有构建方式。应以实际命令、BuilderConfig 和构建日志为准。若只有 TF32 对照失败，说明差异
与所选数值路径相关，但是否禁用仍应由任务级精度与性能数据决定。

#### 3.3 逐张量寻找最早的明显分歧

确认最终输出确实失败后，可临时将中间张量标记为输出：

```bash
polygraphy run model.onnx \
  --onnxrt --trt \
  --data-loader-script your_data_loader.py \
  --onnx-outputs mark all \
  --trt-outputs mark all \
  --rtol 1e-3 --atol 1e-3
```

重点观察拓扑路径上最早出现明显误差增长的位置，而不是机械地认定日志中第一个 `FAILED` 的
节点就是根因。分支图、接近零的参考值、节点排序以及误差累积都可能干扰判断。此外，标记所有
输出会抑制部分融合并增加显存占用；大型模型应按可疑区域分批标记张量，并用未插桩的完整网络
复现最终误差。

#### 3.4 提取最小可复现子图

找到可疑张量后，可提取该区域。Polygraphy 0.49.26 的元数据格式是
`名称:[形状]:数据类型`：

```bash
polygraphy surgeon extract model.onnx \
  --inputs 'suspect_input:[1,64,80,80]:float32' \
  --outputs 'suspect_output:float32' \
  --output single_region.onnx
```

提取子图并不自动证明某个算子有错。为了复现完整网络中的问题，应尽量保存并使用原网络运行到
子图边界时的真实输入；随机输入可能覆盖不到相同数值范围。提取后先运行 ONNX Checker 和 ORT，
再用相同输入比较 TensorRT。

大网络也可以使用实验性的 `debug reduce` 自动缩减失败图：

```bash
polygraphy debug reduce model.onnx \
  --output reduced_failing_model.onnx \
  --model-input-shapes 'images:[1,3,640,640]' \
  --check polygraphy run polygraphy_debug.onnx \
    --onnxrt --trt \
    --data-loader-script your_data_loader.py \
    --rtol 1e-3 --atol 1e-3
```

`debug reduce` 根据检查命令的退出状态判断当前子图是好是坏，因此必须先确认检查命令能稳定
复现目标故障。对于动态 Shape 和 Shape 子图，必要时先固定调试形状；不要把缩减结果误当成
完整模型的最终修复。

### 4. FP32 问题的修复顺序

#### 4.1 优先修复导出源

若 ONNX 图的属性或语义本身不符合预期，优先修改 PyTorch 导出源并重新导出。例如：

- 明确 `Resize`/`interpolate` 的模式和 `align_corners` 等语义；
- 在算法允许时为归一化或除法使用合理的 `epsilon`；
- 避免依赖未定义行为、输入相关控制流或导出器无法可靠表达的操作。

不要为了让 ORT 与 TRT 对齐而随意改动算子属性。修改必须符合原模型语义，并重新验证 PyTorch
与 ONNX 的输出。

#### 4.2 无法重导出时再做 ONNX 图手术

下面的示例只演示操作方式；`half_pixel` 是否正确必须由原框架语义决定：

```python
import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load("model.onnx"))
for node in graph.nodes:
    if node.name == "Resize_45" and node.op == "Resize":
        node.attrs["coordinate_transformation_mode"] = "half_pixel"

graph.cleanup().toposort()
fixed = gs.export_onnx(graph)
onnx.checker.check_model(fixed)
onnx.save(fixed, "model_fixed.onnx")
```

常量折叠可用于简化可静态计算的子图，但不是通用的精度修复方法：

```bash
polygraphy surgeon sanitize model.onnx \
  --fold-constants \
  --output sanitized_model.onnx
```

#### 4.3 最后才考虑插件或 tactic 级诊断

只有在确认 TensorRT 10.14 无法正确表达或实现目标语义，且无法合理重写模型时，才考虑自定义
插件。TensorRT 10.14 的新插件应优先采用 `IPluginV3` 接口；旧的 `IPluginV2` 系列和已弃用
插件不应作为新课程代码的默认方案。优先检查 TensorRT 是否已有原生层或受支持的 ONNX 映射。

若怀疑某个 tactic 或融合路径存在问题，应保存构建日志与可复现模型，并使用 Polygraphy 的
`debug build`、tactic replay 或精度约束进行受控实验。TensorRT 没有一个适用于所有网络的
“关闭指定融合”通用开关，也不应在没有证据时随意禁用 tactic source。

### 5. 第二阶段：FP16 与显式 Q/DQ INT8

只有 FP32 基线通过后，才能把新增偏差归因于低精度路径。先分别测试 FP16 和 INT8，不要同时
改变精度、输入、优化 profile 和模型版本。

#### 5.1 FP16

逐层检查误差首次明显放大的区域，同时检查：

- 中间张量是否出现 `NaN`、`Inf`、上溢或下溢；
- 归一化、指数、除法和大范围 reduction 周围的数值范围；
- 层的计算精度与输出张量类型，二者并不是同一个概念；
- 放宽容差后，解码结果和任务指标是否仍满足质量门槛。

Polygraphy 可以实验性地搜索哪些层需要更高精度。该命令需要一个能自动判定好坏的 `--check`
命令；省略时会进入交互模式：

```bash
polygraphy debug precision model.onnx \
  --fp16 \
  --mode bisect \
  --precision float32 \
  --check your_accuracy_check_command
```

搜索结果是诊断线索，不是最终设计。对于弱类型网络，可通过层精度约束做验证；对于强类型网络
或显式量化模型，应在模型类型和量化图允许的边界内调整，不能假定
`ILayer::setPrecision(kFLOAT)` 对所有 TensorRT 10.14 网络都有效。

#### 5.2 INT8

本课程以 ModelOpt 导出的**显式 Q/DQ** 模型为主，完整工程流程在课程 14。TensorRT 10.14 中
基于 `IInt8Calibrator` 的隐式量化流程已经弃用，因此不应把切换
`ENTROPY_CALIBRATION_2`/`MINMAX_CALIBRATION` 作为本课程的首选修复方法。

显式 Q/DQ INT8 应重点检查：

1. 代表性校准数据是否复用了部署时完全相同的预处理；
2. Q/DQ 的 scale、粒度、轴和对称性是否符合目标算子与硬件约束；
3. 是否存在异常值导致有效量化范围被少数样本占据；
4. 敏感层是否需要保留为 FP16/FP32，并由 ModelOpt 配置和导出的 Q/DQ 图明确表达；
5. 多样本张量漂移、解码后检测结果及数据集级指标是否同时通过。

不存在对所有模型都正确的“100～500 张校准图片”固定答案。样本数量只是变量之一，分布覆盖、
类别与场景代表性、预处理一致性和最终任务指标更重要。

### 6. 常见现象速查

| 现象 | 优先检查 | 不应直接下的结论 |
| --- | --- | --- |
| FP32 最终输出不一致 | 模型/输入版本、Shape、TF32 配置、首个明显漂移张量、解析与 tactic 日志 | “一定是量化问题” |
| `Resize` 附近漂移 | mode、坐标变换、nearest rounding、opset 和原框架语义 | “统一改成 `half_pixel` 就能修复” |
| `GridSample` 或边界算子漂移 | padding、坐标归一化、`align_corners`、TensorRT 10.14 实际支持情况 | “必须换第三方插件” |
| NMS 结果顺序不同 | 阈值附近候选框、相同分数排序、输出集合与任务指标 | “原始浮点网络已失真” |
| FP16 出现巨大误差 | `NaN`/`Inf`、动态范围、敏感 reduction/norm/exp、层精度 | “所有 Norm/Softmax 都必须回退 FP32” |
| INT8 深层输出失真 | Q/DQ scale、校准分布、异常值、预处理、敏感层策略 | “只要增加校准图片数量就能解决” |
| 常量或 Shape 子图异常 | shape inference、动态维度、可折叠子图及解析日志 | “常量折叠一定安全且能修复精度” |

### 7. 最终决策树

```text
发现 ORT 与 TRT 输出偏差
 ├─ 比较条件是否完全一致？
 │   ├─ 否：先统一模型、输入、Shape、输出语义和构建配置
 │   └─ 是：运行 FP32 基线
 ├─ FP32 是否通过已定义的数值门槛？
 │   ├─ 否：检查 TF32 配置 → 分批标记中间张量 → 保存边界输入
 │   │      → 提取/缩减可复现子图 → 修复导出语义、解析或已证实的 tactic 问题
 │   └─ 是：分别运行 FP16 或显式 Q/DQ INT8
 ├─ 低精度是否新增不可接受的漂移？
 │   ├─ 是：定位首个明显放大区域 → 检查动态范围/QDQ scale
 │   │      → 对敏感区域采用有依据的混合精度或重新量化
 │   └─ 否：进入多样本验证
 └─ 单输入、代表性样本集、解码结果和任务级指标全部通过后，才形成发布证据
```

每次修复后都应从完整模型重新执行基线，而不是只验证提取出的子图。最终报告应同时保留模型
标识、环境身份、精度配置、输入来源、容差、数值统计和任务级验证结论。
