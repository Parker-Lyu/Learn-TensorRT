# ModelOpt 与 trtexec：TensorRT 精度转换和 Engine 构建

本文是 [06 课程 README](README.md) 的补充阅读，解释精度转换、Engine 构建、验证和性能测量之间的关系。环境以课程固定的 TensorRT 10.14.1.48、CUDA Toolkit 13.0 为准。

文中的命令均从仓库根目录、在持久开发容器 `learn-tensorrt` 内执行。环境准备遵循 [00 课程](../00_environment_check/README.md)，源 ONNX 和代表性输入的生成遵循本课 README 的 Prerequisites。独立的原生命令用于解释参数；日常操作优先使用课程脚本，以保留验证门禁、缓存复用和完整证据。

## 1. 核心概念

### 1.1 ModelOpt 是什么？

NVIDIA ModelOpt 是模型优化工具包，主要通过 Python API 和命令行工具使用，支持模型量化、混合精度转换等工作流。

本课程重点使用其中的 **ONNX AutoCast**，将原始 ONNX 转换为显式包含 FP32/FP16 类型的混合精度 ONNX。主要用途包括：

- 将适合的算子和权重转换为 FP16。
- 让部分对精度敏感的算子保留 FP32。
- 在不同精度的计算节点之间插入必要的 `Cast` 节点。
- 为后续 TensorRT 构建提供具有显式类型信息的 ONNX 图。

ModelOpt 负责模型优化；Engine 构建由 TensorRT 完成，部署端推理则由 TensorRT 运行时及应用程序完成。

### 1.2 trtexec 是什么？

`trtexec` 是 TensorRT 提供的独立命令行程序，主要用于：

- 解析 ONNX 模型并构建 TensorRT Engine。
- 配置网络类型、精度选项和动态 Shape Profile。
- 序列化 Engine，或加载已有 Engine。
- 执行推理性能测试。
- 导出计时数据、层信息和层级 Profile 数据。

可以直接在终端执行它，也可以通过 Python 的 `subprocess` 调用。本课程的 Python 构建脚本采用后一种方式。

### 1.3 三者的职责

| 工具 | 核心职责 |
| --- | --- |
| ModelOpt | 模型优化、混合精度转换、量化等 |
| TensorRT | 解析计算图、选择 Tactic、构建和执行优化后的 Engine |
| trtexec | 通过命令行调用 TensorRT，构建、运行和分析 Engine |

典型部署流程：

```text
PyTorch
   |
   v
导出 ONNX
   |
   v
可选：ModelOpt AutoCast
   |
   v
ONNX
   |
   v
TensorRT Parser
   |
   v
TensorRT Builder
   |-- 选择 Tactic
   |-- 规划内存
   |-- 构建 Engine
   |
   v
序列化 Engine
   |
   v
C++ / CUDA 推理程序
```

ModelOpt 并不是所有 TensorRT 部署项目的必经步骤。是否需要显式转换，取决于模型的类型信息、目标精度和所选构建路线。

## 2. 两种主要的 FP16 部署路线

### 2.1 路线 A：FP32 ONNX 直接构建 FP16 Engine

```text
FP32 ONNX
    |
    v
trtexec --fp16
    |
    v
TensorRT Engine
```

原生命令示例：

```bash
trtexec \
    --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
    --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp16_legacy.engine \
    --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
    --fp16
```

在 TensorRT 10.14 的弱类型网络中，`--fp16` 允许 Builder 选择支持的 FP16 实现。它不保证所有层都使用 FP16，输入输出也不一定随之变成 FP16。需要检查最终 Engine 的层信息和绑定类型。

这条路线流程简单，无须提前改变 ONNX 中的精度，适合复现已有弱类型部署和建立兼容性对照。**本课程将它保留为 legacy 对照路线**，并使用强类型混合精度路线作为后续 FP16 示例的主要产物来源。

### 2.2 路线 B：ModelOpt 生成混合精度 ONNX

```text
FP32 ONNX
    |
    v
ModelOpt AutoCast
    |
    v
混合精度 ONNX
    |
    v
trtexec --stronglyTyped
    |
    v
TensorRT Engine
```

ModelOpt 显式转换 ONNX 中的精度，TensorRT 再根据图中的类型规则推导中间张量类型并构建 Engine。这条路线适合：

- 明确控制部分算子的精度。
- 让敏感算子保留 FP32。
- 检查和复现混合精度图中的类型边界。
- 使用强类型网络工作流。

TensorRT 10.14.1.48 支持本课程使用的 `--stronglyTyped`。强类型约束的是张量类型和算子的类型规则，不能将其理解为每个算子的所有内部计算都采用同一种精度。

### 2.3 两条路线的区别

| 对比项 | 直接构建 FP16 Engine | ModelOpt 混合精度路线 |
| --- | --- | --- |
| 输入 ONNX | 通常是 FP32 | 显式包含 FP32/FP16 类型信息 |
| 精度控制 | Builder 根据精度选项选择实现 | ONNX 类型和算子类型规则约束构建 |
| 是否需要 ModelOpt | 不需要 | 本课程用它生成混合精度图 |
| 类型转换 | 由后端根据所选实现处理 | ONNX 中可以显式存在 Cast |
| 是否保证更快 | 否 | 否 |
| 本课定位 | 兼容性和性能对照 | 推荐的显式混合精度路线 |

两条路线的实际收益取决于模型、GPU、TensorRT 版本、算子实现和精度配置。强类型路线在本课程中更便于表达和检查精度策略，但仍需验证输出并测量最终 Engine。

## 3. ModelOpt AutoCast 的典型用法

### 3.1 命令行调用

以下命令展示 [prepare_fp16_onnx.py](prepare_fp16_onnx.py) 对静态模型执行的主要调用：

```bash
python3 -m modelopt.onnx.autocast \
    --onnx_path 05_torch_to_onnx/outputs/yolov8n.onnx \
    --output_path 06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx \
    --low_precision_type fp16 \
    --keep_io_types \
    --calibration_data 06_trtexec_engine/outputs/static_autocast_input.npz \
    --providers cpu \
    --nodes_to_exclude '/model\.22/.*' \
    --log_level INFO
```

其中的 `.npz` 文件由课程准备脚本从 lesson 05 的已保存输入生成，数组键为 `images`。因此，推荐直接运行准备脚本，统一完成数据准备、转换、结构检查和数值验证；上面的内部调用不构成独立的完整验证流程。

### 3.2 重要参数

| 参数 | 作用 |
| --- | --- |
| `--onnx_path` | 原始 ONNX 文件 |
| `--output_path` | 转换后 ONNX 文件 |
| `--low_precision_type fp16` | 指定低精度类型为 FP16 |
| `--keep_io_types` | 保留模型输入输出的数据类型，本课为 float32 |
| `--calibration_data` | 提供代表性输入数据，供转换流程使用 |
| `--providers cpu` | 在该转换工作流中使用 CPU Execution Provider |
| `--nodes_to_exclude` | 指定排除低精度转换的节点匹配规则 |
| `--log_level INFO` | 输出 INFO 级别日志 |

`calibration_data` 这一参数名不意味着本次在进行 INT8 量化。这里明确设置的低精度类型是 FP16。不同 ModelOpt 版本的参数和数据格式可能变化，应使用课程固定环境中的实现。

### 3.3 Cast 节点的作用

混合精度图可能具有如下结构：

```text
FP32 Input
    |
    v
Cast: FP32 -> FP16
    |
    v
Convolution: FP16
    |
    v
Cast: FP16 -> FP32
    |
    v
Sensitive Operation: FP32
```

`Cast` 改变张量数据类型，使不同精度的算子能够连接。理解它时需要注意：

1. FP16 转回 FP32 无法恢复前面舍入时已经丢失的信息。
2. ONNX 中存在 Cast，不代表最终会执行一个独立的转换 Kernel。
3. TensorRT 可能融合或优化相关操作，需要检查层信息、Profile 和实际性能。
4. FP16 权重和 Cast 节点数量可以辅助检查转换结果，不能证明性能收益或任务精度。

### 3.4 为什么排除检测头？

本课程使用：

```python
DETECTION_HEAD_PATTERN = r"/model\.22/.*"
```

该规则匹配当前 YOLOv8n 导出图中的检测头节点，使其排除在低精度转换范围之外。这是本课采用的精度策略，不是所有检测头都必须使用 FP32。

准备脚本要求该规则至少匹配一个源节点。如果更换模型或导出方式，应先确认节点名称和输出语义，再通过数值及任务级评估判断是否需要调整策略。

## 4. ONNX 转换后的验证

### 4.1 ONNX 结构检查

```python
import onnx

model = onnx.load("06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx")
onnx.checker.check_model(model)
```

结构检查可以发现基本的图结构和类型规则问题，但无法证明转换后的输出与原模型一致。

### 4.2 输入输出接口检查

应检查输入输出数量、名称、数据类型、Rank，以及固定维度和动态维度约束。Rank 相同并不意味着 Shape 约束相同。

当前准备脚本中的 `compatible_io_contract()` 检查数量、名称、类型和 Rank，并记录转换前后的 Shape 注解。它不证明所有 Shape 约束等价；动态模型仍应结合实际执行和边界 Shape 检查。符号维度名称也可能因图转换而变化，不能仅凭符号名称不同就认定接口改变。

### 4.3 数值验证

将完全相同的输入传给原始 ONNX 和转换后的 ONNX，再比较对应的原始输出：

```text
                  Same Input
                  /        \
                 v          v
         Original ONNX   Converted ONNX
                 |          |
                 v          v
        Reference Output  Candidate Output
                  \        /
                   Compare
```

常用指标包括最大绝对误差、平均绝对误差、P99 绝对误差、相对误差、满足容差的元素比例和 `allclose` 结果。比较前还应检查 Shape、dtype、空输出以及 NaN/Inf。

本课使用的逐元素判断为：

```text
abs(reference - candidate) <= atol + rtol * abs(reference)
```

对于 YOLOv8 的 `[N, 84, anchors]` 原始输出，前四个通道是框坐标，后 80 个通道是类别分数，两者分别使用容差。

本课默认坐标容差为 `rtol=0.05, atol=0.05`，分数容差为 `rtol=0.01, atol=0.02`。这是较宽松的转换冒烟门禁；例如参考坐标为 500 时，允许误差达到 25.05 像素。需要结合实际误差分布和后处理结果解释其意义。

**单个代表性输入通过检查，不代表数据集上的 mAP 不会下降。** 正式部署还需要代表性图像集、不同 Shape 和任务级指标。ONNX Runtime CPU 验证也不能替代 TensorRT GPU 输出验证或性能测量。

### 4.4 验证报告如何约束构建？

本课强类型构建会要求转换报告存在、`comparison.passed` 为布尔值 `true`，并核对源 ONNX 和转换后 ONNX 的 SHA-256。报告默认位于转换后 ONNX 旁边，而不是任意 Engine 输出目录中。

构建脚本会在规划任务及执行任务时检查这些条件，包括复用已有 Engine 的情况。转换失败后遗留的 ONNX 文件不会仅凭“文件存在”就被接受。构建脚本读取并核对报告，不会重新执行 ONNX Runtime 数值比较；需要更新验证时，应重新运行准备脚本。

## 5. trtexec 的典型用法

以下原生命令帮助理解构建参数。输出目录须已通过课程准备步骤创建；直接调用 `trtexec` 不会生成课程脚本的 `.build.json` 构建记录或完整 manifest。

### 5.1 构建 FP32 基准 Engine

```bash
trtexec \
    --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
    --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
    --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
    --noTF32
```

`--noTF32` 禁用 TF32 路径，用于建立本课程的严格 FP32 基准。它并不意味着不同硬件和 Kernel 的输出必须逐位相同。

### 5.2 构建传统 FP16 Engine

```bash
trtexec \
    --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
    --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp16_legacy.engine \
    --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
    --fp16
```

这是本课程用于对照的弱类型 FP16 路线。敏感层可能保留较高精度，最终类型和性能应以构建产物为准。

### 5.3 构建强类型混合精度 Engine

```bash
trtexec \
    --onnx=06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx \
    --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp16_strong.engine \
    --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
    --stronglyTyped
```

在本课程的 TensorRT 10.14 强类型路线中，不要再添加 `--fp16`。混合精度来自 ONNX 中的显式类型和类型转换。

本课 FP32 路线禁用 TF32，FP16 路线对剩余 FP32 工作保留 TensorRT 默认 TF32 策略。因此这是两种部署配置的对照，不应把所有性能差异都归因于 FP16。硬件是否支持 TF32、环境是否设置覆盖项，也需要结合环境记录判断。

## 6. trtexec 的常用参数

### 6.1 输入输出与构建参数

| 参数 | 作用 |
| --- | --- |
| `--onnx` | 指定输入 ONNX |
| `--saveEngine` | 保存构建后的 Engine |
| `--loadEngine` | 加载已有 Engine，避免重复构建 |
| `--memPoolSize=workspace:2048` | 设置 Builder Workspace 内存池预算，单位 MiB |
| `--timingCacheFile` | 指定 Tactic 计时缓存文件 |
| `--fp16` | 弱类型网络的 FP16 精度选项 |
| `--noTF32` | 禁用 TF32 |
| `--stronglyTyped` | 使用强类型网络模式 |

Workspace 预算不等于推理阶段的总 GPU 显存占用，也不等于序列化 Engine 的文件大小。

Timing cache 保存可复用的 Tactic 计时信息，减少重复测量。它与已构建 Engine 是不同的产物，具有各自的兼容性边界。即使使用缓存，也可能需要补测部分 Tactic；不能保证跨环境得到完全相同的 Engine。

### 6.2 性能测试参数

| 参数 | 在课程固定版本中的作用 |
| --- | --- |
| `--warmUp=500` | 测量前预热 500 ms |
| `--duration=5` | 性能测量至少运行 5 秒，还会受到迭代次数等条件影响 |
| `--avgRuns=10` | 按连续 10 次迭代的平均值报告相关性能测量 |
| `--percentile=50,90,95,99` | 请求报告指定延迟百分位数 |
| `--profilingVerbosity=detailed` | 在构建时记录详细的层级分析信息 |
| `--dumpLayerInfo` | 输出 Engine 层信息 |
| `--dumpProfile` | 输出层级性能分析信息 |
| `--separateProfileRun` | 将层级 Profile 测量放到单独一次运行，避免在基准测量中附加 Profiler |

这些参数按当前容器的 `trtexec --help` 解释。更换版本后应重新核对，尤其不能把其他 TensorRT 版本的精度参数直接套用到本课程。

### 6.3 结果导出和指标解读

| 参数 | 作用 |
| --- | --- |
| `--exportTimes` | 导出计时样本 |
| `--exportLayerInfo` | 导出 Engine 层信息 |
| `--exportProfile` | 导出层级 Profile 数据 |

构建和性能测量是不同阶段，只是可以由一次 `trtexec` 调用串联。加载已有 Engine 后也可以重新运行性能测试。

本课性能报告需要区分：

- **Latency**：包含 H2D、GPU 计算和 D2H 的推理延迟，不包含应用层图像预处理、解码和 NMS。
- **GPU Compute Time**：GPU 计算部分的时间，具有独立的百分位数。
- **Throughput qps**：每秒推理批次；当 Batch 大于 1 时，不能直接当作每秒图像数。
- **Engine file MiB**：磁盘上的序列化文件大小。
- **TRT context GPU MiB**：日志中上下文创建后 TensorRT 管理的内存总量，不代表进程全部显存或峰值显存。

由于 Enqueue 开销和流水执行，吞吐量通常不能简单写成 `1000 / latency_ms`。比较配置时应保持输入 Shape、传输设置、预热和测量条件一致。

## 7. 动态 Shape 配置

### 7.1 MIN、OPT、MAX

本课默认配置为：

```text
MIN = 1x3x320x320
OPT = 1x3x640x640
MAX = 4x3x640x640
```

MIN 和 MAX 给出 Profile 的输入范围，OPT 为 Builder 提供优化重点。范围内的 Shape 仍需满足模型自身的约束，例如各维度之间的关系以及算子的形状要求。

### 7.2 Profile 与实际运行 Shape

以下示例使用本课推荐的动态强类型产物名称：

```bash
trtexec \
    --onnx=06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx \
    --saveEngine=06_trtexec_engine/outputs/yolov8n_dynamic_fp16_strong.engine \
    --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
    --stronglyTyped \
    --minShapes=images:1x3x320x320 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:4x3x640x640 \
    --shapes=images:1x3x640x640
```

Optimization Profile 描述允许范围及优化重点；`--shapes` 指定本次推理的实际输入 Shape。这里测量的是 Batch=1、640×640，没有遍历 Profile。

若要测量另一个 Shape，可对已构建 Engine 使用 `--loadEngine` 和新的 `--shapes`，并将各次测量保存到不同的日志和导出文件中。无需为每次 Shape 测量重复构建同一个 Engine。

## 8. 本课程的自动化脚本

### 8.1 三个脚本的分工

| 脚本 | 职责 |
| --- | --- |
| [prepare_fp16_onnx.py](prepare_fp16_onnx.py) | 准备代表性输入，执行 AutoCast，检查 ONNX 并写转换验证报告 |
| [build_and_benchmark.py](build_and_benchmark.py) | 检查转换证据，管理构建、缓存和推理测量，生成 manifest |
| [summarize_results.py](summarize_results.py) | 检查日志和导出证据，生成包含 Shape、内存及百分位数的报告 |

常规操作顺序如下；所需前置命令和实际运行输出见 [README 的 Run 部分](README.md#run)：

```bash
python3 06_trtexec_engine/prepare_fp16_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
python3 06_trtexec_engine/build_and_benchmark.py
python3 06_trtexec_engine/summarize_results.py
```

### 8.2 五种构建名称

| 构建名称 | Engine 文件 |
| --- | --- |
| `static_fp32` | `yolov8n_static_fp32.engine` |
| `static_fp16_legacy` | `yolov8n_static_fp16_legacy.engine` |
| `dynamic_fp16_legacy` | `yolov8n_dynamic_fp16_legacy.engine` |
| `static_fp16_strong` | `yolov8n_static_fp16_strong.engine` |
| `dynamic_fp16_strong` | `yolov8n_dynamic_fp16_strong.engine` |

这些 Engine 默认生成在本课 `outputs/` 中。后续课程的 FP16 示例使用带 `_strong` 的名称；legacy 产物用于对照。

### 8.3 构建驱动的执行流程

```text
parse_args()
    |
    v
validate()：检查基本参数和 trtexec
    |
    v
planned()：选择任务、检查文件、核对转换报告与模型哈希
    |
    +-- dry-run：command() 生成并打印构建命令，然后退出
    |
    v
env()：采集兼容性和环境信息
    |
    v
run()：再次核对转换证据
    |-- engine_signature()：计算模型与构建环境标识
    |-- reusable_engine()：核对构建记录和 Engine 哈希
    |-- command()：选择构建命令或加载已有 Engine
    |-- 执行 trtexec、检查新导出文件、按需保存构建记录
    |
    v
manifest()：全部选定任务成功后写入本次测量记录
```

`--dry-run` 会检查所选模型及转换报告，打印的是构建命令；实际运行是否加载已有 Engine，还要由正常执行路径检查构建记录后决定。

### 8.4 常用脚本参数

| 参数 | 作用 |
| --- | --- |
| `--builds` | 选择指定路线，未指定时运行五种路线 |
| `--skip-dynamic` | 跳过动态 Shape 路线 |
| `--skip-strongly-typed` | 跳过强类型路线 |
| `--dynamic-min` / `--dynamic-opt` / `--dynamic-max` | 配置动态 Profile；本脚本在 OPT Shape 进行测量 |
| `--output-dir` | 指定 Engine 和测量证据的输出目录 |
| `--timing-cache-file` | 指定当前 GPU/软件环境下已有的 timing cache |
| `--rebuild` | 强制重新构建 Engine，仍使用 timing cache |
| `--dry-run` | 检查所选输入并打印构建命令，不执行构建或测量 |

### 8.5 两层缓存与产物记录

**Timing cache 复用**减少 Builder 的部分 Tactic 测量工作。**Engine 复用**则直接跳过构建阶段，但仍执行新的性能测试。

脚本仅在模型哈希、Engine 哈希、构建设置和环境信息与 `.build.json` 记录匹配时复用 Engine。仅有一个同名 `.engine` 文件不够；没有可信构建记录时，会使用 timing cache 重新构建。

输出结构包括：

```text
06_trtexec_engine/outputs/
├── *_autocast_fp16.onnx
├── *_autocast_input.npz
├── *_fp16_onnx_validation.json
├── *.engine
├── *.build.json
├── *.log
├── *_times.json
├── *_layers.json
├── *_profile.json
├── trtexec_timing.cache
├── build_manifest.json
└── benchmark_summary.md
```

`build_manifest.json` 记录本次成功完成的选定任务、运行命令、环境、Shape、TF32 策略及测量时间。只选择一条路线时，manifest 也只记录该次选择；需要完整对照时再次运行默认矩阵，匹配的 Engine 会被复用。

所有这些环境相关产物保持忽略，不随源码提交。构建失败不会发布新的成功 manifest；残留文件或旧 Markdown 报告不能证明当前任务通过。报告生成也会检查成功日志、非空有效导出及所需统计数据。

## 9. 工程实践建议

### 第一阶段：建立 FP32 基准

```text
FP32 ONNX -> 严格 FP32 Engine -> GPU 正确性检查 -> 性能测试
```

固定模型、输入、预处理和测量条件，并记录环境身份。

### 第二阶段：建立弱类型 FP16 对照

```text
FP32 ONNX -> legacy FP16 Engine -> GPU 正确性检查 -> 性能测试
```

记录延迟、吞吐量和输出差异，保留这一对照用于理解 Builder 的精度选择。

### 第三阶段：引入显式混合精度

```text
FP32 ONNX -> ModelOpt AutoCast -> ONNX 转换验证
         -> 强类型 Engine -> GPU 正确性检查 -> 性能测试
```

比较两种 FP16 路线的实际结果。显式精度策略有助于解释和复现，但性能仍应由测量决定。

### 第四阶段：扩展动态 Shape

对不同 Batch 和分辨率检查 Profile 覆盖、输出正确性、延迟、吞吐量、显存和构建时间。优先加载已有 Engine 测量其他 Shape，避免重复构建。

本课默认转换门禁仅使用静态 `[1,3,640,640]` 输入和动态 `[2,3,640,640]` 重复图像批次。默认性能矩阵仅测量 `[1,3,640,640]`。它们都不能代替对整个动态范围的正确性验证。

### 第五阶段：进入 C++ 部署

在 Engine 通过相应验证后，使用 C++ TensorRT API 完成 Engine 加载、Execution Context 创建、输入输出 Buffer 管理、CUDA Stream 管理和异步推理，再测量包含预处理和后处理的端到端性能。

C++ 推理程序通常不需要在运行时调用 ModelOpt；模型转换和 Engine 构建已在前置阶段完成。

## 10. 英文技术说明

### ModelOpt and trtexec

ModelOpt is a model optimization toolkit. Its ONNX AutoCast workflow can generate a mixed-precision ONNX graph with explicit FP32 and FP16 types. `trtexec` is a TensorRT command-line tool for building engines, configuring supported precision modes and dynamic shapes, and benchmarking inference performance.

### FP16 engine construction

In TensorRT 10.14, a weakly typed FP16 engine can be built directly from an FP32 ONNX model with `trtexec --fp16`. This compatibility route does not require ModelOpt. This course uses AutoCast and `--stronglyTyped` for its explicit mixed-precision deployment route, while retaining weakly typed engines for comparison.

### Cast nodes

Cast nodes convert tensors between FP32 and FP16 so that operations with different type requirements can connect. TensorRT may optimize or fuse these conversions. The final engine must be measured to determine their actual cost.

### Conversion validation

Run the original and converted models with identical inputs, compare raw outputs using justified numerical tolerances, and evaluate representative inputs and task-level metrics. Validate TensorRT GPU outputs separately after engine construction, then benchmark latency, throughput, and resource use.

### Profiles and runtime shapes

An optimization profile defines the permitted input shape range and an optimum shape for tactic selection. The runtime shape is the actual input shape used for a particular inference execution. A benchmark at one runtime shape does not measure the entire profile.

## 11. 要点回顾

1. **ModelOpt 负责模型优化和显式混合精度转换。**
2. **trtexec 通过命令行调用 TensorRT 构建、运行和分析 Engine。**
3. **弱类型 FP16 构建不要求先生成 FP16 ONNX；显式混合精度也不保证性能更好。**
4. **先验证数值和任务正确性，再比较真实硬件上的延迟、吞吐量和资源占用。**
5. **区分 timing cache 与 Engine 复用，核对兼容性和哈希，并保留每次新测量的证据。**

本文以课程固定版本为背景。可用选项及精确行为以该环境的 `trtexec --help`、ModelOpt 实现和课程脚本为准。
