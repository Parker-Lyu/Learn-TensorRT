# 12 - YOLOv8 INT8 Quantization Engineering

本课使用 `nvcr.io/nvidia/pytorch:25.11-py3`（TensorRT 10.14.1.48、CUDA 13.0）完成一次可复现的 YOLOv8 部署评估。课程主线是显式 `QuantizeLinear`/`DequantizeLinear`（Q/DQ）训练后量化；所有精度结论来自同一数据、同一预处理和同一质量契约。

## 学习目标

1. 下载并固定 COCO 数据集，生成带 SHA-256 的 calibration/validation manifest。
2. 在完整验证集上评估 PyTorch FP32、PyTorch FP16、TensorRT FP32 和 TensorRT FP16 基线。
3. 使用 ModelOpt 生成 Q/DQ 图，在 TensorRT 10.14 strongly typed network 中构建 INT8 引擎。
4. 使用预先声明的 mAP50-95、mAP50、precision、recall gate 判断 INT8 是否可接受。
5. 仅对通过质量 gate 的候选进行匹配性能测试。
6. 在独立参考目录中阅读 TensorRT legacy entropy calibrator API，并将其结果作为对照，不作为推荐路径。

## 运行顺序

所有 GPU 命令均在课程基线容器内，从仓库根目录执行。若还没有开发容器，可在仓库根目录
构建并启动课程镜像（需要 NVIDIA Container Toolkit）：

```bash
docker build -f docker/Dockerfile.dev -t learn-tensorrt:dev .
docker run --gpus all --rm -it --name learn-tensorrt \
  -v "$PWD:/workspace/Learn-TensorRT" -w /workspace/Learn-TensorRT \
  learn-tensorrt:dev
```

完整命令见
[`docs/reproduction.md`](docs/reproduction.md)。

```bash
python3 assets/coco/prepare_coco.py
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py --materialize
python3 12_yolov8_int8_quantization_engineering/tools/analyze_calibration_representativeness.py
python3 12_yolov8_int8_quantization_engineering/tools/verify_preprocessing_parity.py
```

导出课程使用的静态 ONNX，然后建立四个基线，并导出、构建和检查 Q/DQ INT8：

```bash
(cd 05_torch_to_onnx && python3 export_yolov8_onnx.py)
python3 12_yolov8_int8_quantization_engineering/modelopt/export_qdq.py \
  --high-precision fp16 --name yolov8n_qdq_fp16
python3 12_yolov8_int8_quantization_engineering/modelopt/build_engines.py
python3 12_yolov8_int8_quantization_engineering/compare_engines.py \
  --experiment-id modelopt_qdq_int8
python3 12_yolov8_int8_quantization_engineering/modelopt/inspect_precision.py
```

`compare_engines.py` 在同一验证集上生成统一 JSON/Markdown 结果。INT8 必须同时满足相对
PyTorch FP32 和 TensorRT FP16 的 gate；失败候选不会进入性能结论。

仓库中的 [`reports/12_int8_quantization.md`](../reports/12_int8_quantization.md) 是一次完整
TensorRT 10.14 复现实验的精简证据。它展示如何从质量 gate、Engine Inspector 和匹配性能
共同得出部署结论；学习者仍应在自己的 GPU 和驱动环境中重新生成全部本地证据。

## 质量契约

`configs/quality_contract.json` 固定输入 shape、后处理、指标实现和阈值。不要在看到结果后
修改阈值。任何 manifest、模型、预处理或运行时身份变化都要求重新构建和重新评估。

## Legacy API 参考

`reference_legacy_calibrator/` 只展示 `IInt8EntropyCalibrator2` 的历史接口、cache 身份和
独立评估命令，帮助读者理解 calibration cache 与显式 Q/DQ 的差异。该目录不是主线实现，
不影响 Q/DQ 候选的 gate 或部署推荐。

## 输出与测试

引擎、cache、预测、性能采样和中间报告均写入被忽略的 `outputs/`。运行 CPU 单元测试：

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 12_yolov8_int8_quantization_engineering/tests -v
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 12_yolov8_int8_quantization_engineering/modelopt -p 'test_*.py' -v
```

TensorRT、CUDA、PyTorch 和 ModelOpt 版本由 `configs/environments.json` 记录；没有 GPU 或
容器时只能执行静态检查和不依赖运行时的测试，不能声称完成引擎验证。
