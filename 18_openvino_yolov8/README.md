# 18 - OpenVINO YOLOv8

Goal: deploy the same YOLOv8n ONNX model with OpenVINO for CPU-focused comparison.

Topics:

- OpenVINO Runtime
- Model compilation
- CPU inference
- Async infer requests
- `benchmark_app`

Acceptance criteria:

- The ONNX model runs with OpenVINO.
- Latency is compared with TensorRT on the same input size.
