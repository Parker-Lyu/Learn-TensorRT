# 05 - Torch To ONNX

Goal: export YOLOv8n from PyTorch to ONNX and validate the exported model.

Topics:

- Ultralytics YOLO export
- ONNX opset
- Static shape and dynamic shape
- ONNX Runtime validation
- ONNX graph inspection

Acceptance criteria:

- `yolov8n.onnx` is generated.
- Input and output tensor names are recorded.
- ONNX output is compared with PyTorch output on the same image.
