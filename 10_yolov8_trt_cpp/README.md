# 10 - YOLOv8 TensorRT C++

Goal: build the main end-to-end C++ deployment artifact.

Topics:

- OpenCV preprocessing
- TensorRT runtime
- CUDA buffers
- YOLO decode
- NMS
- Coordinate scaling
- Latency breakdown

Acceptance criteria:

- The program accepts image and engine paths.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.
