# 10 - YOLOv8 TensorRT C++

Goal: build the main end-to-end C++ deployment artifact.

Topics:

- OpenCV preprocessing
- TensorRT runtime
- CUDA buffers
- YOLO decode
- NMS
- Coordinate scaling
- Visualization
- CLI arguments
- Latency breakdown
- Library targets for reusable preprocessing, inference, and postprocessing components
- Focused tests for preprocessing and postprocessing edge cases

Acceptance criteria:

- The program accepts image and engine paths.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.
- Reusable preprocessing, inference, and postprocessing code is not trapped inside `main`.
- Focused tests cover representative invalid input and boundary cases.
