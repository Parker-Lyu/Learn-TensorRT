# 17 - CUDA Preprocess And NPP

Goal: move preprocessing hotspots from CPU OpenCV to GPU code when profiling proves it is useful.

Topics:

- Simple CUDA kernels
- BGR to RGB conversion
- Normalization
- HWC to CHW conversion
- Optional resize or letterbox kernel
- NVIDIA NPP overview
- CPU OpenCV versus CUDA preprocessing benchmark

Acceptance criteria:

- At least one preprocessing step runs on GPU.
- The GPU result is checked against the OpenCV reference.
- A benchmark compares CPU preprocessing and GPU or NPP preprocessing.
