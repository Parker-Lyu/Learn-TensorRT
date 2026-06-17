# 20a - Jetson Orin Xavier DLA Deployment

Goal: understand Jetson Orin/Xavier TensorRT deployment, cross compilation, and DLA constraints.

Scope:

- This is an edge-deployment extension.
- The command plan can be prepared on x86 first, then fully verified when a Jetson target is available.

Why it matters:

- Many CV deployment roles target embedded NVIDIA devices instead of desktop GPUs.
- Jetson deployment requires careful version control across JetPack, CUDA, TensorRT, cuDNN, DeepStream, and the kernel driver stack.

Topics:

- JetPack, CUDA, TensorRT, cuDNN, and DeepStream compatibility
- Native Jetson build versus x86-to-aarch64 cross compilation
- CMake toolchain file for aarch64 targets
- Container versus bare-metal deployment
- DLA-supported layer constraints
- `trtexec --useDLACore`
- GPU fallback behavior
- FP16 and INT8 on DLA
- Power modes, clocks, thermals, and memory bandwidth
- Orin/Xavier benchmark checklist

Acceptance criteria:

- The target hardware, JetPack version, TensorRT version, power mode, and clocks are recorded.
- The lesson documents both native-build and cross-compilation paths.
- A YOLO TensorRT engine is attempted with DLA, and unsupported layers or GPU fallback are recorded.
- Latency, throughput, memory, and power-mode notes are compared with the desktop GPU baseline when hardware is available.
- If Jetson hardware is not available, the exact future validation commands and expected evidence are still documented.
