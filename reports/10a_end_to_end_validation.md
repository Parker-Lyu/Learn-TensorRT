# 10a - End-to-End Validation Report

## Scope

This checkpoint records reproducible evidence for one controlled YOLOv8n input before performance
optimization. It establishes single-input numerical alignment and a working C++ end-to-end path; it
does not establish dataset-level detection accuracy, optimized performance, or service reliability.

## Environment and Dependencies

| Item | Evidence |
| --- | --- |
| Reference environment | TensorRT development container from lesson 00 |
| Environment check | `10a_end_to_end_validation_report/outputs/environment_check.log` |
| C++ focused tests | Passed; `10a_end_to_end_validation_report/outputs/cpp_tests.log` |

<details>
<summary>Saved environment-check output</summary>

```text
========== System ==========

-- OS release --
PRETTY_NAME="Ubuntu 22.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.3 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
[OK] OS release

-- Kernel --
Linux parker-ASUS 6.8.0-124-generic #124~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 21:05:19 UTC  x86_64 x86_64 x86_64 GNU/Linux
[OK] Kernel

-- Working directory --
/workspace/Projects/Learn-TensorRT
[OK] Working directory

========== Required commands ==========
[OK] nvidia-smi: /usr/bin/nvidia-smi
[OK] nvcc: /usr/local/cuda/bin/nvcc
[OK] trtexec: /opt/tensorrt/bin/trtexec
[OK] cmake: /usr/local/bin/cmake
[OK] g++: /usr/bin/g++
[OK] python3: /usr/bin/python3

========== GPU ==========

-- nvidia-smi --
Tue Jul 14 09:22:33 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.04             Driver Version: 580.159.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2060        On  |   00000000:01:00.0 Off |                  N/A |
| N/A   48C    P8              1W /   90W |       6MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
[OK] nvidia-smi

========== CUDA ==========

-- nvcc --version --
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
[OK] nvcc --version

========== TensorRT ==========

-- trtexec help/version output --
&&&& RUNNING TensorRT.trtexec [TensorRT v8601] # trtexec --help
=== Model Options ===
  --uff=<file>                UFF model
  --onnx=<file>               ONNX model
  --model=<file>              Caffe model (default = no model, random weights used)
  --deploy=<file>             Caffe prototxt file
  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output is required for UFF and Caffe
  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified multiple times; at least one is required for UFF models
  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use X,Y,Z=H,W,C order in --uffInput)

=== Build Options ===
  --maxBatch                         Set max batch size and build an implicit batch engine (default = same size as --batch)
                                     This option should not be used when the input model is ONNX or when dynamic shapes are provided.
  --minShapes=spec                   Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec                   Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec                   Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec              Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec              Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec              Calibrate with dynamic shapes using a profile with the max shapes provided
                                     Note: All three of min, opt and max shapes must be supplied.
[OK] trtexec help/version output

-- TensorRT C++ libraries --
[OK] TensorRT C++ libraries

-- TensorRT Python import --
tensorrt: 8.6.1
[OK] TensorRT Python import

========== C++ build tools ==========

-- cmake --version --
cmake version 3.24.0

CMake suite maintained and supported by Kitware (kitware.com/cmake).
[OK] cmake --version

-- g++ --version --
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[OK] g++ --version

========== Python ==========

-- python3 --version --
Python 3.10.12
[OK] python3 --version

-- Required Python package imports --
ultralytics: installed
onnx: installed
onnxruntime: installed
[OK] Required Python package imports

-- Optional Python package availability --
cv2: installed
numpy: installed
torch: installed
tensorrt: installed
[OK] Optional Python package availability

========== OpenCV ==========

-- pkg-config opencv4 --
4.5.4
[OK] pkg-config opencv4

-- Python OpenCV import --
cv2: 4.13.0
[OK] Python OpenCV import

========== Project mount ==========

-- Repository is writable 
... (truncated; see saved log)
```

</details>

## Controlled Artifacts

| Artifact | Path |
| --- | --- |
| Image | `assets/img2.jpeg` |
| ONNX model | `05_torch_to_onnx/outputs/yolov8n.onnx` |
| TensorRT engine | `06_trtexec_engine/outputs/yolov8n_static_fp32.engine` |
| NCHW float32 input | `05_torch_to_onnx/outputs/input_nchw_float32.npy` |

PyTorch, ONNX Runtime, and TensorRT compare the same saved NCHW tensor. The C++ program uses the
same source image and serialized engine.

## Functional Validation

### PyTorch and ONNX Runtime raw output

| Metric | Value |
| --- | ---: |
| Shape | [1, 84, 8400] |
| Max absolute error | 0.0016174316 |
| Mean absolute error | 1.4336949e-06 |
| P99 absolute error | 4.5776367e-05 |
| Tolerance | rtol=0.001, atol=0.001 |
| Allclose | pass |

### ONNX Runtime and TensorRT raw output

| Metric | Value |
| --- | ---: |
| Shape match | pass |
| Max absolute error | 0.0015258789 |
| Mean absolute error | 1.0054538e-06 |
| P99 absolute error | 3.0517578e-05 |
| Tolerance | rtol=0.001, atol=0.001 |
| Allclose | pass |

### C++ end-to-end smoke test

- Focused preprocessing/postprocessing tests passed.
- C++ inference completed on the controlled image with **6** detections.
- Machine-readable result: `10a_end_to_end_validation_report/outputs/cpp/detections.json`.

## Pipeline Architecture and Ownership

```text
cv::Mat image
  -> preprocess_image (letterbox, BGR->RGB, NCHW float32)
  -> TensorRtRunner
       owns runtime -> engine -> execution context
       owns input/output device buffers
       owns a reusable CUDA stream and timing events
  -> decode_yolov8_output (decode, class-aware NMS, coordinate mapping)
  -> draw_detections and JSON/image reporting
```

`main` owns orchestration and `TensorRtRunner`. The runner uses a private implementation and RAII
wrappers for CUDA allocations, a reusable stream, and reusable events. It synchronizes the D2H
completion event before decoding host output. Lesson 10 supports one static float32 input and one
float32 output.

## Mean Per-stage Latency Baseline

| Stage | Milliseconds |
| --- | ---: |
| preprocess | 8.0078945 |
| h2d | 0.9746704 |
| enqueue_host | 0.84166693 |
| gpu_compute | 4.6162 |
| d2h | 0.60826944 |
| postprocess | 3.6708306 |
| total | 17.924775 |

Each value is the arithmetic mean of 100 measured samples after
10 warmup iteration(s). The raw samples remain in the C++ JSON result.
Engine deserialization is not included in `total`; this is neither a throughput claim nor an
optimized benchmark. Lesson 11 adds timeline diagnosis.

## What This Evidence Proves

- The documented container workflow can build, test, and run the C++ pipeline.
- PyTorch and ONNX Runtime meet the recorded tolerance for one controlled input.
- ONNX Runtime and TensorRT meet the recorded tolerance for that same input.
- The C++ pipeline produces an annotated image and machine-readable detections.

## What It Does Not Prove Yet

- Dataset-level mAP or detection-quality regression.
- FP16/INT8 acceptance or multi-image accuracy.
- Optimized latency, throughput, concurrency, video behavior, or long-running stability.
- Serialized-engine portability across GPUs, drivers, CUDA versions, or TensorRT versions.

## English Project Summary

I exported YOLOv8n from PyTorch to ONNX, built a TensorRT engine, and implemented an end-to-end C++
inference application. For one controlled image, I compared PyTorch and ONNX Runtime raw outputs,
then compared ONNX Runtime and TensorRT with the same NCHW tensor. The C++ program performs
letterbox preprocessing, TensorRT inference, YOLOv8 decoding, class-aware NMS, coordinate mapping,
visualization, and JSON reporting. This checkpoint proves reproducibility and single-input alignment,
but it is not a dataset-level accuracy or performance certification.

## English Walkthrough (3–5 Minutes)

1. State the deployment goal and identify the controlled image, ONNX model, and TensorRT engine.
2. Explain why raw-output alignment precedes decode and NMS.
3. Walk through the C++ pipeline and ownership boundary inside `TensorRtRunner`.
4. Show the focused tests, annotated image, JSON result, and latency baseline.
5. Close with limitations and the next steps: Nsight profiling in lesson 11, then multi-image
   FP32/FP16/INT8 validation in lesson 12.
