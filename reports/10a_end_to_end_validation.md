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
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=noble
LOGO=ubuntu-logo
[OK] OS release

-- Kernel --
Linux bd2fc3c23a5b 7.0.0-28-generic #28~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Jul  1 15:50:57 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
[OK] Kernel

-- Working directory --
/workspace/Learn-TensorRT
[OK] Working directory

-- NVIDIA PyTorch container release --
NVIDIA_PYTORCH_VERSION=25.11
CUDA_VERSION=13.0.2.006
[OK] NVIDIA PyTorch container release

========== Required commands ==========
[OK] nvidia-smi: /usr/bin/nvidia-smi
[OK] nvcc: /usr/local/cuda/bin/nvcc
[OK] trtexec: /opt/tensorrt/bin/trtexec
[OK] cmake: /usr/local/bin/cmake
[OK] g++: /usr/bin/g++
[OK] python3: /usr/bin/python3

========== GPU ==========

-- nvidia-smi --
Thu Jul 30 10:00:26 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.84                 Driver Version: 595.84         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:01:00.0  On |                  Off |
|  0%   40C    P8             18W /  500W |     872MiB /  24564MiB |      4%      Default |
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
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
[OK] nvcc --version

========== TensorRT ==========

-- trtexec version --
&&&& RUNNING TensorRT.trtexec [TensorRT v101401] [b48] # trtexec --version
[07/30/2026-10:00:26] [I] TF32 is enabled by default. Add --noTF32 flag to further improve accuracy with some performance cost.
=== Model Options ===
  --onnx=<file>               ONNX model

=== Build Options ===
  --minShapes=spec                   Build with dynamic shapes using a profile with the min shapes provided
  --optShapes=spec                   Build with dynamic shapes using a profile with the opt shapes provided
  --maxShapes=spec                   Build with dynamic shapes using a profile with the max shapes provided
  --minShapesCalib=spec              Calibrate with dynamic shapes using a profile with the min shapes provided
  --optShapesCalib=spec              Calibrate with dynamic shapes using a profile with the opt shapes provided
  --maxShapesCalib=spec              Calibrate with dynamic shapes using a profile with the max shapes provided
                                     Note: All three of min, opt and max shapes must be supplied.
                                           However, if only opt shapes is supplied then it will be expanded so
                                           that min shapes and max shapes are set to the same values as opt shapes.
                                           Input names can be wrapped with escaped single quotes (ex: 'Input:0').
                                     Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128
                                     For scalars (0-D shapes), use input0:scalar or simply input0: with nothing after the colon.
                                     Each input shape is supplied as a key-value pair where key is the input name and
                                     value is the dimensions (including the batch dimension) to be used for that input.
[OK] trtexec version

-- TensorRT C++ libraries --
[OK] TensorRT C++ libraries

-- TensorRT Python import and baseline version --
tensorrt: 10.14.1.48
[OK] TensorRT Python import and baseline version

========== C++ build tools ==========

-- cmake --version --
cmake version 3.31.6

CMake suite maintained and supported by Kitware (kitware.com/cmake).
[OK] cmake --version

-- g++ --version --
g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[OK] g++ --version

========== Python ==========

-- python3 --version --
Python 3.12.3
[OK] python3 --version

-- Required Python package imports --
torch: 2.10.0a0+b558c986e8.nv25.11
modelopt: 0.37.0
ultralytics: 8.3.225
onnx: 1.21.0
onnxruntime: 1.23.2
onnxslim: 0.1.94
onnxsim: v0.6.5
[OK] Required Python package imports

-- Optional Python package avai
... (truncated; see saved log)
```

</details>

## Controlled Artifacts

| Artifact | Path |
| --- | --- |
| Image | `assets/img.jpeg` |
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
| Max absolute error | 0.0015563965 |
| Mean absolute error | 1.4327373e-06 |
| P99 absolute error | 4.5776367e-05 |
| Tolerance | rtol=0.001, atol=0.001 |
| Allclose | pass |

### ONNX Runtime and TensorRT raw output

| Metric | Value |
| --- | ---: |
| Shape match | pass |
| Max absolute error | 0.0014648438 |
| Mean absolute error | 1.0059709e-06 |
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
       owns input/output pinned-host and device buffers
       owns a reusable CUDA stream and timing events
  -> decode_yolov8_output (decode, class-aware NMS, coordinate mapping)
  -> draw_detections and JSON/image reporting
```

`main` owns orchestration and `TensorRtRunner`. The runner uses a private implementation and RAII
wrappers for pinned-host/device CUDA allocations, a reusable stream, and reusable events. It
synchronizes the D2H completion event before decoding host output. Lesson 10 supports one static float32 input and one
float32 output.

## Mean Per-stage Latency Baseline

| Stage | Milliseconds |
| --- | ---: |
| preprocess | 0.71552421 |
| h2d | 0.25591168 |
| enqueue_host | 0.25150033 |
| gpu_compute | 0.85063968 |
| d2h | 0.14707744 |
| postprocess | 0.41205541 |
| total | 2.7624556 |

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
