# 22 - LLM Inference Intro

Goal: understand the minimum LLM inference concepts needed for modern deployment roles.

This is an extension chapter. The main repository line remains YOLO and TensorRT C++ deployment.

Topics:

- Tokenization
- Prefill and decode
- KV cache
- Batch size and sequence length
- First-token latency
- Decode throughput
- FP16, INT8, INT4, and weight-only quantization
- ONNX Runtime, OpenVINO GenAI, TensorRT-LLM, vLLM, and llama.cpp at a high level
- CPU/GPU memory limits

Acceptance criteria:

- You can run or explain one minimal local LLM inference example.
- You can explain why LLM inference bottlenecks differ from YOLO inference.
- You can explain KV cache and memory-bandwidth pressure during decode.
- You know when to mention TensorRT-LLM, OpenVINO GenAI, vLLM, or llama.cpp in interviews.
