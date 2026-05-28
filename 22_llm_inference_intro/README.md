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
- TensorRT-LLM, OpenVINO GenAI, vLLM, and llama.cpp at a high level

Acceptance criteria:

- You can run or explain one minimal local LLM inference example.
- You can explain why LLM inference bottlenecks differ from YOLO inference.
- You can explain KV cache and memory-bandwidth pressure during decode.
