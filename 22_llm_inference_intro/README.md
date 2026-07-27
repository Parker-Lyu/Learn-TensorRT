# 22 - LLM Inference Introduction

This awareness elective runs a deterministic two-layer autoregressive Transformer locally. It is
small enough to inspect, but still performs tokenization, causal attention, prefill, token-by-token
decode, and real KV-cache growth.

```bash
cd 22_llm_inference_intro
python3 -m unittest discover -s tests -v
python3 benchmark.py
python3 generate_report.py
```

The fixed revision is derived from the committed architecture and seed. The benchmark holds output
length constant, compares input lengths 16/64 and batches 1/4, performs warmup and repeated runs,
and reports TTFT, TPOT, prefill/decode/total throughput, weight memory, estimated KV memory, peak
host RSS, and zero GPU memory for this CPU backend. The evidence records the CPU model, logical CPU
count, Python version, and NumPy version beside those measurements.

This is not a pretrained language model or serving-stack benchmark. It isolates inference mechanics.
Mention TensorRT-LLM for optimized NVIDIA deployment, vLLM for paged-attention server throughput,
llama.cpp for portable quantized local inference, and OpenVINO GenAI for Intel-oriented deployment.
FP16/INT8/INT4 reduce memory differently; weight-only quantization shrinks weights but does not by
itself shrink every activation or KV-cache tensor.
