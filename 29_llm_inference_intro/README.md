# 29 - LLM Inference Introduction

## Purpose

This awareness elective runs a deterministic two-layer autoregressive Transformer locally. It is
small enough to inspect, but still performs tokenization, causal attention, prefill, token-by-token
decode, and real KV-cache growth.

## Prerequisites

- Use the pinned development container or another Python environment with the documented NumPy dependency.
- No pretrained model download or GPU is required.

## Deliverables

- Deterministic inspectable autoregressive Transformer
- Controlled benchmark and report generator
- Correctness tests and generated LLM inference report

## Run

```bash
python3 29_llm_inference_intro/benchmark.py
python3 29_llm_inference_intro/generate_report.py
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

## Outputs

- `outputs/llm_benchmark.json` contains machine-readable measurements.
- `outputs/llm_benchmark.md` is the generated, environment-specific summary.

## Tests

Run the Python tests from the repository root:

```bash
PYTHONPATH=29_llm_inference_intro \
python3 -m unittest discover -s 29_llm_inference_intro/tests -v
```

## Checkpoints

1. Trace tokenization, causal attention, prefill, decode, and KV-cache growth in a small autoregressive model.
2. Measure TTFT, time per output token, throughput, and memory across controlled input-length and batch experiments.
3. Explain how LLM inference bottlenecks and batching semantics differ from YOLO inference.
