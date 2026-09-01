# 30 - LLM Inference Introduction

## Purpose

This awareness elective runs a deterministic two-layer autoregressive Transformer locally. It is
small enough to inspect, but still performs tokenization, causal attention, prefill, token-by-token
decode, and real KV-cache growth.

Prefill accepts the complete `[batch, sequence]` prompt, computes all prompt positions in parallel
under a causal mask, and fills each layer's KV cache in one call. Decode then processes one new token
per sequence at each step. This distinction mirrors the execution shape of optimized LLM runtimes;
production servers may split long prompts into chunks, but do not treat prefill as repeated decode.

## Prerequisites

- Use the pinned development container or another Python environment with the documented NumPy dependency.
- No pretrained model download or GPU is required.

## Deliverables

- Deterministic inspectable autoregressive Transformer
- Controlled benchmark and report generator
- Correctness tests and generated LLM inference report

## Run

```bash
python3 30_llm_inference_intro/benchmark.py
```

<details><summary>Example output (local run, partial)</summary>

```text
"backend": "NumPy CPU autoregressive"
"weight_memory_mib": 0.375
"input_length": 16, "batch": 1, "ttft_ms": 0.164
"input_length": 64, "batch": 4, "total_tokens_per_second": 49306.9
"peak_gpu_memory_mib": 0.0
```
</details>

Generate the Markdown summary from the benchmark JSON:

```bash
python3 30_llm_inference_intro/generate_report.py
```

Example output (partial):

```text
# Tiny Local LLM Inference Benchmark
Peak GPU memory: 0 MiB because this controlled example uses CPU NumPy.
```

The fixed revision is derived from the committed architecture and seed. The benchmark holds output
length constant, compares input lengths 16/64 and batches 1/4, performs warmup and repeated runs,
and reports TTFT, TPOT, prefill/decode/total throughput, weight memory, estimated KV memory, peak
host RSS, and zero GPU memory for this CPU backend. The evidence records the CPU model, logical CPU
count, Python version, and NumPy version beside those measurements.
When the requested output length is one, no decode step exists, so TPOT and decode throughput are
reported as `N/A` rather than as zero.

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
PYTHONPATH=30_llm_inference_intro \
python3 -m unittest discover -s 30_llm_inference_intro/tests -v
```

## Checkpoints

1. Trace tokenization, causal attention, prefill, decode, and KV-cache growth in a small autoregressive model.
2. Measure TTFT, time per output token, throughput, and memory across controlled input-length and batch experiments.
3. Explain how LLM inference bottlenecks and batching semantics differ from YOLO inference.
