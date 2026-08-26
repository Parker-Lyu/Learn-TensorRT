#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "outputs/llm_benchmark.json").read_text())


def format_metric(value, precision):
    return "N/A" if value is None else f"{value:.{precision}f}"


rows = "\n".join(
    f"| {r['input_length']} | {r['batch']} | {r['ttft_ms']:.3f} | "
    f"{format_metric(r['time_per_output_token_ms'], 3)} | {r['prefill_tokens_per_second']:.1f} | "
    f"{format_metric(r['decode_tokens_per_second'], 1)} | {r['total_tokens_per_second']:.1f} | "
    f"{r['estimated_kv_cache_mib']:.3f} |" for r in data["results"])
text = f"""# Tiny Local LLM Inference Benchmark

Model revision: `{data['model']['revision']}`; tokenizer: {data['tokenizer']}; backend:
{data['backend']}; CPU: {data['hardware']['cpu_model']}
({data['hardware']['logical_cpu_count']} logical CPUs); Python {data['software']['python']}; NumPy
{data['software']['numpy']}; weights: {data['model']['weight_format']} ({data['model']['weight_memory_mib']:.3f} MiB).

| Input tokens | Batch | TTFT ms | TPOT ms | Prefill tok/s | Decode tok/s | Total tok/s | KV MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Peak host RSS: {data['hardware']['peak_host_rss_mib']:.2f} MiB. Peak GPU memory: 0 MiB because this
controlled example uses CPU NumPy. Output length is fixed at {data['methodology']['fixed_output_length']}.

YOLO executes one mostly fixed compute graph per image. Autoregressive LLM decode repeatedly reads
model weights and a growing KV cache for one new token, so decode is often memory-bandwidth bound.
Longer prompts increase prefill work and KV memory; larger batches improve aggregate throughput but
increase working-set size and per-request latency.
"""
(ROOT / "outputs/llm_benchmark.md").write_text(text)
print(text)
