# 18a - Triton Inference Server

This server-inference elective packages lesson 14's dynamic TensorRT plan in a Triton model
repository, enables dynamic batching, sends real preprocessed YOLOv8 input, and measures client
latency under concurrency.

## Environment Boundary

The reference server is `nvcr.io/nvidia/tritonserver:23.10-py3`, aligned with the course's 23.10
TensorRT development container. The local machine currently does not provide Docker or
`tritonserver`, so server execution must happen on a Docker-capable NVIDIA host. Do not rebuild the
plan in a different TensorRT version and assume it remains binary-compatible.

Client dependencies are separate from global Python:

```bash
python3 -m pip install --target 18a_triton_inference_server/.deps \
  -r 18a_triton_inference_server/requirements-client.txt
```

## Prepare and Start

```bash
python3 18a_triton_inference_server/prepare_model_repository.py

docker run --rm --gpus all --network host \
  -v "$PWD/18a_triton_inference_server/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models --strict-model-config=true
```

Confirm `/v2/health/ready` and the model-ready endpoint before benchmarking. The model configuration
declares max batch 4, preferred batches 2 and 4, and a 5 ms maximum queue delay.

## Client and Load Test

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=18a_triton_inference_server/.deps \
python3 18a_triton_inference_server/client.py --concurrency 1 --requests 100

PYTHONNOUSERSITE=1 PYTHONPATH=18a_triton_inference_server/.deps \
python3 18a_triton_inference_server/client.py --concurrency 4 --requests 100
```

Compare client P50/P90/P99 and throughput with server Prometheus metrics at port 8002, especially
request queue time, compute-infer time, execution count, and inference count. Dynamic batching helps
when concurrent requests arrive within the queue-delay window and the larger GPU batch saves more
time than it waits. It hurts latency-sensitive traffic when the queue delay dominates the batch
efficiency gain.

## Verification Available Without Triton

```bash
python3 -m unittest discover -s 18a_triton_inference_server/tests -v
python3 18a_triton_inference_server/prepare_model_repository.py
```

These checks validate the model repository and metrics logic but do not claim the server loaded the
engine. A real acceptance run requires the pinned Triton container, GPU, health checks, client
results at two concurrency levels, and saved Prometheus evidence.
