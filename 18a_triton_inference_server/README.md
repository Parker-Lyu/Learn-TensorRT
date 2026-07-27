# 18a - Triton Inference Server

This server-inference elective packages lesson 14's dynamic TensorRT plan in a Triton model
repository, enables dynamic batching, sends real preprocessed YOLOv8 input, and measures client
latency under concurrency.

## Environment Boundary

The reference server is `nvcr.io/nvidia/tritonserver:25.11-py3`. Build lesson 14's plan with
TensorRT 10.14 in the pinned `nvcr.io/nvidia/pytorch:25.11-py3` development environment, then load
it with this pinned server image. TensorRT plans are environment-specific; rebuild the plan rather
than reusing an artifact from another TensorRT, CUDA, or GPU environment.

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
  nvcr.io/nvidia/tritonserver:25.11-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config
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

Each client result records the RTX GPU, compute capability, driver, memory, and Triton server
version. Compare client P50/P90/P99 and throughput with server Prometheus metrics at port 8002,
especially
request queue time, compute-infer time, execution count, and inference count. Dynamic batching helps
when concurrent requests arrive within the queue-delay window and the larger GPU batch saves more
time than it waits. It hurts latency-sensitive traffic when the queue delay dominates the batch
efficiency gain.

## Verification

Build lesson 14's engine before preparing the repository. Static tests do not replace a server run.

## Static Checks

```bash
python3 -m unittest discover -s 18a_triton_inference_server/tests -v
python3 18a_triton_inference_server/prepare_model_repository.py
```

These checks validate only the model repository and metrics logic. Acceptance requires the pinned
Triton container, GPU health checks, successful model loading, client results at concurrency 1 and
4, and saved Prometheus evidence.
