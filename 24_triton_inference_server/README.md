# 24 - Triton Inference Server

## Purpose

This server-inference elective packages lesson 17's dynamic TensorRT plan in a Triton model
repository, enables dynamic batching, sends real preprocessed YOLOv8 input, and measures client
latency under concurrency.

### Where Commands Run

This lesson uses two containers plus the host shell. Every command block below is labeled with one
of these environments:

- **Host shell**: your normal terminal on the Linux host. Docker commands always run here; course
  containers are managed from the host, never from inside another container.
- **Dev container** (`learn-tensorrt`): the persistent course development container from course 00.
  Enter it with `docker exec -it learn-tensorrt bash` and run commands from
  `/workspace/Learn-TensorRT`. All Python tooling in this lesson runs here.
- **Triton container** (`nvcr.io/nvidia/tritonserver:25.11-py3`): a temporary `docker run` server
  process started from the host shell. You do not open a shell inside it; you interact with it
  over HTTP.

Both containers use `--network host`, so the client in the dev container reaches the Triton server
at `localhost:8000`. The dev container must have been created with `--network host` as documented
in `00_environment_check/agent_env_setup.md`; containers created before that change still use the
bridge network and must be recreated. As a fallback, pass the host's LAN IP to the client, for
example `--url 192.168.1.10:8000`.

## Prerequisites

- Course 00's `learn-tensorrt` dev container, created with `--network host`.
- Build lesson 17's dynamic-batch TensorRT engine **in the dev container**:

  ```bash
  # Dev container (/workspace/Learn-TensorRT)
  ./17_dynamic_batching/build_dynamic_engine.sh
  ```

- Docker and NVIDIA Container Toolkit on the host for the Triton server container.

## Deliverables

- `prepare_model_repository.py` reproducible repository generator
- Validated client and load-test tooling
- Metrics utilities, configuration checks, and focused tests

## Setup

The reference server is `nvcr.io/nvidia/tritonserver:25.11-py3`. Build lesson 17's plan with
TensorRT 10.14 in the pinned `nvcr.io/nvidia/pytorch:25.11-py3` development environment, then load
it with this pinned server image. TensorRT plans are environment-specific; rebuild the plan rather
than reusing an artifact from another TensorRT, CUDA, or GPU environment.

Client dependencies are separate from global Python. Install them **in the dev container**:

```bash
# Dev container (/workspace/Learn-TensorRT)
python3 -m pip install --target 24_triton_inference_server/.deps \
  -r 24_triton_inference_server/requirements-client.txt
```

## Run

### 1. Prepare the Model Repository

Run the model-repository generator **in the dev container**. It validates `config.pbtxt` and copies lesson 17's
engine into the versioned repository layout:

```bash
# Dev container (/workspace/Learn-TensorRT)
python3 24_triton_inference_server/prepare_model_repository.py
```

Example output (local run):

```text
copied 17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine -> 24_triton_inference_server/model_repository/yolov8/1/model.plan
```

### 2. Start the Triton Server

Run the server **from the host shell** at the repository root. This is Docker container management,
which is host work; the server itself runs in the temporary Triton container in the foreground:

Start the Triton server from the host shell; keep this foreground process running while testing:

```bash
# Host shell (repository root)
docker run --rm --gpus all --network host \
  -v "$PWD/24_triton_inference_server/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.11-py3 \
tritonserver --model-repository=/models --disable-auto-complete-config
```

Example output (local run, key lines):

```text
| Model  | Version | Status |
| yolov8 | 1       | READY  |
Started HTTPService at 0.0.0.0:8000
Started Metrics Service at 0.0.0.0:8002
```

Keep this terminal open; the server log shows model loading and batching decisions. Stop it with
`Ctrl-C` when the lesson is done.

### 3. Verify Server Health

In a second terminal, confirm the health endpoints **from the host shell or the dev container**
(both reach the host-networked server):

Confirm both server health endpoints before sending inference requests:

```bash
# Host shell or dev container
curl -i localhost:8000/v2/health/ready
curl -i localhost:8000/v2/models/yolov8/ready
```

Example output (successful readiness checks):

```text
HTTP/1.1 200 OK
Content-Length: 0
Content-Type: text/plain

HTTP/1.1 200 OK
Content-Length: 0
Content-Type: text/plain
```

The readiness endpoints intentionally return an empty response body. The `HTTP/1.1 200 OK`
status confirms that the server and the `yolov8` model are ready.

The model configuration declares max batch 4, preferred batches 2 and 4, and a 5 ms maximum queue
delay. Confirm the model is ready before benchmarking.

### 4. Run the Client and Load Test

Run the client **in the dev container**. Leave the server running from step 2:

```bash
# Dev container (/workspace/Learn-TensorRT)
PYTHONNOUSERSITE=1 PYTHONPATH=24_triton_inference_server/.deps \
python3 24_triton_inference_server/client.py --concurrency 1 --requests 100

```

Example output (local run, concurrency 1):

<details><summary>Example output (partial)</summary>

```text
"server_version": "2.63.0"
"concurrency": 1,
"requests": 100,
"p50": 10.752620999483042,
"p99": 20.533939999950235,
"throughput_requests_per_second": 78.74079125066447
```
</details>

Run the higher-concurrency variant to observe dynamic batching under load:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=24_triton_inference_server/.deps \
python3 24_triton_inference_server/client.py --concurrency 4 --requests 100
```

Each client result records the RTX GPU, compute capability, driver, memory, and Triton server
version. Compare client P50/P90/P99 and throughput with server Prometheus metrics at port 8002,
especially request queue time, compute-infer time, execution count, and inference count. Dynamic
batching helps when concurrent requests arrive within the queue-delay window and the larger GPU
batch saves more time than it waits. It hurts latency-sensitive traffic when the queue delay
dominates the batch efficiency gain.

## Outputs

- The generated model repository, client JSON, and Prometheus captures remain under ignored local output paths.
- Static checks are not server-runtime evidence.

## Tests

### Verification

Build lesson 17's engine before preparing the repository. Static tests do not replace a server run.
Both commands run **in the dev container**:

```bash
# Dev container (/workspace/Learn-TensorRT)
python3 -m unittest discover -s 24_triton_inference_server/tests -v
python3 24_triton_inference_server/prepare_model_repository.py
```

These checks validate only the model repository and metrics logic. Acceptance requires the pinned
Triton container, GPU health checks, successful model loading, client results at concurrency 1 and
4, and saved Prometheus evidence.

## Checkpoints

1. Prepare a reproducible Triton TensorRT model repository and send validated client requests.
2. Measure concurrency, dynamic batching, queue delay, compute time, throughput, and client latency.
3. Explain how model instances and batching configuration affect GPU utilization and latency targets.

## Appendix: Triton Inference Server Basics

### What Triton Is

NVIDIA Triton Inference Server is an open-source model-serving system. Instead of embedding
TensorRT calls in each application, you place serialized models in a **model repository** directory
and Triton loads, schedules, and serves them behind stable network endpoints. Applications become
thin clients that send tensors over HTTP or gRPC. Triton provides:

- Multiple **backends** (TensorRT plan, ONNX Runtime, PyTorch, TensorFlow, Python, and more),
  selected per model by the `platform` or `backend` field.
- **Dynamic batching**: the server merges concurrent single requests into larger GPU batches.
- **Concurrent model instances**: several copies of a model on one or more GPUs.
- **Versioned models**: several versions of one model side by side for rollout and rollback.
- Standard **health endpoints** and **Prometheus metrics** for operations.

### Network Endpoints

With `--network host`, the default ports are host ports:

| Port | Protocol | Purpose |
| ---- | -------- | ------- |
| 8000 | HTTP/REST | Inference, health, and model metadata |
| 8001 | gRPC | Inference, health, and model metadata |
| 8002 | HTTP | Prometheus metrics |

Useful HTTP endpoints beyond inference:

- `GET /v2/health/live` - server process is up.
- `GET /v2/health/ready` - server is ready to serve.
- `GET /v2/models/<name>/ready` - one specific model finished loading.
- `GET /v2/models/<name>` - model metadata and its effective configuration.
- `GET :8002/metrics` - Prometheus text metrics (queue time, compute time, counts).

### Model Repository Layout

Triton discovers models by directory structure. Each model needs a directory named after the
model, a numeric version subdirectory holding the model file, and a `config.pbtxt`:

```text
model_repository/
└── yolov8/
    ├── config.pbtxt        # Model configuration (this lesson)
    └── 1/                  # Version 1
        └── model.plan      # TensorRT plan (copied from lesson 17)
```

The version directory name must be a positive integer. The plan filename for the `tensorrt_plan`
platform defaults to `model.plan`. `prepare_model_repository.py` builds this layout
deterministically instead of relying on manual copies.

The server flag `--disable-auto-complete-config` forbids Triton from inferring a missing or
partial configuration from the model file. The lesson uses it deliberately: the committed
`config.pbtxt` is the single source of truth, and a malformed or incomplete configuration fails at
startup instead of silently serving with guessed shapes.

### `config.pbtxt` Explained

This lesson's `model_repository/yolov8/config.pbtxt`:

```text
name: "yolov8"
platform: "tensorrt_plan"
max_batch_size: 4

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 2, 4 ]
  max_queue_delay_microseconds: 5000
  preserve_ordering: true
}

instance_group [
  { kind: KIND_GPU count: 1 }
]

model_transaction_policy { decoupled: false }
```

Field by field:

- `name`: must match the model's directory name; it becomes the name in every request URL.
- `platform: "tensorrt_plan"`: the model file is a serialized TensorRT engine (`model.plan`), so
  Triton uses its TensorRT backend directly with no further conversion.
- `max_batch_size: 4`: requests may be batched up to batch 4. The `dims` of inputs and outputs are
  then **per-sample** shapes; the batch dimension is implicit. This must not exceed the maximum
  batch the TensorRT engine was built for in lesson 17.
- `input` / `output`: tensor names must match the engine's binding names exactly (`images`,
  `output0` for the Ultralytics YOLOv8 export). `dims: [3, 640, 640]` is one CHW image;
  `dims: [84, 8400]` is the raw YOLOv8 prediction grid for one image (4 box coordinates + 80 class
  scores, 8400 candidate boxes). A batched response has shape `[batch, 84, 8400]`.
- `dynamic_batching`: enables the server-side batcher.
  - `preferred_batch_size: [2, 4]` tells the scheduler to prefer forming batches of 2 or 4, which
    match lesson 17's optimized engine profiles, instead of any size up to 4.
  - `max_queue_delay_microseconds: 5000` caps how long an early request may wait for batch partners:
    after 5 ms the batch executes even if it is below the preferred size. This is the direct
    latency-versus-throughput knob.
  - `preserve_ordering: true` returns responses in request arrival order, which keeps the client's
    latency accounting straightforward.
- `instance_group`: `KIND_GPU count: 1` loads one copy of the model on the default GPU. Raising
  `count` runs parallel instances to overlap execution, at the cost of extra GPU memory per
  instance.
- `model_transaction_policy { decoupled: false }`: each request produces exactly one response
  (standard request/response). Decoupled models, such as streaming generators, are out of scope
  here.

### How a Request Flows

1. The client sends a `[1, 3, 640, 640]` FP32 tensor over HTTP to `/v2/models/yolov8/infer`.
2. The dynamic batcher holds the request up to 5 ms, merging concurrent requests into one batch
   (preferring size 2 or 4).
3. The TensorRT backend runs the engine once for the whole batch on the GPU instance.
4. Responses are split back per request, preserving arrival order, and returned as
   `[batch, 84, 8400]` tensors.

The client-side P50/P90/P99 therefore includes network time, queue delay, and compute. The
Prometheus metrics at port 8002 separate server-side queue time from compute-infer time, which is
how you tell whether latency comes from batching wait or from the GPU work itself.
