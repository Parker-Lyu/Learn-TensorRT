# 17 - TensorRT Dynamic Batching

## Purpose

This lesson runs batch sizes 1, 2, and 4 through one TensorRT engine. It makes runtime shapes, NCHW
sample offsets, output offsets, and the latency/throughput trade-off explicit.

## Prerequisites

Complete the lesson 05 dynamic export and validation, then lesson 06 FP16 ONNX preparation:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
python3 06_trtexec_engine/prepare_fp16_onnx.py --models dynamic
```

The engine-build script consumes only
`06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx`; lesson 06's validation command is a correctness gate rather
than an input-generation step for this lesson. Before building, the script checks that the lesson 06
validation report passed and that its recorded SHA256 matches the ONNX file.

## Deliverables

- Dynamic-profile engine-build tooling and reusable batched-input layout helpers
- Reusable dynamic batch runner and CLI
- Batch-layout tests and saved batch benchmark evidence

## Layout

For input `[N,C,H,W]`, sample `n` starts at `n*C*H*W`. For YOLO output `[N,84,8400]`, sample `n`
starts at `n*84*8400`. The focused CPU test verifies these offsets and rejects invalid dimensions.

Batching usually increases images/second because one enqueue uses the GPU more efficiently, but a
real-time request must wait for its batch to fill and then shares the larger batch compute time.
Therefore throughput can improve while per-frame latency becomes worse.

## Build

Build this lesson's engine from the validated lesson 06 dynamic AutoCast ONNX model:

```bash
./17_dynamic_batching/build_dynamic_engine.sh
```

The script uses `trtexec --stronglyTyped` with the explicit ModelOpt AutoCast graph. The deprecated
weakly typed `--fp16` route is intentionally taught in lesson 06, but is not used here; it
does not rewrite the ONNX graph or make the network strongly typed. Its optimization profile is
`min=1x3x640x640`, `opt=2x3x640x640`, and `max=4x3x640x640`. It reuses lesson 06's timing cache and
uses builder optimization level 0 so the classroom build remains reasonably short. Benchmark
higher optimization levels before choosing production settings.

The optional positional arguments select a different dynamic ONNX input and engine output:

```bash
./17_dynamic_batching/build_dynamic_engine.sh \
  path/to/model.onnx \
  path/to/model.engine
```

The model must still expose an input named `images` that supports the profile declared in the
script. When overriding the ONNX path, also set `VALIDATION_REPORT` to its matching successful
lesson 06-format validation report.

Configure and build the C++ runner from the repository root inside the pinned development
container:

```bash
cmake -S 17_dynamic_batching -B 17_dynamic_batching/build
cmake --build 17_dynamic_batching/build --parallel
```

The generated engine, timing cache, and build directory are ignored.

## Run

Use the pinned TensorRT development container:

```bash
./17_dynamic_batching/build/dynamic_batching \
  --engine 17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine \
  --warmup 5 \
  --iterations 50
```

The explicit `--engine` argument above matches the program's default engine path and may be omitted.
The program calls `setInputShape()` before every enqueue, queries the resulting output shape,
allocates buffers for that concrete shape, and writes `outputs/batch_benchmark.csv`. It uses
deterministic synthetic NCHW tensors to isolate batching and buffer-layout behavior; it is not an
accuracy-validation run. The program also writes `outputs/batch_benchmark_environment.json` with
the GPU, compute capability, TensorRT version, and CUDA runtime/driver versions so measurements
remain attached to their execution platform.

### Experiments

1. Compare batch 1 and batch 4 compute latency and images/second in the generated CSV.
2. Change the optimization profile's opt shape and rebuild; compare tactic selection and timing.
3. Request batch 5 and observe that validation rejects a shape outside the declared profile.
4. Explain why offsets must use the concrete runtime output shape rather than a hard-coded batch.

## Outputs

- Committed deliverables include the engine-build script, C++ runner and layout helpers, tests, and
  this README.
- The TensorRT engine is an environment-specific generated artifact under ignored `outputs/`.
- Lesson 06's reused timing cache is generated under its ignored `outputs/` directory.
- `outputs/batch_benchmark.csv` and `outputs/batch_benchmark_environment.json` are ignored local
  benchmark evidence; they exist only after a successful run on the target GPU.

## Tests

Run the configured CTest suite:

```

<details><summary>Example output (local run)</summary>

```text
batch=1 compute=0.901 ms throughput=1109.927 images/s output_offset[1]=0
batch=2 compute=1.089 ms throughput=1836.264 images/s output_offset[1]=705600
batch=4 compute=1.537 ms throughput=2602.540 images/s output_offset[1]=705600
saved batch_benchmark.csv and batch_benchmark_environment.json
```
</details>
bash
ctest --test-dir 17_dynamic_batching/build --output-on-failure
```

The focused test covers CPU-only shape validation and batch-offset calculations. CMake configure
still requires the pinned container's CUDA and TensorRT development files because the same build
also defines the TensorRT runner; running the test itself does not require a GPU.

## Checkpoints

1. Build and run one TensorRT engine across multiple runtime batch sizes.
2. Calculate NCHW input/output offsets and set dynamic shapes through an optimization profile.
3. Compare batch latency and throughput using matched benchmark conditions.
