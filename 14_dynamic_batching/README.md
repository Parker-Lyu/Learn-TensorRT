# 14 - TensorRT Dynamic Batching

This lesson runs batch sizes 1, 2, and 4 through one TensorRT engine. It makes runtime shapes, NCHW
sample offsets, output offsets, and the latency/throughput trade-off explicit.

## Prerequisites and engine build

Complete the lesson 05 dynamic export and validation first. The validation tensor is generated from
the shared default image, `assets/img.jpeg`:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

TensorRT 10.14 strongly typed networks express reduced precision in the model rather than through a
builder precision flag. Install the lesson-local ONNX dependencies, run NVIDIA ModelOpt AutoCast,
and build the engine:

```bash
./14_dynamic_batching/setup_autocast_deps.sh
./14_dynamic_batching/build_dynamic_engine.sh
```

AutoCast preserves FP32 input/output tensors, converts the suitable backbone nodes to FP16, and
keeps the YOLO detection head in FP32. The exclusion keeps dynamic shape calculations and final box
decoding in their declared types. `trtexec --stronglyTyped` then builds the explicit mixed-precision
graph without weakly typed precision selection.

The profile is `min=1x3x640x640`, `opt=2x3x640x640`, and `max=4x3x640x640`. The script reuses lesson
06's timing cache and uses builder optimization level 0 so the classroom build remains reasonably
short; benchmark higher optimization levels before choosing production settings. Engines, the
AutoCast ONNX graph, calibration input, and timing cache are generated environment-specific
artifacts and remain ignored.

## Build and run

Use the pinned TensorRT development container:

```bash
cmake -S 14_dynamic_batching -B 14_dynamic_batching/build
cmake --build 14_dynamic_batching/build -j
ctest --test-dir 14_dynamic_batching/build --output-on-failure
./14_dynamic_batching/build/dynamic_batching \
  --engine 14_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine \
  --warmup 5 \
  --iterations 50
```

The program calls `setInputShape()` before every enqueue, queries the resulting output shape,
allocates buffers for that concrete shape, and writes `outputs/batch_benchmark.csv`. It also writes
`outputs/batch_benchmark_environment.json` with the GPU, compute capability, TensorRT version, and
CUDA runtime/driver versions so measurements remain attached to their execution platform.

## Layout

For input `[N,C,H,W]`, sample `n` starts at `n*C*H*W`. For YOLO output `[N,84,8400]`, sample `n`
starts at `n*84*8400`. The focused CPU test verifies these offsets and rejects invalid dimensions.

Batching usually increases images/second because one enqueue uses the GPU more efficiently, but a
real-time request must wait for its batch to fill and then shares the larger batch compute time.
Therefore throughput can improve while per-frame latency becomes worse.

## Experiments

1. Compare batch 1 and batch 4 compute latency and images/second in the generated CSV.
2. Change the optimization profile's opt shape and rebuild; compare tactic selection and timing.
3. Request batch 5 and observe that validation rejects a shape outside the declared profile.
4. Explain why offsets must use the concrete runtime output shape rather than a hard-coded batch.
