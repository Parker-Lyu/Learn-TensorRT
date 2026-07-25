# 14 - TensorRT Dynamic Batching

This lesson runs batch sizes 1, 2, and 4 through one TensorRT engine. It makes the runtime shape,
NCHW sample offsets, output offsets, and latency/throughput trade-off explicit.

## Build the Engine

The checked-in course artifacts already include a compatible lesson 06 dynamic FP16 engine. To
build a lesson-local engine with a fixed 640x640 spatial shape and dynamic batch profile:

```bash
./build_dynamic_engine.sh
```

The profile is `min=1x3x640x640`, `opt=2x3x640x640`, and `max=4x3x640x640`. Engines are generated,
environment-specific artifacts and remain ignored. The script reuses lesson 06's timing cache and
uses builder optimization level 0 so the classroom build remains reasonably short; production
release builds should benchmark higher optimization levels separately. TensorRT 10.14 removed the
old `--minTiming` and `--avgTiming` builder flags, so the script does not use them.

## Build and Run

Use the pinned TensorRT development container:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/dynamic_batching --warmup 5 --iterations 50
```

The default engine is `../06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine`. Use the
lesson-local engine with:

```bash
./build/dynamic_batching --engine outputs/yolov8n_batch1_4_fp16.engine
```

The program calls `setInputShape()` before every enqueue, queries the resulting output shape,
allocates buffers for that concrete shape, and writes `outputs/batch_benchmark.csv`. It also writes
`outputs/batch_benchmark_environment.json` with the GPU, compute capability, TensorRT version, and
CUDA runtime/driver versions so the measurements are not separated from their execution platform.

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
