# 20 - CUDA Preprocessing and NPP

## Purpose

This lesson moves measured preprocessing work to the GPU. NPP performs bilinear resize, then one
CUDA kernel fuses BGR-to-RGB conversion, `uint8` normalization, and HWC-to-CHW layout conversion.
The output is checked against an OpenCV CPU reference before timing is accepted.

## Prerequisites

- Complete lesson 03 and use its CPU preprocessing contract as the reference.
- Use the pinned development container with CUDA, NPP, OpenCV, and an accessible NVIDIA GPU.

## Deliverables

- Reusable CPU/CUDA/NPP preprocessing library
- Correctness and benchmark executable
- Focused preprocessing tests and saved timing evidence

## Memory Modes

- `pageable`: copies directly from ordinary OpenCV and `std::vector` memory. CUDA may use internal
  staging, so async copies do not guarantee host/GPU overlap.
- `pinned`: reuses page-locked staging buffers. DMA is predictable, but the CPU staging copies and
  pinned-memory footprint must be counted.
- `mapped`: the GPU accesses mapped host buffers without explicit H2D/D2H copies. On a discrete GPU
  this is PCIe traffic on demand and can make the kernel slower; zero explicit copy is not zero cost.

The benchmark reports host staging separately so mapped or pinned paths are not credited with
hidden CPU copies. On integrated Jetson memory, mapped behavior can differ substantially from a
desktop discrete GPU.

## Why NPP Plus a Custom Kernel

NPP supplies optimized resize primitives and application-managed stream-context APIs. The lesson
fills the CUDA 13 `NppStreamContext` fields once and passes that context explicitly; it does not
change NPP process-global stream state. The custom kernel fuses three
simple per-pixel transformations that would otherwise require intermediate tensors. This division
keeps the lesson readable while demonstrating when a library primitive and a small fused kernel
fit together.

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 20_cuda_preprocess_npp -B 20_cuda_preprocess_npp/build
cmake --build 20_cuda_preprocess_npp/build --parallel
```

The generated build directory is ignored.

## Run

Run the commands from the repository root:

Use the pinned TensorRT development container and an NVIDIA GPU:

```bash
./20_cuda_preprocess_npp/build/cuda_preprocess_npp --iterations 50
```

<details><summary>Example output (local run)</summary>

```text
pageable stage=0.0001 h2d=0.1811 resize=0.0110 conversion=0.0051 gpu=0.0195 d2h=0.3403 max_error=0.2863 mean_error=0.0158
pinned stage=0.1242 h2d=0.1343 resize=0.0091 conversion=0.0051 gpu=0.0176 d2h=0.2453 max_error=0.2863 mean_error=0.0158
GPU=NVIDIA GeForce RTX 4090 compute_capability=8.9 CUDA_runtime=13000 CUDA_driver=13020
cpu=2.6289 ms saved preprocess_benchmark.csv
```
</details>


The default input is `../assets/img.jpeg`. Results are written to
`outputs/preprocess_benchmark.csv`. The adjacent `preprocess_benchmark_environment.json` records
the GPU, compute capability, and CUDA runtime/driver versions for the measured data. CPU
preprocessing, host staging, H2D, NPP resize, fused conversion, combined GPU preprocessing, and D2H
are separate columns. The exact-size unit test requires the fused
conversion to match within `1e-6`. For resized images, the executable uses mean absolute error
`<=0.02` and maximum error `<=0.30`, because NPP and OpenCV use different bilinear sampling
coordinates near some boundaries. Both errors remain in the CSV; do not hide a large local error
behind the mean.

## Outputs

- Correctness metrics and CPU/GPU timing are printed and saved under ignored `outputs/`.
- Transfer time remains separate from resize and fused-kernel time.

## Tests

### Validation and Profiling

Run CUDA memory checking on the focused smoke test:

```bash
compute-sanitizer --tool memcheck ./20_cuda_preprocess_npp/build/cuda_preprocess_tests
```

Then profile the executable with Nsight Systems and verify the `host_input_staging`, `h2d_submit`,
`npp_resize_submit`, `bgr_to_rgb_nchw_submit`, and `d2h_submit` NVTX ranges correspond to work on
the same CUDA stream. NVTX ranges describe host submission; CUDA events provide device duration.
Compare the CSV before claiming GPU preprocessing is faster: for one
small image, transfer and launch overhead can outweigh the kernel savings. The most useful path is
often decode/capture directly into GPU-visible memory (NVDEC, DeepStream NVMM, or a platform camera
API), avoiding the host round trip rather than merely optimizing it.

Run the configured CTest suite:

```bash
ctest --test-dir 20_cuda_preprocess_npp/build --output-on-failure
```

## Checkpoints

1. Move selected preprocessing work to CUDA/NPP and validate it against the OpenCV reference.
2. Measure transfer and preprocessing costs separately before claiming an optimization.
3. Explain the trade-offs among explicit copies, mapped memory, Unified Memory, and GPU-native decode paths.
