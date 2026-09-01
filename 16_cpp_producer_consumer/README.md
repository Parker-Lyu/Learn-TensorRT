# 16 - C++ Producer-Consumer Pipeline

## Purpose

This lesson builds the concurrency foundation used by camera and video inference systems. One
thread repeatedly reads encoded image frames, while another thread simulates a slower inference
stage. A reusable bounded queue makes overload and shutdown behavior explicit.

The lesson is CPU-only on purpose: queue correctness can be tested quickly and with
ThreadSanitizer before CUDA streams and TensorRT execution are introduced in later lessons.

## Prerequisites

- Complete `01_hello_world`; lesson 03 is useful for the image-processing extension.
- No GPU or TensorRT engine is required because this lesson intentionally tests concurrency on CPU.

## Deliverables

- Reusable bounded queue and `producer_consumer_pipeline` library
- `cpp_producer_consumer` executable
- Queue, overload, cancellation, failure, and lifecycle tests

## Learning Goals

- Use `std::thread`, `std::mutex`, and `std::condition_variable` with clear ownership.
- Keep memory bounded when the producer is faster than the consumer.
- Compare blocking, drop-newest, and drop-oldest overload policies.
- Define drain and discard shutdown semantics.
- Wake blocked producers and consumers during shutdown.
- Propagate worker exceptions back to the thread owner.
- Measure queue depth, dropped frames, and time spent waiting for inference.

## Design

```text
image files -> producer thread -> BoundedQueue<ImageFrame> -> consumer thread
                    read bytes      capacity + policy          simulated inference
```

`BoundedQueue<T>` is independent of image and TensorRT types, so later lessons can reuse it for
decoded frames, preprocessed tensors, or inference results. Its public behavior is:

- `push()` returns whether an item was accepted, dropped, or rejected after close.
- `pop()` blocks until an item is available or the closed queue is empty.
- `close(Drain)` rejects new work and lets the consumer finish queued work.
- `close(Discard)` rejects new work and immediately removes queued work.
- `close()` wakes both blocked producers and blocked consumers.

The normal producer completion path uses drain. Cancellation and worker failures use discard so a
blocked peer can exit promptly. The first worker exception is saved and rethrown by `run()` after
both threads have joined.

## Directory Layout

- `include/bounded_queue.hpp`: reusable bounded, thread-safe queue.
- `include/image_pipeline.hpp`: frame, configuration, statistics, and pipeline API.
- `src/image_pipeline.cpp`: producer/consumer lifetime and exception propagation.
- `src/main.cpp`: CLI and statistics output.
- `tests/test_producer_consumer.cpp`: CPU-only queue, overload, shutdown, failure, and stress tests.


## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 16_cpp_producer_consumer -B 16_cpp_producer_consumer/build
cmake --build 16_cpp_producer_consumer/build --parallel
```

The generated build directory is ignored.

## Run

Run the commands from the repository root:

From this lesson directory:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer
```

<details><summary>Example output (local run)</summary>

```text
frames read: 20
frames processed: 9
frames dropped: 11
queue high watermark: 4/4
average queue latency: 49.94 ms
max queue latency: 122.82 ms
```
</details>

The default locates the repository's `assets/img.jpeg` from the executable,
produce a frame every 10 ms, simulate 40 ms inference, limit the queue to four frames, and retain
the newest frames:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer \
  --frames 100 \
  --queue-capacity 4 \
  --producer-delay-ms 10 \
  --consumer-delay-ms 40 \
  --policy drop-oldest
```

Run with one or more custom images by repeating `--image`:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer --image assets/img.jpeg
```

Use `--help` for every option. `--fail-producer-at` and `--fail-consumer-at` are deliberate failure
injection hooks for observing exception propagation and clean shutdown.

### Overload Experiments

Run the same workload with each policy:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy block
./16_cpp_producer_consumer/build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy drop-newest
./16_cpp_producer_consumer/build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy drop-oldest
```

- `block` preserves every frame and maximizes completed work, but slows acquisition and couples the
  producer to inference speed.
- `drop-newest` preserves older queued work, which is useful when every accepted job should finish,
  but queueing latency can remain high.
- `drop-oldest` favors fresh camera frames and usually lowers real-time latency, at the cost of
  skipping older frames.

Increasing queue capacity absorbs short bursts but cannot fix a sustained throughput mismatch. If
input remains faster than inference, a larger queue mainly increases memory use and latency.

## Outputs

- The executable reports produced, processed, dropped, queue-peak, and timing statistics to standard output.
- Build and sanitizer artifacts remain under ignored build directories.

## Tests

```bash
ctest --test-dir 16_cpp_producer_consumer/build --output-on-failure
```

The tests cover zero capacity, FIFO drain behavior, both dropping policies, wakeup of blocked
producers and consumers, bounded overload accounting, repeated pipeline construction/run/destruction,
explicit stop, and producer/consumer exception propagation.

For ThreadSanitizer, use a separate build directory:

```bash
cmake -S 16_cpp_producer_consumer -B 16_cpp_producer_consumer/build-tsan -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build 16_cpp_producer_consumer/build-tsan -j
ctest --test-dir 16_cpp_producer_consumer/build-tsan --output-on-failure
```

ThreadSanitizer is a CPU-only concurrency check. Run it separately from CUDA/TensorRT programs;
sanitizer and driver runtimes can otherwise produce unrelated diagnostics.

## Checkpoints

1. Make the consumer four times slower than the producer and compare dropped frames and queue
   latency for all policies.
2. Change `close(Drain)` on normal completion to `close(Discard)` and explain the accounting change.
3. Inject a consumer failure while using the blocking policy and verify the producer does not hang.
4. Replace the simulated inference sleep with a CPU image operation while keeping queue ownership
   and shutdown behavior unchanged.
