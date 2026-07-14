# 13 - C++ Producer-Consumer Pipeline

This lesson builds the concurrency foundation used by camera and video inference systems. One
thread repeatedly reads encoded image frames, while another thread simulates a slower inference
stage. A reusable bounded queue makes overload and shutdown behavior explicit.

The lesson is CPU-only on purpose: queue correctness can be tested quickly and with
ThreadSanitizer before CUDA streams and TensorRT execution are introduced in later lessons.

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

## Build and Run

From this lesson directory:

```bash
cmake -S . -B build
cmake --build build -j
./build/cpp_producer_consumer
```

The defaults locate the repository's `assets/dog.webp` and `assets/img2.jpeg` from the executable,
produce a frame every 10 ms, simulate 40 ms inference, limit the queue to four frames, and retain
the newest frames:

```bash
./build/cpp_producer_consumer \
  --frames 100 \
  --queue-capacity 4 \
  --producer-delay-ms 10 \
  --consumer-delay-ms 40 \
  --policy drop-oldest
```

Run with one or more custom images by repeating `--image`:

```bash
./build/cpp_producer_consumer --image ../assets/img2.jpeg --image ../assets/dog.webp
```

Use `--help` for every option. `--fail-producer-at` and `--fail-consumer-at` are deliberate failure
injection hooks for observing exception propagation and clean shutdown.

## Overload Experiments

Run the same workload with each policy:

```bash
./build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy block
./build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy drop-newest
./build/cpp_producer_consumer --frames 100 --queue-capacity 4 --policy drop-oldest
```

- `block` preserves every frame and maximizes completed work, but slows acquisition and couples the
  producer to inference speed.
- `drop-newest` preserves older queued work, which is useful when every accepted job should finish,
  but queueing latency can remain high.
- `drop-oldest` favors fresh camera frames and usually lowers real-time latency, at the cost of
  skipping older frames.

Increasing queue capacity absorbs short bursts but cannot fix a sustained throughput mismatch. If
input remains faster than inference, a larger queue mainly increases memory use and latency.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

The tests cover zero capacity, FIFO drain behavior, both dropping policies, wakeup of blocked
producers and consumers, bounded overload accounting, repeated pipeline construction/run/destruction,
explicit stop, and producer/consumer exception propagation.

For ThreadSanitizer, use a separate build directory:

```bash
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-tsan -j
ctest --test-dir build-tsan --output-on-failure
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

## Acceptance Criteria

- A producer thread reads image frames into a bounded queue and a consumer processes them.
- Queue capacity and overload policy are configurable and observable in reported statistics.
- Drain/discard close behavior is documented, rejects new pushes, and wakes all waiters.
- Worker failures reach the owner, and all started threads are joined before returning.
- Overload and repeated lifecycle tests keep queue depth bounded.
- CPU-only tests can be run under ThreadSanitizer.
