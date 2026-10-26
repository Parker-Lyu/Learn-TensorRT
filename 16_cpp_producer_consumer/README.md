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
decoded frames, preprocessed tensors, or inference results. Payloads must be nothrow move
constructible and nothrow destructible (checked at compile time). This keeps `pop()` safe even
when returning its `std::optional<T>`; a throwing move could otherwise damage the queued value or
lose it after removal. Use an owning handle such as `std::unique_ptr<T>` for other payload types.
`push(T value)` consumes an rvalue even if closed or dropped; it does not return rejected ownership.
A copy made by the caller or a deque allocation can still throw before successful insertion.
Its public behavior is:

- `push()` returns whether an item was accepted, dropped, or rejected after close.
- `pop()` blocks until an item is available or the closed queue is empty.
- `close(Drain)` rejects new work and lets the consumer finish queued work.
- `close(Discard)` rejects new work and immediately removes queued work.
- `close()` wakes both blocked producers and blocked consumers.

The normal producer completion path uses drain. Cancellation and worker failures use discard so a
blocked peer can exit promptly. The first worker exception is saved and rethrown by `run()` after
both threads have joined.

`close(Discard)` can escalate an earlier drain; repeated discard does not double-count removed
items. DropOldest inserts the replacement before removing the old value, so a failed insertion
preserves the queue. This may temporarily hold one additional item under the mutex, but externally
visible queue size stays bounded. Failed blocking insertion wakes another producer that can use
the free slot. The pipeline also joins the producer if consumer-thread creation fails.
`run()` is single-use and must not be called concurrently; `stop()` may be called by another thread.
The owner must wait for `run()` to finish before destroying the pipeline or its input paths.
Simulated delays use interruptible condition-variable waits. Stopping wakes them as well as queue
waiters, but cannot interrupt a file read or an actual CUDA kernel. Discard removes pending work;
an already-popped frame may be cancelled and is not counted as processed or as a queue drop.

Latency mean and maximum use constant-space online statistics. The timestamp is taken after file
reading, so this metric includes blocking in `push()` but excludes file I/O and consumer work; it
is not sensor-to-result latency. Queue capacity bounds item count, not encoded-byte size or all
pipeline memory. DropOldest can hold one extra item temporarily, and producer/consumer frames live
outside the queue.

### TensorRT 10 integration boundary

A later inference consumer should own an execution context, a non-default CUDA stream, and its
I/O buffers. Concurrent workers need separate contexts; a queue mutex does not protect subsequent
context calls. This follows NVIDIA's [TensorRT 10.x threading model](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/architecture/how-trt-works.html#threading).

Use name-based `setTensorAddress()` and `enqueueV3(stream)` and check their return values.
Keep buffers alive until their GPU uses finish; an input-consumed event permits input reuse, while
outputs require stream completion and any later copy must also finish. See the
[TensorRT 10.14 runtime contract](https://github.com/NVIDIA/TensorRT/blob/release/10.14/include/NvInferRuntime.h).
Limit in-flight GPU slots separately from queued items. On failure or stop, reject new jobs, drain
GPU work, join workers, then release resources. Queue discard cannot cancel kernels. This lesson
implements none of those GPU operations and reports no TensorRT performance result.

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

From this lesson directory, run the default producer/consumer workload:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer
```

<details><summary>Example output (local run)</summary>

```text
policy: drop-oldest
frames read: 20
frames processed: 10
frames dropped: 10
queue high watermark: 4/4
average queue latency: 62.85 ms
max queue latency: 155.82 ms
```
</details>

To reproduce an overloaded queue with explicit timing and drop policy, run:
This variant locates the repository's `assets/img.jpeg`, produces a frame every 10 ms, simulates
40 ms inference, limits the queue to four frames, and retains the newest frames:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer \
  --frames 100 \
  --queue-capacity 4 \
  --producer-delay-ms 10 \
  --consumer-delay-ms 40 \
  --policy drop-oldest
```

To use a custom input image (repeat `--image` to provide more than one), run:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer --image assets/img.jpeg
```

To list all options, including deliberate producer/consumer failure-injection hooks, run:

```bash
./16_cpp_producer_consumer/build/cpp_producer_consumer --help
```

`--fail-producer-at` and `--fail-consumer-at` are deliberate failure
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
explicit stop, and producer/consumer exception propagation. They also exercise move-only ownership,
4 producers / 3 consumers transferring 1,200 unique items, constant-space latency summary invariants,
stop-before-run, single-use rejection, and cancellation with 30-second simulated delays on either
worker or both. CTest limits the suite to 30 seconds. Timing-based wakeup checks remain smoke checks;
they do not deterministically establish every wait interleaving.

For ThreadSanitizer, use a separate build directory:

```bash
cmake -S 16_cpp_producer_consumer -B 16_cpp_producer_consumer/build-tsan -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build 16_cpp_producer_consumer/build-tsan -j
ctest --test-dir 16_cpp_producer_consumer/build-tsan --output-on-failure
```

ThreadSanitizer is a CPU-only concurrency check. Run it separately from CUDA/TensorRT programs;
sanitizer and driver runtimes can otherwise produce unrelated diagnostics.

2026-10-22 local verification under the user's local-only instruction: the producer-consumer test
suite compiled with Apple Clang 21.0.0, ISO C++17, `-Wall -Wextra -Wpedantic -pthread`, and passed on
macOS. At that revision this included drain-to-discard escalation and a synthetic failed DropOldest insertion.
CMake/CTest and ThreadSanitizer were not run; thread-creation failure was inspected, not injected.

2026-10-26 local-only review: Apple Clang 21.0.0 compiled both the tests and executable as ISO
C++17 with `-Wall -Wextra -Wpedantic -Werror -pthread`; the suite passed, including the new cases
above. A separate `-fsanitize=thread -g -fno-omit-frame-pointer` build passed without sanitizer
reports. A compile-negative check rejected a throwing-move payload with the expected static
assertion. The former throwing-move runtime test is superseded by that stricter type contract.
The executable also ran with the existing image, 20 frames, blocking policy, and zero delays:
20 read, 20 processed, 0 dropped. Latency and peak depth vary by scheduling.
Docker and CMake are unavailable on this Mac, so no container, CMake/CTest, GPU, or TensorRT run
was performed. Thread startup/allocation failure was not injected. No speedup is claimed.

## Checkpoints

1. Make the consumer four times slower than the producer and compare dropped frames and queue
   latency for all policies.
2. Change `close(Drain)` on normal completion to `close(Discard)` and explain the accounting change.
3. Inject a consumer failure while using the blocking policy and verify the producer does not hang.
4. Replace the simulated inference sleep with a CPU image operation while keeping queue ownership
   and shutdown behavior unchanged.
