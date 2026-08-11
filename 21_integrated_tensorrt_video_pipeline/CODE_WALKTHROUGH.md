# Lesson 21 Code Walkthrough: From C++ Scheduling to CUDA/TensorRT Inference

> Intended audience: readers with limited C++ experience who are new to CUDA and want to understand the overall design before studying the critical code paths.
>
> This document explains only the implementation currently in the repository. It does not replace the build, run, and acceptance instructions in this lesson's `README.md`. Line numbers refer to the source revision used when this document was written; if the source changes, locate code by function name instead.

## 1. First, Understand What This Lesson Does

This is not an example that merely invokes TensorRT once for a single image. It assembles a fairly complete video inference pipeline:

```text
One or more image/video sources
        │ (one capture thread per source)
        ▼
A separate bounded queue for each source
        │ (round-robin or latest-frame retrieval)
        ▼
FrameScheduler forms dynamic batches
        │
        ▼
TensorRtBackend reserves an available slot
        │
        ├─ Copy CPU images to pinned host memory
        ├─ H2D: host memory → GPU memory
        ├─ NPP resize + letterbox
        ├─ CUDA kernel: BGR/HWC/uint8 → RGB/CHW/float
        ├─ TensorRT enqueueV3
        └─ D2H: inference output → pinned host memory
        │
        ▼
CPU YOLO decoding, NMS, JSONL output, and annotated image output
        │
        ▼
Summarize throughput, latency, dropped frames, and environment information
```

The real focus is not any single API, but these four engineering problems:

1. **Backpressure**: memory must not grow without bound when capture is faster than inference.
2. **Concurrent ownership**: multiple CUDA streams can run concurrently, but they must not incorrectly share execution contexts or work buffers.
3. **Asynchronous identity**: batches do not necessarily finish in submission order, so result identity cannot be inferred from completion order.
4. **Explainable metrics**: captured, dropped, submitted, and completed work must reconcile, and CPU time must not be confused with GPU time.

## 2. Recommended Reading Order

Do not force yourself to read every file from top to bottom in filename order. Use these five passes instead:

1. `include/pipeline_core.hpp` + `src/pipeline_core.cpp`: first understand data identity, queues, and the slot state machine.
2. `include/frame_source.hpp` + `src/frame_source.cpp`: understand where frames come from.
3. `include/frame_scheduler.hpp` + `src/frame_scheduler.cpp`: understand how multiple threads produce batches.
4. `src/integrated_pipeline.cpp`: study the orchestration loop and connect the modules.
5. `src/tensorrt_backend.cu`: only then study CUDA/NPP/TensorRT in depth; finish with `result_writer.cpp` and `metrics.cpp`.

Distinguish the two entry points:

- `src/main.cpp` builds `integrated_tensorrt_video_pipeline`. It is only a **CPU slot/identity test program** and performs no real inference.
- `src/pipeline_app.cpp` builds `integrated_tensorrt_video_pipeline_gpu`, the entry point for the complete GPU pipeline.

## 3. Minimum C++ Knowledge Needed to Read This Code

### 3.1 Values, References, and Moves

- `const T&`: a read-only borrow; `T` is not copied.
- `T&`: a mutable borrow.
- `T value`: the function receives an independent object, copied or moved at the call site.
- `std::move(x)`: does not move data by itself. It converts `x` into an rvalue from which resources may be taken by a subsequent move construction or assignment. A moved-from `x` remains destructible, but you must not assume that it retains its previous contents.

This lesson frequently moves `std::vector`, `std::unique_ptr`, and metadata objects to transfer ownership without copying large objects.

`cv::Mat` is unusual: ordinary assignment normally copies only a reference-counted matrix header, while the pixel buffer remains shared. Only `clone()` performs a deep pixel copy. Images returned by `FrameSource` are read-only until they enter the GPU path, so sharing is safe here.

### 3.2 `unique_ptr` and RAII

`std::unique_ptr<T>` represents exclusive ownership. It cannot be copied, only moved, and automatically destroys its object when it leaves scope. This is a standard example of RAII (Resource Acquisition Is Initialization).

Raw CUDA resources are not C++ objects, so `Slot::~Slot()` in `tensorrt_backend.cu` explicitly destroys device buffers, pinned buffers, events, and streams. The outer `unique_ptr<Slot>` owns each `Slot`, completing the automatic cleanup chain.

### 3.3 `optional`

`std::optional<std::size_t>` means that a slot index may or may not be present:

```cpp
const auto reserved = backend.try_reserve();
if (!reserved) { /* no available slot */ }
backend.submit(*reserved, ...); // * extracts the index
```

This is safer than using `-1` for failure because `std::size_t` is unsigned and converting `-1` can cause subtle bugs.

### 3.4 Mutexes, Condition Variables, and Atomics

- `std::mutex` protects a group of values that must remain mutually consistent, such as queue contents and `closed_`.
- `std::lock_guard` automatically unlocks at scope exit and suits short critical sections that do not wait.
- `std::unique_lock` can unlock and relock temporarily and is required by `condition_variable::wait`.
- `std::condition_variable` puts threads to sleep while waiting for “not empty” or “not full,” avoiding CPU-burning busy loops.
- `std::atomic` suits simple counters and flags, but cannot replace a mutex that protects compound state.

### 3.5 `try/catch (...)` and `exception_ptr`

`catch (...)` catches any exception. `std::current_exception()` stores the current exception in an `exception_ptr`; `std::rethrow_exception()` can later rethrow it while preserving its original type and message. This lesson uses that mechanism to propagate capture-thread exceptions to the control thread and to preserve the first root cause during cleanup.

## 4. Directory Structure and Build Targets

### 4.1 Header Responsibilities

| File | Main contents |
|---|---|
| `include/pipeline_core.hpp` | Metadata, accounting structures, slot state machine, bounded queue |
| `include/frame_source.hpp` | Frame-source abstraction and factory functions |
| `include/frame_scheduler.hpp` | Multi-source capture, queues, and dynamic batch scheduling |
| `include/config.hpp` | Command-line configuration |
| `include/tensorrt_backend.hpp` | Public GPU backend interface and result types |
| `include/result_writer.hpp` | Postprocessing result output |
| `include/metrics.hpp` | Metric collection and aggregation |
| `include/integrated_pipeline.hpp` | Single entry point for the complete pipeline |

### 4.2 CMake Targets

`CMakeLists.txt` deliberately separates testable logic:

- `integrated_pipeline_core`: pure C++ queue, slot, and identity logic.
- `integrated_frame_scheduler`: frame sources and scheduler; depends on OpenCV.
- `integrated_tensorrt_backend`: CUDA, NPP, and TensorRT backend.
- `integrated_yolo_postprocess`: reuses the Lesson 11 YOLO pre/postprocessing code; this lesson invokes only postprocessing during result output.
- `integrated_tensorrt_video_pipeline`: CPU identity smoke program.
- `integrated_tensorrt_gpu_smoke`: tests only the GPU backend, without the complete scheduler.
- `integrated_tensorrt_video_pipeline_gpu`: complete application.

This is easier to test independently than placing every `.cpp/.cu` file in one executable, and it makes dependency direction explicit.

## 5. First Pass: Walk Through the Entire Project by Function

This section answers only “what is each function responsible for?” and intentionally postpones synchronization details.

### 5.1 Configuration: `config.cpp`

- `positive(value, name)`: converts a string to a positive integer and rejects zero and negative values.
- `usage()`: returns the complete command syntax.
- `parse_config(argc, argv)`: parses positional arguments first, then `--xxx` options, and validates the batch limit and duration mode.

The parser treats an argument as the next positional argument whenever its first character is not `-`. It is a lightweight teaching implementation, not a general-purpose CLI framework.

### 5.2 Frame Sources: `frame_source.cpp`

- `ImageSequenceSource::read()`: loops over an in-memory image sequence and returns `false` after the configured frame count is reached.
- `VideoFileSource::read()`: decodes exactly one frame per call; at end of file, it either stops or reopens the video according to `repeat_`.
- `make_repeatable_image_source()`: wraps a single image as a sequence of length one.
- `make_image_sequence_source()`: creates an image-sequence source.
- `make_synthetic_source()`: creates a 640×480 solid-color test source.
- `make_path_source()`: recognizes input in this order: `synthetic`, `sequence:...`, ordinary image, then video.

`FrameSource` is an abstract base class. `virtual bool read(...) = 0` is a pure virtual function that derived classes must implement. Callers depend only on the interface and need not know whether input comes from images or video.

### 5.3 Scheduling: `frame_scheduler.cpp`

- Three `FrameScheduler` constructors normalize different input forms into `vector<unique_ptr<FrameSource>>`.
- `start()`: starts one capture thread per source.
- `capture(stream)`: repeatedly reads frames, attaches metadata, and pushes them into that source's bounded queue.
- `next_batch(maximum, timeout)`: retrieves frames from multiple queues and forms a batch of at most `maximum` frames.
- `stop(discard)`: closes queues, wakes threads, and `join()`s every capture thread.
- `rethrow_source_error()`: rethrows capture-thread exceptions on the main thread.
- `evicted()/discarded()/queue_peak()/queue_depth()/done()`: aggregate queue state.

### 5.4 CPU Core: `pipeline_core.cpp`

- `Accounting::validate_terminal()`: validates the terminal-state accounting identity.
- `SlotPool::reserve()`: blocks until an available slot exists.
- `SlotPool::try_reserve()`: immediately returns an available slot or an empty value.
- `mark_submitted()`: performs `Reserved → Submitted` and saves immutable batch identity.
- `begin_collection()`: performs `Submitted → Completing`.
- `release()`: performs `Completing → Free`, clears metadata, and notifies waiters.
- `fail()`: marks an occupied slot as `Failed` and prevents unsafe reuse.
- `IdentityDispatcher::dispatch()`: validates `batch_index` and assigns results by metadata rather than guessing from completion order.

### 5.5 GPU Backend: `tensorrt_backend.cu`

- `Logger::log()`: prints only TensorRT warnings and more severe messages.
- `check_cuda()/check_npp()`: convert error codes into C++ exceptions.
- `read_engine()`: reads a serialized engine as binary data.
- `npp_context()`: combines current CUDA-device properties and a stream into an NPP context.
- `normalize<<<...>>>()`: converts BGR→RGB, HWC→CHW, and uint8→float/255 per pixel.
- `replace_device_buffer()/replace_pinned_buffer()`: reallocate only when capacity is insufficient, then reuse memory during steady state.
- `Slot::~Slot()`: waits for the slot's stream and releases resources according to their type.
- `TensorRtBackend::Impl::Impl()`: deserializes the engine and creates a separate stream, events, and execution context for every slot.
- `ensure_capacity()`: grows buffers for the current batch's largest source image and TensorRT I/O sizes.
- `submit()`: asynchronously queues an entire batch's H2D, preprocessing, inference, and D2H operations on one slot stream.
- `ready()`: queries the `done` event without blocking.
- `collect()`: waits for a specified slot, copies pinned output into a C++ vector, reads stage timings, and releases the slot.
- `identity()`: reads GPU, compute capability, CUDA, and TensorRT version information.

### 5.6 Orchestration: `integrated_pipeline.cpp`

- `sources(config)`: creates frame sources from configured paths; ordinary mode allocates a frame count to each source, while duration mode permits continuous reading.
- `run_integrated_pipeline(config)`: creates all modules, starts the scheduler, submits and collects batches in a loop, and finally writes metrics.

### 5.7 Postprocessing and Metrics

- `ResultWriter::write()`: slices TensorRT output by batch; reuses Lesson 11 YOLO decoding/NMS; maps boxes back to original-image coordinates; writes detection JSONL and the first annotated image; and records end-to-end latency.
- `PipelineMetrics::record_batch()`: accumulates stage timing, batch distribution, and per-stream latency, while streaming every batch's raw sample to JSONL.
- `PipelineMetrics::write()`: generates the final `metrics.json`.

### 5.8 Two Small Entry Points

- `main.cpp`: occupies two CPU slots and deliberately collects them in reverse order, proving that identity follows metadata rather than completion position.
- `gpu_smoke.cpp`: reads one image, replicates it into a batch, and calls the backend directly; with `--two-slots`, it submits two batches and then collects them in reverse order.

## 6. Second Pass: What Actually Happens to One Frame

Assume the command has two sources, `BATCH=4`, and `SLOTS=2`:

1. `pipeline_app.cpp:main()` calls `parse_config()`.
2. `run_integrated_pipeline()` creates one shared engine and two slots. Each slot owns its own context, stream, events, and memory.
3. `scheduler.start()` starts two capture threads.
4. Each capture thread calls `FrameSource::read()`. After success it creates:

   ```text
   ScheduledFrame
   ├─ image: cv::Mat
   └─ metadata
      ├─ stream_id: source stream
      ├─ frame_id: frame number within that stream
      ├─ batch_index: assigned only when the batch is formed
      └─ captured_at: monotonic-clock time after successful capture
   ```

5. Each frame enters its source's `BoundedQueue`. When full, the queue either blocks or removes its oldest frame.
6. The main thread calls `next_batch(4, 4ms)` and retrieves frames from both streams in turn, up to four frames.
7. The main thread reserves a slot, assigns a `batch_id`, computes queue waiting time, and calls `backend.submit()`.
8. `submit()` only queues GPU work in order on the slot stream. Except for uncommon synchronization such as memory growth, it does not wait for the entire GPU chain to finish.
9. The slot enters `Submitted`, while the main thread can use the other slot for the next batch.
10. The main thread calls `ready()` to find any completed slot. If none is ready, it calls blocking `collect()` on `pending.front()` to avoid pure polling.
11. `collect()` waits for the `done` event, retrieves output and metadata, and returns the slot to `Free`.
12. `ResultWriter` uses `transform` in the metadata to map detection boxes back to source-image coordinates and writes identity as `(stream_id, frame_id)`.

Key point: **a slot index identifies a reusable execution resource; it is not the identity of a frame or batch.**

## 7. Difficult Topic 1: Understanding the Bounded Queue Line by Line

The core is in `pipeline_core.hpp:89-136`.

### 7.1 Construction

```cpp
BoundedQueue(std::size_t capacity, OverloadPolicy policy)
    : capacity_(capacity), policy_(policy) {
    if (capacity == 0) throw std::invalid_argument(...);
}
```

- `capacity_` and `policy_` are `const` and never change after construction.
- With a capacity of zero, “not full” could never be true, so the constructor rejects it.

### 7.2 `push()`: Producer Path

```cpp
std::unique_lock<std::mutex> lock(mutex_);
```

From this point, only one thread at a time can observe or modify `values_`, `closed_`, and the statistics.

```cpp
if (policy_ == OverloadPolicy::Block) {
    not_full_.wait(lock, [this] {
        return closed_ || values_.size() < capacity_;
    });
    if (closed_) return false;
}
```

- Under the block policy, the producer sleeps while the queue is full.
- `wait` unlocks the mutex internally so a consumer can `pop()`, then locks it again after waking.
- The predicate must also test `closed_`; otherwise a producer waiting on a full queue might never finish after closure.
- A wait may wake spuriously, so the predicate rechecks the condition.

```cpp
else if (values_.size() == capacity_) {
    values_.pop_front();
    ++evicted_;
}
```

The drop-oldest policy never blocks. A full queue discards its oldest frame. This reduces the staleness of a real-time view at the cost of no longer producing results for every frame.

```cpp
if (closed_) return false;
values_.push_back(std::move(value));
peak_ = std::max(peak_, values_.size());
not_empty_.notify_one();
return true;
```

- New values are rejected after closure.
- The value is moved into the deque.
- `peak_` records the historical peak for this stream.
- One consumer waiting for “not empty” is awakened.

### 7.3 `pop()` and `try_pop()`

`pop()` waits for “closed or not empty.” If the queue is closed but still contains values, it continues draining them; this behavior enables normal end-of-stream draining. It returns `nullopt` only when the queue is both closed and empty.

`try_pop()` never waits. The scheduler must poll multiple stream queues. Blocking on the first queue could hide a ready frame in the second, so the non-blocking form is required here.

Both functions call `not_full_.notify_one()` after removing data so that a producer blocked under the block policy can continue.

### 7.4 `close(discard)`

```cpp
closed_ = true;
if (discard) {
    discarded_ += values_.size();
    values_.clear();
}
not_empty_.notify_all();
not_full_.notify_all();
```

- `discard=false`: accept no new frames, but allow consumers to drain queued frames; suitable for normal completion.
- `discard=true`: immediately clear frames that have not been submitted; suitable for an exceptional abort or duration expiry.
- Both sides call `notify_all()` because producers and consumers may both be asleep.

`close()` is irreversible. The class has no reopen operation, matching a single pipeline lifecycle.

## 8. Difficult Topic 2: The FrameScheduler Threading Model

### 8.1 Why Each Source Has Its Own Queue

If all sources shared one queue, a fast source could completely overwhelm a slow one. The current design uses one thread and one queue per source, then has `next_batch()` poll the streams in turn. This makes fairness and per-stream statistics easier to implement.

### 8.2 Understanding `capture()` in Stages

This corresponds to `frame_scheduler.cpp:64-93`:

```cpp
std::uint64_t frame_id = 0;
cv::Mat image;
while (!stopping_) {
```

Each capture thread has its own `frame_id`, so global identity must be `(stream_id, frame_id)`. `stopping_` is atomic, avoiding a data race between the stop thread and capture threads that an ordinary `bool` would create.

```cpp
if (!sources_[stream]->read(image)) break;
if (image.empty()) throw ...;
ScheduledFrame item{
    image,
    {stream, frame_id++, 0, Clock::now(), {}}
};
++captured_;
```

- `read=false` means normal end of stream.
- Metadata uses aggregate initialization in the same field order as `FrameMetadata`.
- `batch_index` starts at zero because no batch exists yet.
- The timestamp is taken after `read()` succeeds.
- `captured_` is incremented before the enqueue attempt. If the queue has been closed, `rejected_on_close_` must account for that frame later.

```cpp
if (!queues_[stream]->push(std::move(item))) {
    ++rejected_on_close_;
    break;
}
```

`false` means only that the queue was already closed. The frame was read but not admitted into the queue, so it receives a separate count.

On an exception, the code stores the first exception in `source_error_` and then closes all queues. A mutex protects `source_error_` because multiple capture threads could fail concurrently. Finally, whether termination is normal or exceptional, the stream calls `close(false)` and increments `finished_`.

### 8.3 Understanding `next_batch()` in Stages

This corresponds to `frame_scheduler.cpp:95-126`:

```cpp
const auto deadline = Clock::now() + timeout;
while (batch.size() < maximum) {
```

It never waits forever merely to fill a batch. It returns after reaching the maximum batch size, all sources finish, or the timeout expires. Low traffic therefore does not cause enormous latency while waiting for a full batch.

```cpp
for (std::size_t checked = 0; checked < queues_.size(); ++checked) {
    const std::size_t index = (cursor_ + checked) % queues_.size();
    auto item = queues_[index]->try_pop();
```

- `cursor_` identifies the next round's starting point.
- `% queues_.size()` implements circular indexing.
- Once one frame is obtained from a stream, the code leaves the `for` loop and starts the next round. Frames from different sources are therefore interleaved under round-robin scheduling.

Additional latest-first logic:

```cpp
while (auto newer = queues_[index]->try_pop()) {
    item = std::move(newer);
    ++stale_;
}
```

The scheduler keeps removing frames until only the newest frame from that stream remains. Previously retrieved frames count as stale drops. This is not queue overflow, but it is still an explicit frame drop, so `evicted()` adds `stale_` to the sum of every queue's `evicted()` value.

```cpp
item->metadata.batch_index = batch.size();
batch.push_back(std::move(*item));
cursor_ = (index + 1) % queues_.size();
```

Before `push`, `batch.size()` is exactly the new frame's index in the batch. Updating the cursor makes the next call prefer the following stream.

When no frame is currently available, the implementation sleeps for 100 microseconds before checking again. This is simple, but less efficient than a condition variable shared across queues; it is a tradeoff between implementation complexity and scheduling efficiency.

### 8.4 Why `stop()` Must Always `join()`

If a `std::thread` object is destroyed while still `joinable()`, the program immediately calls `std::terminate()`. More importantly, destroying the backend, source, or queue while a capture thread still accesses it would cause a use-after-free. The destructor therefore also calls `stop(true)` as an RAII safety net.

## 9. Difficult Topic 3: The Slot State Machine and Asynchronous Ownership

The state transitions are:

```text
Free --reserve--> Reserved --successful submit--> Submitted
                                              │
                                   begin_collection
                                              ▼
                                        Completing --release--> Free

Any occupied state --error--> Failed
```

### 9.1 Why Slots Exist

A slot is an exclusive resource package for one batch that is about to execute or is currently executing. It includes:

- one `IExecutionContext`;
- one CUDA stream;
- a set of CUDA events;
- input, output, source, and letterbox device buffers;
- source and output pinned host buffers;
- metadata and timing for the current batch.

Sharing an `ICudaEngine` is safe and appropriate: the engine primarily stores the network and its optimization results. Mutable concurrent execution state belongs to `IExecutionContext`, so each slot creates a separate context.

### 9.2 Why State Checks Still Need a Mutex

“Check, then modify” must be an atomic transaction. If two threads both observe the same `Free` slot and occupy it, they overwrite the same resources. `SlotPool` uses one mutex for the entire slot array; `reserve()/try_reserve()` find a slot and change it to `Reserved` inside one critical section.

### 9.3 Why `release()` Unlocks Before Notifying

The source uses an extra scope to destroy the `lock_guard` before calling `available_.notify_one()`. If it notified while holding the lock, the awakened thread could immediately block on the same mutex. Unlocking first generally reduces useless contention. Both forms can be correct; this code chooses the common pattern.

### 9.4 Why `Failed` Does Not Automatically Return to `Free`

When submission fails partway through, the stream or buffer state might no longer satisfy the preconditions for reuse. The current process exits on failure and permanently marks the slot `Failed` rather than risking reuse of a partially submitted resource package.

## 10. Difficult Topic 4: A Minimal CUDA Knowledge Map

### 10.1 Host, Device, and Pinned Memory

- Ordinary `cv::Mat` pixels live in pageable host memory.
- `cudaMallocHost` allocates pinned (page-locked) host memory, from which the GPU can perform stable asynchronous DMA transfers.
- `cudaMalloc` allocates device memory, accessible only to the GPU or through CUDA APIs.
- For `cudaMemcpyAsync` to genuinely overlap CPU and GPU work, host-side memory normally needs to be pinned.

This lesson therefore copies a potentially non-contiguous, strided `cv::Mat` into `pinned_source` row by row before starting asynchronous H2D.

### 10.2 What Is a Stream?

A CUDA stream can be understood as a GPU command queue:

- Commands within one stream execute in enqueue order.
- Different non-blocking streams may overlap when hardware resources permit.
- A successful asynchronous CPU API call means only that a command was queued, not that the GPU finished it.

This lesson gives each slot one stream. It therefore does not need synchronization calls between H2D, resize, normalize, TensorRT, and D2H: ordering in the same stream enforces their data dependencies.

### 10.3 Events Have Two Uses

1. **Completion signal**: record `done` last, query it without blocking via `cudaEventQuery(done)`, or wait precisely for that slot via `cudaEventSynchronize(done)`.
2. **GPU timing**: record events around stages in the same stream, then use `cudaEventElapsedTime()` to compute GPU time.

The code does not call `cudaDeviceSynchronize()`, which would wait for all work on the device, destroy concurrency between slots, and potentially wait for unrelated CUDA work.

## 11. Difficult Topic 5: Understanding the `normalize` Kernel Line by Line

This corresponds to `tensorrt_backend.cu:74-85`:

```cpp
__global__ void normalize(const unsigned char* source,
                          float* destination,
                          int width, int height,
                          int batch_index) {
```

`__global__` means the CPU launches the function and many GPU threads execute it. `source` is the letterboxed image produced by NPP in BGR/HWC/uint8 layout; `destination` is TensorRT input in RGB/CHW/float layout.

```cpp
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
```

CUDA organizes threads as grid → block → thread. Each thread handles one `(x,y)` pixel:

- `blockIdx`: the current block's position in the grid;
- `blockDim`: dimensions of each block, 16×16 at launch here;
- `threadIdx`: the thread's position within the block.

```cpp
if (x >= width || y >= height) return;
```

The grid size is rounded up, so edge blocks contain threads outside the image and require this guard.

```cpp
const std::size_t pixel = y * width + x;
const std::size_t plane = width * height;
const std::size_t output = batch_index * 3 * plane;
```

- `pixel` is the pixel's linear index in a single channel plane.
- `plane` is the number of elements in one channel.
- `output` advances to the current batch element in NCHW layout.

```cpp
destination[output + pixel]           = source[pixel * 3 + 2] / 255.0F;
destination[output + plane + pixel]   = source[pixel * 3 + 1] / 255.0F;
destination[output + 2 * plane+pixel] = source[pixel * 3]     / 255.0F;
```

OpenCV/NPP source data stores neighboring `[B,G,R]` values for each pixel. The destination stores the complete R plane, then G, then B. Dividing by `255.0F` maps `[0,255]` to `[0,1]`; the `F` suffix ensures float arithmetic.

The launch is:

```cpp
normalize<<<
    dim3((width + 15) / 16, (height + 15) / 16),
    dim3(16, 16),
    0,
    slot.stream
>>>(...);
```

- The first argument is the grid, with width and height divided by 16 and rounded up.
- The second is the block: 256 threads per block.
- The third is dynamic shared-memory size in bytes, zero because none is used.
- The fourth selects the slot stream.
- A kernel launch is asynchronous. The following `cudaGetLastError()` primarily checks immediate errors such as invalid launch configuration; execution errors surface at a later synchronization point.

## 12. Difficult Topic 6: Backend Initialization and Memory Reuse

### 12.1 The `Impl` Constructor

The order in `tensorrt_backend.cu:154-189` is intentional:

1. `read_engine()` loads the engine into a host vector.
2. `createInferRuntime(logger)` creates the TensorRT runtime.
3. `deserializeCudaEngine()` deserializes the shared engine.
4. Iterate over I/O tensors and record the input and output names.
5. For every slot:
   - create a non-blocking stream;
   - create completion and stage-timing events;
   - create an NPP context tied to that stream;
   - create a separate `IExecutionContext` from the shared engine.

The PImpl design—the public class contains only a `unique_ptr<Impl>`—hides CUDA/TensorRT headers inside the `.cu` file, reduces header dependencies, and keeps resource details out of the public interface.

### 12.2 `ensure_capacity()`

It first computes `source_stride`, the largest source-image byte count in the current batch. Every image receives an equal-sized slot in the source/pinned buffer. Small images leave gaps, but address calculation remains simple and regions cannot overlap.

Replacement functions run only if some capacity is insufficient. They use this order:

```text
allocate new memory → release old memory after success → update pointer and capacity
```

This is safer than “free, then allocate”: if new allocation fails, the old buffer remains intact. Growth across several buffers is not one atomic transaction, however. If an intermediate allocation fails, buffers grown earlier remain enlarged until exception cleanup eventually releases them.

`cudaMalloc/cudaFree` may synchronize and can be expensive, so `capacity_growth_ms` is measured separately rather than disguised as steady-state inference time. It should become zero after fixed-shape execution stabilizes.

## 13. The Core Path: Understanding `submit()` in Stages

### 13.1 Input and Ownership Checks

`tensorrt_backend.cu:221-239` validates the slot index, non-empty batch, batch≤4, matching metadata count, `Reserved` state, and `CV_8UC3` format for every image.

These checks matter. If image and metadata counts differ, successful inference still cannot associate results correctly. Submitting without reserving could overwrite a slot that is already running.

### 13.2 Set Dynamic Shape and Determine Output Size

```cpp
const nvinfer1::Dims4 shape(batch, 3, 640, 640);
slot.context->setInputShape(input_name, shape);
const Dims output_shape = slot.context->getTensorShape(output_name);
```

A dynamic-batch engine must set the actual input shape on each context whenever the shape changes. The code then asks the context for the resolved output shape. A dimension `<=0` is still dynamic or invalid and cannot be used to allocate output.

Multiplying all output dimensions gives `output_elements`; multiplying that by `sizeof(float)` gives the byte count. Element count and byte count are different concepts.

### 13.3 Host Staging

```cpp
unsigned char* destination = staging + batch * source_stride;
for (int row = 0; row < image.rows; ++row) {
    std::memcpy(destination + row * row_bytes,
                image.ptr(row), row_bytes);
}
```

Do not assume that every `cv::Mat` is contiguous: an ROI or some decoded images can have extra step. Copying row by row with `image.ptr(row)` moves only valid pixels. The staged rows are tightly packed, so the later NPP source step is `image.cols * 3`.

### 13.4 H2D

The stream records `h2d_start`, asynchronously copies each image from its pinned slot to `device_source`, and records `h2d_end`. These events are also merely queued; submission does not read their elapsed time.

### 13.5 Letterbox + NPP Resize

For each image:

1. `cudaMemsetAsync(letterbox, 114, ...)` fills the complete destination image with gray value 114.
2. `scale = min(640/src_w, 640/src_h)` preserves aspect ratio while keeping both dimensions at or below 640.
3. Round to compute resized width and height.
4. Compute `pad_x/pad_y` to center the resized image.
5. Store `scale/pad/source dimensions` in that frame's metadata for inverse postprocessing.
6. Offset the `destination` pointer to the upper-left corner inside the padding.
7. `nppiResize_8u_C3R_Ctx()` writes the resized result directly into the inner letterbox region.

The address offset is:

```cpp
(pad_y * input_width + pad_x) * 3
```

It skips `pad_y` rows, then `pad_x` pixels, at three bytes per pixel.

The NPP destination step remains one full letterbox row, `640*3` bytes, not the resized width. Otherwise subsequent rows would be written to incorrect addresses.

### 13.6 Normalize, Bind TensorRT, Infer, and D2H

After NPP, `normalize` launches in the same stream with no explicit wait. Then:

```cpp
setTensorAddress(input_name, slot.input);
setTensorAddress(output_name, slot.output);
enqueueV3(slot.stream);
```

TensorRT is queued in the same stream, so it necessarily reads input after preprocessing completes. D2H is then queued after inference, followed by the `done` event.

Only after every CUDA command and the `done` event have been queued successfully does the code call:

```cpp
slot_pool.mark_submitted(index, std::move(metadata));
```

On an exception, it first calls `cudaStreamSynchronize(slot.stream)` so already queued work can no longer access resources, marks the slot `Failed`, and rethrows the original exception.

### 13.7 An Easy-to-Miss Metadata Detail

The `metadata` parameter is first copied into `slot.metadata`, and preprocessing writes the computed `transform` into `slot.metadata.frames[...]`. At the end, the original parameter `metadata` is moved into `SlotPool`, so its transform may still have the default value. The actual result comes from `slot.metadata` in `collect()`, however, so postprocessing receives the correct transform. The copy in `SlotPool` primarily enforces lifecycle and state constraints here; it is not the source of the final GPU-result metadata.

## 14. `ready()` and `collect()`: Why Both Querying and Waiting Exist

### 14.1 `ready()`

```cpp
cudaEventQuery(done)
```

It returns:

- `cudaErrorNotReady`: the GPU has not reached `done`; return `false` normally.
- `cudaSuccess`: work is complete.
- any other error: throw an exception.

It does not block, allowing the control loop to scan pending work and reclaim any slot that has already finished.

### 14.2 `collect()`

1. Perform `Submitted → Completing` to prevent duplicate collection by another caller.
2. Call `cudaEventSynchronize(done)` to wait only for this slot.
3. Copy pinned output into a `std::vector<float>`, making the returned result independent of reusable slot memory.
4. Preserve the output shape from the context.
5. Read stage timings from paired events.
6. Call `release()` to return the slot to `Free`.

If collection fails, the slot enters `Failed` and is not silently reused.

## 15. Understanding the `run_integrated_pipeline()` Control Loop

### 15.1 Construction Order Also Determines Destruction Order

Local objects are destroyed in reverse construction order. The backend is constructed first and the scheduler later, so the scheduler is destroyed before the backend during normal scope exit. Capture threads stop before GPU resources are released, matching their ownership dependency.

### 15.2 Why `PendingBatch` Keeps Images

Backend submission first copies image contents into pinned buffers, so the GPU no longer depends on the original `cv::Mat`. After collection, however, `ResultWriter::write()` still needs the source images to draw annotations. Pending state therefore retains `images` until postprocessing completes.

### 15.3 The Main `while` Exit Condition

```cpp
while (!scheduler.done() || !pending.empty())
```

Even after all sources and queues finish, every submitted batch must still be collected. This is normal end-of-stream draining of submitted work.

### 15.4 Submission Loop

```cpp
while (backend.available_slots() != 0 && !scheduler.done())
```

The code submits as much as possible while a slot is free and scheduler work remains. `next_batch()` waits up to 4 ms to form a batch, after which it computes:

- `batch_fill_ms`: host time spent in `next_batch()`;
- `queue_wait_ms`: average time across batch frames from `captured_at` until imminent submission.

It then reserves a slot, organizes images and metadata, calls `submit()`, and adds the slot to pending work.

Calling `try_reserve()` after `available_slots()` may seem redundant, but it explicitly validates an assumption: the complete application currently has only one submission thread, so reservation should succeed. If concurrent submission is added later, a race exists between the calls, and the exception exposes that the design assumption is no longer valid.

### 15.5 Collection Policy

The loop scans pending batches and immediately collects the first slot for which `ready()` succeeds. A slow batch at the front therefore does not block a later batch that has already completed.

If none is ready and pending work exists, it blocks while collecting the front item. Without this fallback, the control loop would spin while repeatedly querying events and consume an entire CPU core.

Results may consequently be written out of order. This is not a bug: every JSON record carries `(stream_id, frame_id, batch_id)`, and consumers must interpret identity rather than rely on file line order.

### 15.6 Exception Cleanup

```cpp
const std::exception_ptr causal = std::current_exception();
scheduler.stop(true);
for (const PendingBatch& batch : pending) {
    try { backend.collect(batch.slot); } catch (...) {}
}
std::rethrow_exception(causal);
```

The order means:

1. Save the original exception.
2. Reject new capture and discard queued work that was not submitted.
3. CUDA commands already submitted cannot be pretend-cancelled; collect/quiesce them one by one so no running GPU operation still uses resources.
4. Do not let a cleanup exception overwrite the original root cause.

### 15.7 Terminal Accounting

The real pipeline currently checks this identity at the end:

```text
captured == completed + evicted + aborted
```

- completed: frames that completed postprocessing;
- evicted: queue-overflow drops plus old frames discarded by latest-first;
- aborted: frames cleared at closure plus frames rejected because closure occurred during enqueue.

`Accounting::validate_terminal()` is a more fine-grained, CPU-testable model, but the current `run_integrated_pipeline()` does not use it directly. Do not assume that the real path already maintains every field in `Accounting` individually.

## 16. Postprocessing: Mapping Boxes Back to the Original Image

`ResultWriter::write()` first checks that batch metadata, source-image count, and output element count agree, then slices the output by `elements_per_image`.

Preprocessing applies:

```text
original coordinates --multiply by scale--> resized coordinates --add pad--> 640×640 model coordinates
```

Postprocessing must invert it:

```text
original coordinates = (model coordinates - pad) / scale
```

The code converts the backend's `Transform` into the Lesson 11 `LetterboxInfo`, then calls `decode_yolov8_output()` for YOLOv8 decoding, threshold filtering, NMS, and coordinate restoration.

Only the first `annotated_0.jpg` is saved, preventing a long run from continuously writing large numbers of images. `maximum_detection_records` can limit the number of detection JSONL records. After the limit is reached, frame latency is still measured, but detection records are no longer written.

## 17. Metrics: Which Times Can Be Combined and Which Cannot

### 17.1 Two Clock Domains

- Host: `std::chrono::steady_clock`, used for queuing, batch formation, host staging, CPU postprocessing, and end-to-end latency.
- GPU: CUDA events in the same slot stream, used for H2D, preprocessing, TensorRT, and D2H.

Do not use CPU time immediately before and after `enqueueV3()` as GPU inference time, because enqueue normally only queues work.

### 17.2 Latency and Throughput Are Different

- FPS = completed frame count / total wall-clock time.
- Frame latency = time when `ResultWriter` processes a frame − `captured_at`.
- A batch stage time belongs to the whole batch. Simply dividing it by batch size does not make it per-frame end-to-end latency.

### 17.3 Bounded-Memory Statistics

`latencies_` keeps at most 8,192 recent values and `batches_` at most 256 representative batches. Complete batch timing is streamed immediately to `batch_timing_samples.jsonl` for every batch. Long executions therefore do not grow memory continuously by retaining all statistics in vectors.

A comment calls this a deterministic rolling reservoir. More precisely, it is a **fixed-size circular overwrite/recent window**, not classic random reservoir sampling. Final percentiles primarily represent the retained recent samples, not a strictly unbiased sample of the complete history.

### 17.4 A Boundary in the Current Metrics Implementation

`PipelineMetrics::submitted_` increases together with `completed_` in `record_batch()`, which is called only after collection and postprocessing succeed. In runtime `metrics_snapshots.jsonl`, `submitted` is therefore not the true number of batches submitted but possibly still incomplete; it normally advances in lockstep with completed. Equality is fine for a successful final run, but the value cannot reveal real-time in-flight work. Actual in-flight work can be inferred from `slot_count - available_slots` or the pending size.

## 18. Source Details and Limitations That Must Be Stated Directly

### 18.1 Whether Capture-to-Result Latency Includes Video Decode

The comment in `frame_source.hpp` says that because “the timestamp is taken after read succeeds,” capture-to-result latency includes video decoding time. That does not follow logically. The actual order in `frame_scheduler.cpp` is:

```cpp
sources_[stream]->read(image); // decoding already happened
Clock::now();                  // timestamp taken afterward
```

Consequently, the current `captured_at → result` interval **does not include the current `read()` call's decoding time**. The later wording in this lesson's `README.md`—“timestamp is assigned after successful decode; latency starts at admission”—does match the code. Follow execution order when reading it.

### 18.2 Multi-Source `frame_count` Distribution Can Round Down

Ordinary mode uses:

```cpp
frames_per_source = max(1, frame_count / source_count);
```

With `frame_count=5` and two sources, each source processes two frames, for only four in total. If source count exceeds frame count, each source still processes at least one frame and the total exceeds `frame_count`. In multi-source mode, this parameter is therefore closer to “a target total used for equal distribution” and does not guarantee that the final count exactly matches the input value.

### 18.3 I/O Tensor Assumptions Are Looser Than the Error Message

The constructor iterates over all I/O, retaining the last input name and output name observed, then checks only that neither is empty. The error says “expected exactly one input and one output,” but the code does not explicitly verify that the counts are exactly one each. The assumption holds for this lesson's YOLO engine. A multi-input or multi-output engine requires explicit counts and deliberate tensor selection.

### 18.4 `metrics.json` Is Handwritten JSON

GPU names, policy names, and other strings are written directly to the stream without general JSON escaping. Common NVIDIA GPU names do not expose the issue, but this is not a JSON library capable of handling arbitrary strings.

Not every limitation necessarily needs to be fixed in this lesson immediately, but understanding the boundaries is more important than treating teaching code as an unconditionally complete production framework.

## 19. Why Error Injection Is Worth Studying

These environment variables are not business functionality. They force hard-to-reach cleanup paths:

- `LESSON21_FAIL_SOURCE_FRAME`: can a capture-thread exception propagate to the main thread?
- `LESSON21_FAIL_SUBMIT_BATCH`: can failure before or during submission stop the pipeline?
- `LESSON21_FAIL_INSUFFICIENT_CAPACITY`: force capacity preparation failure.
- `LESSON21_FAIL_TENSOR_ADDRESS`: force tensor binding failure.
- `LESSON21_FAIL_ENQUEUE`: force TensorRT enqueue failure.
- `LESSON21_FAIL_POSTPROCESS_BATCH`: make CPU postprocessing fail after the GPU finishes.
- `LESSON21_ABORT_AFTER_SUBMISSIONS`: abort while GPU work is already in flight.

Production code cannot validate only the success path. In asynchronous GPU programs especially, “the CPU threw an exception” does not mean the GPU stopped running. Cleanup order directly determines whether the process encounters a use-after-free or hangs.

## 20. What the Tests Prove

### 20.1 CPU Core Tests

`test_pipeline_core.cpp` checks:

- two slots are never allocated twice;
- `try_reserve()` is empty when every slot is occupied;
- reverse-order collection preserves the correct identity;
- a `Failed` slot cannot be released;
- drop-oldest retrieval results and counts;
- `close(false)` drains, while `close(true)` discards;
- the accounting identity rejects inconsistent counts.

### 20.2 Scheduler Tests

`test_frame_scheduler.cpp` checks unique multi-source identity, `batch_index`, queue peaks, and source-exception propagation. To remain compact, the test places more code on individual lines than desirable; it is not a C++ formatting example, but its test intent is valid.

### 20.3 GPU Tests

- batch 1/2/4 and two-slot smoke tests;
- batch 5, beyond the dynamic profile, must fail;
- output identity, environment, stage timings, and batch distribution;
- batch-1 and batch-4 detections are approximately consistent;
- overload/latest-first must explicitly count dropped frames;
- longer runs retain bounded statistics and stop growing capacity in steady state;
- multiple error-injection paths must exit nonzero without hanging.

GPU tests require the dynamic engine generated in Lesson 17, an NVIDIA GPU, and the specified container. Passing CPU tests cannot substitute for that evidence.

## 21. Recommended Hands-On Code Study Sequence

### Exercise 1: Draw States Without Running the GPU

Trace `main.cpp` on paper:

```text
slot 0: Free → Reserved → Submitted ─────────────→ Completing → Free
slot 1: Free → Reserved → Submitted → Completing → Free
result order: stream 1, then stream 0
```

Question: why are results not associated with the wrong streams? Because stream/frame identity lives in batch metadata.

### Exercise 2: Simulate Drop-Oldest

With capacity 2, push 1, 2, and 3 in order:

```text
[1] → [1,2] → remove 1 → [2,3]
evicted=1, peak=2
```

Compare this with `test_pipeline_core.cpp:34-38`.

### Exercise 3: Simulate a Round-Robin Batch

Assume Q0 contains `a0,a1`, Q1 contains `b0,b1`, `cursor=0`, and `maximum=3`. The result should resemble `a0,b0,a1`, with the next cursor starting at Q1. Then switch to latest-first and observe why each stream's older frames count as stale.

### Exercise 4: Run Only the Backend Smoke Test

Do not start with the complete pipeline. Follow this lesson's README to run `integrated_tensorrt_gpu_smoke` with batch 1 in the container, then batch 4 and `--two-slots`. Focus on output element count, each GPU stage time, and reverse-order collection.

### Exercise 5: Draw a Slot Timeline

```text
slot.stream:
| H2D | memset+NPP+kernel | TensorRT | D2H | done |

CPU:
submit returns ---------------------- ready? ------- collect
```

Add a second slot stream and consider which intervals on the two rows can overlap.

### Exercise 6: Observe Overload Policies

Run both block and drop-oldest. Do not compare only FPS; also compare:

- `captured/processed/dropped`;
- `queue_peak`;
- p50/p90/p99;
- batch distribution.

Block tends to avoid drops by propagating backpressure to capture. Drop-oldest tends to preserve freshness at the cost of completeness. Neither policy is “better” for every application.

## 22. Final Mastery Checklist

After reading and experimenting, you should be able to answer independently:

1. Why must queues have a capacity limit?
2. How do `close(false)` and `close(true)` differ?
3. Why does multi-source scheduling use `try_pop()` instead of calling `pop()` on one stream?
4. Why do `stream_id + frame_id` identify a frame, while a slot does not?
5. Why is the engine shared while every slot exclusively owns its context, stream, and buffers?
6. What role does pinned host memory play in asynchronous H2D/D2H?
7. Why do NPP, the kernel, and TensorRT in one stream not need a CPU synchronization between each stage?
8. Which of `ready()` and `collect()` is a non-blocking query, and which is a blocking wait?
9. Why cannot the CPU call duration of `enqueueV3()` represent GPU inference time?
10. Why must letterbox `scale/pad_x/pad_y` be preserved in per-frame metadata?
11. When an exception occurs, why stop capture before waiting for already-submitted CUDA work?
12. Why should `captured == completed + evicted + aborted` hold at normal termination?

If you can explain these answers without looking them up, you understand the main body of Lesson 21. CUDA kernel optimization details can come later. The more important outcome of this lesson is establishing four engineering models: **asynchronous execution, resource ownership, backpressure, and verifiable accounting**.
