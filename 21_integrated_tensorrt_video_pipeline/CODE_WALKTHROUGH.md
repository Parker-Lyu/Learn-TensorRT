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