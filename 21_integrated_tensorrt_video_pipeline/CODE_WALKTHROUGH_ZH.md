# 第 21 课代码导读：从 C++ 调度到 CUDA/TensorRT 推理

> 适合读者：C++ 基础较薄弱、刚接触 CUDA，希望先看清全局，再深入关键代码。


## 1. 先明确这一课到底做什么

这一课不是“调用一次 TensorRT 做一张图”的示例，而是把一条较完整的视频推理流水线拼起来：

```text
一个或多个图像/视频源
        │（每个源一个采集线程）
        ▼
每个源自己的有界队列 BoundedQueue
        │（轮询或取最新帧）
        ▼
FrameScheduler 拼动态 batch
        │
        ▼
TensorRtBackend 申请空闲 slot
        │
        ├─ CPU 图像复制到 pinned host memory
        ├─ H2D：主机内存 → GPU 内存
        ├─ NPP resize + letterbox
        ├─ CUDA kernel：BGR/HWC/uint8 → RGB/CHW/float
        ├─ TensorRT enqueueV3
        └─ D2H：推理输出 → pinned host memory
        │
        ▼
CPU YOLO 解码、NMS、写 JSONL 和标注图
        │
        ▼
汇总吞吐、延迟、丢帧和环境信息
```

真正的重点不是某一个 API，而是下面四个工程问题：

1. **背压**：采集速度比推理快时，内存不能无限增长。
2. **并发所有权**：多个 CUDA stream 可以并行，但不能错误共享 execution context 或工作缓冲区。
3. **异步身份**：batch 后提交的不一定后完成，结果身份不能按完成顺序猜。
4. **可解释的统计**：捕获、丢弃、提交和完成必须能对账，CPU 时间与 GPU 时间不能混为一谈。

## 2. 建议阅读顺序

不要按文件名从头到尾硬啃。建议分五轮：

1. `include/pipeline_core.hpp` + `src/pipeline_core.cpp`：先理解数据身份、队列和 slot 状态机。
2. `include/frame_source.hpp` + `src/frame_source.cpp`：理解帧从哪里来。
3. `include/frame_scheduler.hpp` + `src/frame_scheduler.cpp`：理解多线程如何产生 batch。
4. `src/integrated_pipeline.cpp`：看总控循环，先把各模块串起来。
5. `src/tensorrt_backend.cu`：最后深入 CUDA/NPP/TensorRT；再看 `result_writer.cpp` 和 `metrics.cpp`。

两个入口要区分：

- `src/main.cpp` 生成 `integrated_tensorrt_video_pipeline`，只是 **CPU slot/身份测试程序**，不做真实推理。
- `src/pipeline_app.cpp` 生成 `integrated_tensorrt_video_pipeline_gpu`，才是完整的 GPU 流水线入口。

## 3. 读这份代码前需要的最少 C++ 概念

### 3.1 值、引用和移动

- `const T&`：只读借用，不复制 `T`。
- `T&`：可修改借用。
- `T value`：函数获得一个独立对象；调用处可能复制，也可能移动。
- `std::move(x)`：它本身不移动数据，只把 `x` 转成“允许被移动”的右值，随后由移动构造/赋值接管资源。移动后的 `x` 仍可析构，但不要再假定它保留原内容。

本课频繁移动 `std::vector`、`std::unique_ptr` 和 metadata，目的是转交所有权，避免大对象复制。

`cv::Mat` 比较特殊：普通赋值通常只复制一个带引用计数的“矩阵头”，像素缓冲区仍共享；`clone()` 才深拷贝像素。`FrameSource` 返回的图像在进入 GPU 前只读，因此这里可以安全共享。

### 3.2 `unique_ptr` 与 RAII

`std::unique_ptr<T>` 表示唯一所有权：不能复制，只能移动。对象离开作用域时自动析构。这是 RAII（资源获取即初始化）的典型用法。

CUDA 的裸资源并不是 C++ 对象，因此 `tensorrt_backend.cu` 中的 `Slot::~Slot()` 手动销毁 device buffer、pinned buffer、event 和 stream。外层用 `unique_ptr<Slot>` 管理 `Slot`，最终形成完整的自动清理链。

### 3.3 `optional`

`std::optional<std::size_t>` 表示“可能有一个 slot 编号，也可能没有”。

```cpp
const auto reserved = backend.try_reserve();
if (!reserved) { /* 没有空闲 slot */ }
backend.submit(*reserved, ...); // * 取出其中的编号
```

它比用 `-1` 表示失败更安全，因为 `std::size_t` 是无符号类型，`-1` 很容易产生隐蔽错误。

### 3.4 mutex、条件变量和 atomic

- `std::mutex` 保护一组需要整体一致的数据，例如队列内容和 `closed_`。
- `std::lock_guard` 在作用域结束时自动解锁，适合不需要等待的短临界区。
- `std::unique_lock` 可以暂时解锁/重新加锁，是 `condition_variable::wait` 所需类型。
- `std::condition_variable` 让线程睡眠等待“非空”或“非满”，避免空转烧 CPU。
- `std::atomic` 适合简单计数/标志，但不能替代对复合状态的 mutex 保护。

### 3.5 `try/catch (...)` 和 `exception_ptr`

`catch (...)` 捕获任何异常。`std::current_exception()` 把当前异常保存为 `exception_ptr`，之后可用 `std::rethrow_exception()` 保留原类型和原消息重新抛出。本课用它把采集线程中的异常传给主控线程，也用它在清理期间保留“第一个根因”。

## 4. 目录和构建目标

### 4.1 头文件的职责

| 文件 | 主要内容 |
|---|---|
| `include/pipeline_core.hpp` | metadata、对账结构、slot 状态机、有界队列 |
| `include/frame_source.hpp` | 帧源抽象和工厂函数 |
| `include/frame_scheduler.hpp` | 多源采集、队列和动态 batch 调度 |
| `include/config.hpp` | 命令行配置 |
| `include/tensorrt_backend.hpp` | GPU 后端对外接口和返回结果 |
| `include/result_writer.hpp` | 后处理结果写出 |
| `include/metrics.hpp` | 指标采集与汇总 |
| `include/integrated_pipeline.hpp` | 完整流水线的单一入口 |

### 4.2 CMake 目标

`CMakeLists.txt` 有意把可测试逻辑拆开：

- `integrated_pipeline_core`：纯 C++ 队列/slot/身份逻辑。
- `integrated_frame_scheduler`：帧源和调度器，依赖 OpenCV。
- `integrated_tensorrt_backend`：CUDA、NPP、TensorRT 后端。
- `integrated_yolo_postprocess`：复用第 11 课的 YOLO 前后处理代码；本课实际只在结果写出中调用后处理。
- `integrated_tensorrt_video_pipeline`：CPU 身份冒烟程序。
- `integrated_tensorrt_gpu_smoke`：只测试 GPU backend，不包含完整调度器。
- `integrated_tensorrt_video_pipeline_gpu`：完整应用。

这比把所有 `.cpp/.cu` 塞进一个可执行程序更容易独立测试，也明确了依赖方向。

## 5. 第一轮：函数级别走完整个项目

这一节只回答“每个函数负责什么”，暂时不钻进同步细节。

### 5.1 配置：`config.cpp`

- `positive(value, name)`：字符串转正整数，拒绝 0 和负数。
- `usage()`：返回完整命令格式。
- `parse_config(argc, argv)`：先解析固定位置参数，再解析 `--xxx` 选项，并验证 batch 上限和 duration 模式。

这里的解析规则是：只要当前位置参数首字符不是 `-`，就当作下一个位置参数。因此它是轻量教学实现，不是通用 CLI 框架。

### 5.2 帧源：`frame_source.cpp`

- `ImageSequenceSource::read()`：在内存图片序列上循环取图，达到规定帧数后返回 `false`。
- `VideoFileSource::read()`：每次调用只解码一帧；到文件尾时根据 `repeat_` 决定结束还是重开视频。
- `make_repeatable_image_source()`：把单张图包装成长度为 1 的序列。
- `make_image_sequence_source()`：创建图片序列源。
- `make_synthetic_source()`：创建固定颜色的 640×480 测试图源。
- `make_path_source()`：按 `synthetic`、`sequence:...`、普通图片、视频的顺序识别输入。

`FrameSource` 是一个抽象基类：`virtual bool read(...) = 0` 是纯虚函数，派生类必须实现。调用者只依赖接口，不需要知道输入是图片还是视频。

### 5.3 调度：`frame_scheduler.cpp`

- 三个 `FrameScheduler` 构造函数：把不同形式的输入最终统一成 `vector<unique_ptr<FrameSource>>`。
- `start()`：每个 source 启动一个采集线程。
- `capture(stream)`：循环读帧、打 metadata、推入该 source 的有界队列。
- `next_batch(maximum, timeout)`：从多个队列取帧，组成最多 `maximum` 张的 batch。
- `stop(discard)`：关闭队列、唤醒线程、`join()` 所有采集线程。
- `rethrow_source_error()`：在主线程重新抛出采集线程异常。
- `evicted()/discarded()/queue_peak()/queue_depth()/done()`：汇总各队列状态。

### 5.4 CPU 核心：`pipeline_core.cpp`

- `Accounting::validate_terminal()`：检查终态计数恒等式。
- `SlotPool::reserve()`：阻塞等待空闲 slot。
- `SlotPool::try_reserve()`：不等待，立即返回空闲 slot 或空值。
- `mark_submitted()`：`Reserved → Submitted`，并保存不可变 batch 身份。
- `begin_collection()`：`Submitted → Completing`。
- `release()`：`Completing → Free`，清除 metadata 并通知等待者。
- `fail()`：把已占用 slot 标成 `Failed`，禁止错误复用。
- `IdentityDispatcher::dispatch()`：验证 `batch_index` 后按 metadata 写入结果，不猜测完成顺序。

### 5.5 GPU 后端：`tensorrt_backend.cu`

- `Logger::log()`：只打印 TensorRT warning 及以上日志。
- `check_cuda()/check_npp()`：把错误码转成 C++ 异常。
- `read_engine()`：二进制读取序列化 engine。
- `npp_context()`：把当前 CUDA 设备属性和某个 stream 组装成 NPP context。
- `normalize<<<...>>>()`：逐像素完成 BGR→RGB、HWC→CHW、uint8→float/255。
- `replace_device_buffer()/replace_pinned_buffer()`：容量不足才重新分配，稳定阶段复用内存。
- `Slot::~Slot()`：等待本 slot 的 stream，再按资源类型释放。
- `TensorRtBackend::Impl::Impl()`：反序列化 engine；为每个 slot 创建独立 stream、events 和 execution context。
- `ensure_capacity()`：按当前 batch 最大源图和 TensorRT I/O 大小扩充缓存。
- `submit()`：把一整个 batch 的 H2D、预处理、推理、D2H 异步排进同一个 slot stream。
- `ready()`：非阻塞查询 `done` event。
- `collect()`：等待指定 slot 完成，把 pinned output 拷进 C++ vector，读取各阶段耗时，再释放 slot。
- `identity()`：读取 GPU、compute capability、CUDA 和 TensorRT 版本。

### 5.6 总控：`integrated_pipeline.cpp`

- `sources(config)`：把配置中的路径创建成帧源；普通模式给每个源分配帧数，时长模式允许持续读取。
- `run_integrated_pipeline(config)`：创建全部模块，启动 scheduler，循环提交和收集 batch，最后写 metrics。

### 5.7 后处理和指标

- `ResultWriter::write()`：按 batch 切 TensorRT 输出；复用第 11 课 YOLO decode/NMS；把框映射回原图；写检测 JSONL 和第一张标注图；记录端到端延迟。
- `PipelineMetrics::record_batch()`：累计阶段时间、batch 分布和每路延迟，并把每个 batch 的原始样本流式写入 JSONL。
- `PipelineMetrics::write()`：生成最终 `metrics.json`。

### 5.8 两个小入口

- `main.cpp`：占两个 CPU slot，故意按相反顺序回收，证明 identity 随 metadata 而不是随完成位置走。
- `gpu_smoke.cpp`：读取一张图，复制成 batch，直接调用 backend；`--two-slots` 时先提交两份，再逆序收集。

## 6. 第二轮：一帧数据实际经历了什么

假设命令输入两个 source、`BATCH=4`、`SLOTS=2`：

1. `pipeline_app.cpp:main()` 调 `parse_config()`。
2. `run_integrated_pipeline()` 创建一个共享 engine，以及两个 slot。每个 slot 有自己的 context、stream、event 和内存。
3. `scheduler.start()` 启动两个采集线程。
4. 采集线程各自调用 `FrameSource::read()`，成功后创建：

   ```text
   ScheduledFrame
   ├─ image: cv::Mat
   └─ metadata
      ├─ stream_id：来自哪一路
      ├─ frame_id：该路第几帧
      ├─ batch_index：进入 batch 后才填写
      └─ captured_at：采集成功后的单调时钟时间
   ```

5. 每路帧进入自己的 `BoundedQueue`。满时要么阻塞，要么弹掉最老帧。
6. 主线程调用 `next_batch(4, 4ms)`，轮流从两路拿帧，最多凑 4 张。
7. 主线程申请 slot，生成 `batch_id`，计算队列等待时间，然后调用 `backend.submit()`。
8. `submit()` 只是把 GPU 工作依次排入这个 slot 的 stream；除内存增长等少数同步操作外，并不等待整条 GPU 链执行结束。
9. slot 进入 `Submitted`，主线程还能使用另一个 slot 提交下一批。
10. 主线程先用 `ready()` 找已经完成的任意 slot。若都未完成，为避免纯轮询，就对 `pending.front()` 调阻塞式 `collect()`。
11. `collect()` 等 `done` event，拿回输出和 metadata，然后 slot 变回 `Free`。
12. `ResultWriter` 用 metadata 中的 `transform` 把检测框还原到源图坐标，并用 `(stream_id, frame_id)` 写结果身份。

要点：**slot 编号是可复用的执行资源编号，不是 frame 或 batch 的身份。**