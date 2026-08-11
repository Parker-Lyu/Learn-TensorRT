# 第 21 课代码导读：从 C++ 调度到 CUDA/TensorRT 推理

> 适合读者：C++ 基础较薄弱、刚接触 CUDA，希望先看清全局，再深入关键代码。
>
> 本文只解释当前仓库中的实现，不替代本课 `README.md` 的构建、运行和验收说明。文中行号以本文编写时的源码为准；以后源码变化时，应优先按函数名定位。

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

## 7. 重难点一：有界队列逐行理解

核心位于 `pipeline_core.hpp:89-136`。

### 7.1 构造

```cpp
BoundedQueue(std::size_t capacity, OverloadPolicy policy)
    : capacity_(capacity), policy_(policy) {
    if (capacity == 0) throw std::invalid_argument(...);
}
```

- `capacity_`、`policy_` 是 `const`，构造后不再变化。
- 容量 0 会让“非满”永远不成立，所以直接拒绝。

### 7.2 `push()`：生产者路径

```cpp
std::unique_lock<std::mutex> lock(mutex_);
```

从这里开始，同一时刻只有一个线程能观察/修改 `values_`、`closed_` 和统计量。

```cpp
if (policy_ == OverloadPolicy::Block) {
    not_full_.wait(lock, [this] {
        return closed_ || values_.size() < capacity_;
    });
    if (closed_) return false;
}
```

- block 策略下，队列满则生产者睡眠。
- `wait` 内部会解锁 mutex，让消费者能够 `pop()`；被唤醒后再加锁。
- 谓词必须同时检查 `closed_`，否则关闭一个满队列时生产者可能永远醒不过来。
- `wait` 可能“伪唤醒”，谓词会重新检查条件。

```cpp
else if (values_.size() == capacity_) {
    values_.pop_front();
    ++evicted_;
}
```

drop-oldest 策略不阻塞；队列满就丢最旧帧。这降低实时画面的“陈旧度”，代价是结果不再覆盖每一帧。

```cpp
if (closed_) return false;
values_.push_back(std::move(value));
peak_ = std::max(peak_, values_.size());
not_empty_.notify_one();
return true;
```

- 关闭后拒绝新值。
- 值被移动进 deque。
- `peak_` 记录历史单路峰值。
- 唤醒一个等待“非空”的消费者。

### 7.3 `pop()` 与 `try_pop()`

`pop()` 会等待“关闭或非空”；关闭但仍有值时仍然把剩余值排空，这就是正常 EOS drain 的基础。只有关闭且空才返回 `nullopt`。

`try_pop()` 从不等待。scheduler 要轮询多路队列，若在第一路上阻塞，就可能看不到第二路已经就绪的帧，因此这里必须使用非阻塞版本。

两者弹出数据后都调用 `not_full_.notify_one()`，让 block 模式下的生产者继续。

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

- `discard=false`：不再接收新帧，但消费者可以排空已有帧，适合正常结束。
- `discard=true`：立即清掉尚未提交的帧，适合异常中止或 duration 到期。
- 两边都 `notify_all()`，因为生产者和消费者都可能正睡着。

注意：`close()` 是不可逆操作。本类没有 reopen，符合一次流水线生命周期。

## 8. 重难点二：FrameScheduler 的线程模型

### 8.1 为什么每个 source 一个队列

如果所有源共用一个队列，快源可能把慢源完全淹没。当前实现“一源一线程、一源一队列”，`next_batch()` 再轮询各路，更容易实现公平和逐路统计。

### 8.2 `capture()` 逐段解释

对应 `frame_scheduler.cpp:64-93`：

```cpp
std::uint64_t frame_id = 0;
cv::Mat image;
while (!stopping_) {
```

每个采集线程有自己的 `frame_id`，因此全局身份必须是 `(stream_id, frame_id)`。`stopping_` 是 atomic，使 stop 线程和采集线程之间没有普通 bool 的数据竞争。

```cpp
if (!sources_[stream]->read(image)) break;
if (image.empty()) throw ...;
ScheduledFrame item{
    image,
    {stream, frame_id++, 0, Clock::now(), {}}
};
++captured_;
```

- `read=false` 表示正常 EOS。
- metadata 初始化使用聚合初始化，字段顺序对应 `FrameMetadata` 定义。
- `batch_index` 先填 0，因为尚未组 batch。
- 时间戳在 `read()` 成功之后生成。
- `captured_` 在尝试入队前增加，因此若此时队列被关闭，后面要计入 `rejected_on_close_` 才能对账。

```cpp
if (!queues_[stream]->push(std::move(item))) {
    ++rejected_on_close_;
    break;
}
```

`false` 只表示队列已经关闭。帧已被读出但未获准进入队列，所以单独计数。

异常路径先把第一个异常保存到 `source_error_`，再关闭所有队列。`source_error_` 由 mutex 保护，因为可能多个采集线程同时失败。最后无论正常或异常，本路都 `close(false)` 并增加 `finished_`。

### 8.3 `next_batch()` 逐段解释

对应 `frame_scheduler.cpp:95-126`：

```cpp
const auto deadline = Clock::now() + timeout;
while (batch.size() < maximum) {
```

它不会为了凑满 batch 无限等：达到最大 batch、所有源结束，或者超时都会返回。这样在低流量时不至于因等待满批导致巨大延迟。

```cpp
for (std::size_t checked = 0; checked < queues_.size(); ++checked) {
    const std::size_t index = (cursor_ + checked) % queues_.size();
    auto item = queues_[index]->try_pop();
```

- `cursor_` 是下一轮起点。
- `% queues_.size()` 实现环形索引。
- 一次从某一路拿到一帧就跳出 for，再开始下一轮，所以 round-robin 下不同源会交错进入 batch。

latest-first 的额外逻辑：

```cpp
while (auto newer = queues_[index]->try_pop()) {
    item = std::move(newer);
    ++stale_;
}
```

不断弹到该路最新一帧，之前弹出的都算 stale 丢弃。这里不是“队列自己溢出”，但同样属于明确丢帧，因此 `evicted()` 把 `stale_` 和各队列 `evicted()` 相加。

```cpp
item->metadata.batch_index = batch.size();
batch.push_back(std::move(*item));
cursor_ = (index + 1) % queues_.size();
```

在 push 之前，`batch.size()` 正好是新帧在 batch 中的下标。更新 cursor 确保下一次优先看后一条流。

暂时无帧时，这份实现每 100 微秒睡一次再检查。它简单，但不如跨多个队列共用条件变量高效；这是实现复杂度与调度效率之间的取舍。

### 8.4 `stop()` 为什么一定要 `join()`

`std::thread` 对象析构时若仍 `joinable()`，程序会直接 `std::terminate()`。更重要的是，如果 backend、source 或队列已析构而采集线程还在访问它们，会产生 use-after-free。因此析构函数也调用 `stop(true)`，形成兜底 RAII。

## 9. 重难点三：slot 状态机与异步所有权

状态流转是：

```text
Free --reserve--> Reserved --submit成功--> Submitted
                                      │
                         begin_collection
                                      ▼
                                Completing --release--> Free

任何已占用状态 --发生错误--> Failed
```

### 9.1 为什么需要 slot

一个 slot 是“一批正在执行或即将执行的工作所独占的资源包”。它包括：

- 一个 `IExecutionContext`；
- 一个 CUDA stream；
- 一组 CUDA events；
- input/output/source/letterbox device buffer；
- source/output pinned host buffer；
- 当前 batch 的 metadata 和计时。

共享 `ICudaEngine` 是安全且合理的：engine 主要保存网络和优化结果。并发执行的可变状态属于 `IExecutionContext`，所以每个 slot 单独创建 context。

### 9.2 为什么状态检查仍要 mutex

“先检查再修改”必须是原子事务。例如两个线程同时看到一个 Free slot 并同时占用，就会覆盖同一批资源。`SlotPool` 用一个 mutex 保护整个 slot 数组，`reserve()/try_reserve()` 在同一临界区查找并改为 Reserved。

### 9.3 `release()` 为什么先解锁再 notify

源码用额外作用域先让 `lock_guard` 析构，再 `available_.notify_one()`。如果持锁通知，被唤醒线程马上又会卡在同一 mutex 上；先解锁通常减少无意义竞争。正确性两种写法都可实现，这里选择更常见的方式。

### 9.4 Failed 为什么不自动回到 Free

提交中途失败时，slot 内的 stream 或缓存状态可能不再满足复用前提。当前进程选择失败退出，宁可把 slot 永久标成 Failed，也不冒险复用“半提交”资源。

## 10. 重难点四：CUDA 最小知识地图

### 10.1 host、device、pinned memory

- 普通 `cv::Mat` 像素在 host pageable memory。
- `cudaMallocHost` 分配 pinned（页锁定）host memory。GPU DMA 可以稳定地从这里异步传输。
- `cudaMalloc` 分配 device memory，只能由 GPU 或 CUDA API 访问。
- `cudaMemcpyAsync` 要想真正与 CPU/GPU 工作重叠，host 端通常需要 pinned memory。

因此本课先把可能不连续、带行步长的 `cv::Mat` 逐行整理进 `pinned_source`，再异步 H2D。

### 10.2 stream 是什么

可以把 CUDA stream 理解为 GPU 命令队列：

- 同一 stream 中，命令按入队顺序执行。
- 不同 non-blocking stream 在硬件资源允许时可重叠。
- CPU 调用异步 API 成功，只说明命令已入队，不等于 GPU 已做完。

本课每个 slot 一个 stream。于是无需在 H2D、resize、normalize、TensorRT、D2H 之间到处同步；同 stream 的顺序天然保证数据依赖。

### 10.3 event 有两个用途

1. **完成信号**：最后记录 `done`，`cudaEventQuery(done)` 非阻塞查询，`cudaEventSynchronize(done)` 精确等待该 slot。
2. **GPU 计时**：在同一 stream 的阶段前后记录 event，再用 `cudaEventElapsedTime()` 计算 GPU 时间。

不用 `cudaDeviceSynchronize()`，因为它会等待整个设备上的工作，破坏不同 slot 的并发，也可能等待与本流水线无关的 CUDA 工作。

## 11. 重难点五：`normalize` kernel 逐行解释

对应 `tensorrt_backend.cu:74-85`：

```cpp
__global__ void normalize(const unsigned char* source,
                          float* destination,
                          int width, int height,
                          int batch_index) {
```

`__global__` 表示函数由 CPU 发起、在 GPU 上由很多线程执行。`source` 是 NPP 产生的 letterbox 图，布局为 BGR/HWC/uint8；`destination` 是 TensorRT input，目标布局为 RGB/CHW/float。

```cpp
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
```

CUDA 把线程组织成 grid → block → thread。这里每个线程负责一个 `(x,y)` 像素：

- `blockIdx`：当前 block 在 grid 中的位置；
- `blockDim`：每个 block 的尺寸，这里 launch 时是 16×16；
- `threadIdx`：线程在 block 内的位置。

```cpp
if (x >= width || y >= height) return;
```

grid 尺寸用向上取整，边缘 block 会有线程落在图像外，必须保护。

```cpp
const std::size_t pixel = y * width + x;
const std::size_t plane = width * height;
const std::size_t output = batch_index * 3 * plane;
```

- `pixel` 是该像素在单通道平面里的线性下标。
- `plane` 是一个通道的元素数。
- `output` 跳到 NCHW 布局中当前 batch 的起点。

```cpp
destination[output + pixel]           = source[pixel * 3 + 2] / 255.0F;
destination[output + plane + pixel]   = source[pixel * 3 + 1] / 255.0F;
destination[output + 2 * plane+pixel] = source[pixel * 3]     / 255.0F;
```

OpenCV/NPP 源数据每个像素相邻存 `[B,G,R]`。目标先放完整 R 平面，再 G，再 B。除以 `255.0F` 把 `[0,255]` 映射到 `[0,1]`；后缀 `F` 保证是 float 运算。

launch：

```cpp
normalize<<<
    dim3((width + 15) / 16, (height + 15) / 16),
    dim3(16, 16),
    0,
    slot.stream
>>>(...);
```

- 第一项是 grid：对宽高做除 16 的向上取整。
- 第二项是 block：每块 256 个线程。
- 第三项是动态 shared memory 字节数，这里不用所以为 0。
- 第四项指定 slot stream。
- kernel launch 本身异步；紧接的 `cudaGetLastError()` 主要检查 launch 配置等即时错误，运行期错误会在之后同步点暴露。

## 12. 重难点六：backend 初始化与内存复用

### 12.1 `Impl` 构造函数

`tensorrt_backend.cu:154-189` 的顺序不能随意颠倒：

1. `read_engine()` 把 engine 放入 host vector。
2. `createInferRuntime(logger)` 创建 TensorRT runtime。
3. `deserializeCudaEngine()` 反序列化共享 engine。
4. 遍历 I/O tensor，记录 input/output 名称。
5. 对每个 slot：
   - 创建 non-blocking stream；
   - 创建完成和阶段计时 events；
   - 创建绑定该 stream 的 NPP context；
   - 从共享 engine 创建独立 `IExecutionContext`。

PImpl 结构（公开类只有 `unique_ptr<Impl>`）把 CUDA/TensorRT 头文件隐藏在 `.cu` 中，减少头文件依赖，也让资源细节不暴露给调用者。

### 12.2 `ensure_capacity()`

它先计算当前 batch 中最大的源图字节数 `source_stride`。每张图在 source/pinned buffer 中占一个同样大的槽位：小图会留下空隙，但地址计算简单且不会互相覆盖。

只有任何一个容量不足时才调用 replacement 函数。它们采用：

```text
先分配新内存 → 成功后释放旧内存 → 更新指针和容量
```

比“先 free 再 malloc”更安全：若新分配失败，旧缓冲区还在。不过此实现的多个 buffer 扩容不是整体事务；中间某次失败时，前面成功扩大的 buffer 会保留，由异常清理负责最终释放。

`cudaMalloc/cudaFree` 可能同步且昂贵，所以把 `capacity_growth_ms` 单独统计，不伪装成稳定阶段推理耗时。固定 shape 稳定运行后，它应为 0。

## 13. 最核心：`submit()` 逐段解释

### 13.1 输入和所有权检查

`tensorrt_backend.cu:221-239` 检查：slot 下标、非空 batch、batch≤4、metadata 数量匹配、状态必须为 Reserved、所有图必须是 `CV_8UC3`。

这些检查很重要：如果 metadata 数量与图片不同，推理即使成功也无法正确关联结果；如果未 reserve 就 submit，可能覆盖正在工作的 slot。

### 13.2 设置动态 shape 并解析输出大小

```cpp
const nvinfer1::Dims4 shape(batch, 3, 640, 640);
slot.context->setInputShape(input_name, shape);
const Dims output_shape = slot.context->getTensorShape(output_name);
```

动态 batch engine 必须在每个 context、每次 shape 改变时设置实际输入 shape。接着从 context 查询已经解析出的输出 shape。任何维度 `<=0` 表示仍动态或无效，不能据此分配 output。

把所有输出维度相乘得到 `output_elements`，再乘 `sizeof(float)` 得到字节数。注意“元素数”和“字节数”是两回事。

### 13.3 host staging

```cpp
unsigned char* destination = staging + batch * source_stride;
for (int row = 0; row < image.rows; ++row) {
    std::memcpy(destination + row * row_bytes,
                image.ptr(row), row_bytes);
}
```

不能总是假定 `cv::Mat` 连续：ROI 或某些解码结果可能有额外 step。逐行用 `image.ptr(row)` 拷贝只复制有效像素。整理后的 staging 每行紧密排列，所以后面的 NPP source step 使用 `image.cols * 3`。

### 13.4 H2D

先在 stream 记录 `h2d_start`，然后每张图从各自 pinned 槽位异步复制到 `device_source`，最后记录 `h2d_end`。这些 event 也只是排队，提交阶段不读取耗时。

### 13.5 letterbox + NPP resize

每张图执行：

1. `cudaMemsetAsync(letterbox, 114, ...)`：整张目标图填灰色 114。
2. `scale = min(640/src_w, 640/src_h)`：保持宽高比，保证两边都不超过 640。
3. 四舍五入计算 resize 后宽高。
4. `pad_x/pad_y` 把缩放图居中。
5. 把 `scale/pad/源尺寸` 写入该帧 metadata，后处理需要逆变换。
6. `destination` 指针偏移到 padding 内部左上角。
7. `nppiResize_8u_C3R_Ctx()` 直接把缩放结果写进 letterbox 的内部区域。

地址偏移：

```cpp
(pad_y * input_width + pad_x) * 3
```

先跳过 `pad_y` 行，再跳过 `pad_x` 个像素，每像素 3 字节。

NPP 的 destination step 仍是完整 letterbox 一行 `640*3` 字节，而不是 resized width；否则下一行会写错位置。

### 13.6 normalize、绑定 TensorRT、推理与 D2H

NPP 结束后在同一 stream launch `normalize`，无需显式等待。然后：

```cpp
setTensorAddress(input_name, slot.input);
setTensorAddress(output_name, slot.output);
enqueueV3(slot.stream);
```

TensorRT 被排到同一 stream，必然在前面的预处理完成后读取 input。随后 D2H 也排在 inference 后，最后记录 `done`。

只有所有 CUDA 命令和 `done` event 都成功入队后，才执行：

```cpp
slot_pool.mark_submitted(index, std::move(metadata));
```

异常时先 `cudaStreamSynchronize(slot.stream)`，确保已经入队的工作不再访问资源，再把 slot 标成 Failed，最后原样抛出异常。

### 13.7 一个容易忽略的 metadata 细节

函数参数 `metadata` 先复制到 `slot.metadata`，预处理得到的 `transform` 写入的是 `slot.metadata.frames[...]`。末尾移动给 `SlotPool` 的是原参数 `metadata`，其中 transform 仍可能是默认值；但真实结果由 `collect()` 从 `slot.metadata` 取，所以后处理拿到的是正确 transform。`SlotPool` 保存的副本在这里主要承担生命周期/状态约束，不是最终 GPU 结果来源。

## 14. `ready()` 与 `collect()`：为什么既查询又等待

### 14.1 `ready()`

```cpp
cudaEventQuery(done)
```

返回：

- `cudaErrorNotReady`：GPU 还没做到 done，正常返回 false；
- `cudaSuccess`：已完成；
- 其他错误：抛异常。

它不阻塞，所以总控可以扫描 pending，优先回收已经完成的任意 slot。

### 14.2 `collect()`

1. `Submitted → Completing`，防止其他调用者重复收集。
2. `cudaEventSynchronize(done)`：只等待这个 slot。
3. 把 pinned output 拷到 `std::vector<float>`，让返回结果不依赖可复用 slot 内存。
4. 从 context 保存 output shape。
5. 用成对 events 读取阶段时间。
6. `release()`：slot 回到 Free。

若收集失败，slot 进入 Failed，不会静默复用。

## 15. 总控循环 `run_integrated_pipeline()` 逐段解释

### 15.1 对象创建顺序也是析构顺序

局部对象按创建的反顺序析构。backend 最先创建、scheduler 后创建，因此正常离开函数时 scheduler 会比 backend 更早析构，采集线程先停止，GPU 资源后释放。这符合所有权依赖。

### 15.2 `PendingBatch` 为什么保留 images

backend 的 submit 会先把图像内容复制进 pinned buffer，所以 GPU 并不继续依赖原 `cv::Mat`。但收集后要绘制标注图，`ResultWriter::write()` 仍需要源图，因此 pending 保存 `images` 直到后处理完成。

### 15.3 主 while 的退出条件

```cpp
while (!scheduler.done() || !pending.empty())
```

即使所有 source 和队列都结束，只要还有已提交 batch，就必须继续 collect。这就是正常 EOS “drain 已提交工作”。

### 15.4 提交循环

```cpp
while (backend.available_slots() != 0 && !scheduler.done())
```

有空闲 slot 且 scheduler 仍有工作时尽量提交。`next_batch()` 最多等 4ms 凑批，返回后计算：

- `batch_fill_ms`：调用 `next_batch()` 花费的 host 时间；
- `queue_wait_ms`：batch 内每帧从 `captured_at` 到即将提交的平均等待时间。

再申请 slot、整理 images/metadata、调用 `submit()`，并把 slot 加到 pending。

`available_slots()` 后再 `try_reserve()` 看似重复，但它把假设明确验证出来：当前完整应用只有一个提交线程，正常应成功；若将来并发提交，二者之间有竞态，代码会抛错提醒设计已不成立。

### 15.5 回收策略

先遍历 pending，找到第一个 `ready()` 的 slot立即收集。这样不会因为队首慢 batch 阻塞已经完成的后续 batch。

如果一个都没 ready 且 pending 非空，就阻塞收集队首。否则主循环会高速空转，不断查询 event，占满一个 CPU 核。

结果因此可能乱序写出。这不是 bug：每条 JSON 都带 `(stream_id, frame_id, batch_id)`，消费者应按身份解释，而不是依赖文件行顺序。

### 15.6 异常清理

```cpp
const std::exception_ptr causal = std::current_exception();
scheduler.stop(true);
for (const PendingBatch& batch : pending) {
    try { backend.collect(batch.slot); } catch (...) {}
}
std::rethrow_exception(causal);
```

顺序含义：

1. 保存原始异常。
2. 拒绝新采集并丢弃未提交队列内容。
3. 已提交到 CUDA 的命令不能假装取消；逐个 collect/quiesce，确保资源不被仍在运行的 GPU 使用。
4. 清理期间的新异常不覆盖原始根因。

### 15.7 终态对账

当前真实流水线最后检查：

```text
captured == completed + evicted + aborted
```

- completed：完成后处理的帧；
- evicted：队列溢出丢弃 + latest-first 丢弃旧帧；
- aborted：关闭时清空 + 入队时发现关闭。

`Accounting::validate_terminal()` 是更细粒度的 CPU 可测试模型，但当前 `run_integrated_pipeline()` 没有直接使用它。不要误以为真实路径已经逐项维护了 `Accounting` 中的每个字段。

## 16. 后处理：如何把框还原到原图

`ResultWriter::write()` 先检查 batch metadata、源图数和 output 元素数是否一致，然后按 `elements_per_image` 切片。

预处理做的是：

```text
原图坐标 --乘 scale--> 缩放图坐标 --加 pad--> 640×640 模型坐标
```

后处理需要逆过程：

```text
原图坐标 = (模型坐标 - pad) / scale
```

代码把 backend 记录的 `Transform` 转成第 11 课的 `LetterboxInfo`，再调用 `decode_yolov8_output()` 完成 YOLOv8 解码、阈值过滤、NMS 和坐标还原。

只保存第一张 `annotated_0.jpg`，避免长运行持续写大量图片。检测 JSONL 可通过 `maximum_detection_records` 限制条数；达到上限后仍计算 frame latency，只是不再写检测记录。

## 17. 指标：哪些时间能相加，哪些不能

### 17.1 两个时钟域

- host：`std::chrono::steady_clock`，用于队列、组 batch、host staging、CPU 后处理和端到端延迟。
- GPU：同一 slot stream 上的 CUDA events，用于 H2D、预处理、TensorRT 和 D2H。

不要用 CPU 在 `enqueueV3()` 调用前后的时间当 GPU inference 时间，因为 enqueue 通常只负责排队。

### 17.2 延迟与吞吐不是一回事

- FPS = 完成帧数 / 整体墙钟时间。
- frame latency = `ResultWriter` 处理该帧时刻 − `captured_at`。
- batch stage time 是每个 batch 的阶段耗时，不能简单除以 batch size 后就称作单帧端到端延迟。

### 17.3 有界内存统计

`latencies_` 最多保留 8192 个近期值，`batches_` 最多保留 256 个代表性 batch；完整 batch timing 每批立刻写入 `batch_timing_samples.jsonl`。因此长时间运行不会因“把所有统计都留在 vector”而持续涨内存。

这里注释称其为 deterministic rolling reservoir，更准确地说是**固定大小的环形覆盖/近期窗口**，不是经典的随机 reservoir sampling。因此最终分位数主要代表保留下来的近期样本，不是对全历史严格无偏抽样。

### 17.4 当前指标实现的一个边界

`PipelineMetrics::submitted_` 在 `record_batch()` 中和 `completed_` 一起增加，而 `record_batch()` 只在 collect 和后处理成功后调用。因此运行中的 `metrics_snapshots.jsonl` 里，`submitted` 并不是真正的“已提交但可尚未完成”总数，通常与 completed 同步；最终成功运行时两者相等没有问题，但它不能用来观察实时 in-flight 数量。真实 in-flight 可由 `slot_count - available_slots` 或 pending 大小推断。

## 18. 必须直说的源码细节和限制

### 18.1 关于视频 decode 是否计入 capture-to-result latency

`frame_source.hpp` 的注释说“时间戳在 read 成功后，因此 capture-to-result latency includes video decode time”，这在逻辑上不成立。`frame_scheduler.cpp` 实际顺序是：

```cpp
sources_[stream]->read(image); // decode 已经发生
Clock::now();                  // 之后才打时间戳
```

所以当前 `captured_at → result` **不包含这一次 `read()` 的 decode 耗时**。本课 `README.md` 后面的描述“timestamp is assigned after successful decode，latency starts at admission”反而与代码一致。阅读时应以执行顺序为准。

### 18.2 `frame_count` 多源分配可能向下取整

普通模式使用：

```cpp
frames_per_source = max(1, frame_count / source_count);
```

若 `frame_count=5`、source 数为 2，每路 2 帧，总共只处理 4 帧；若 source 数大于 frame_count，每路至少 1 帧，总数反而大于 frame_count。因此这个参数在多源模式下更接近“用于均分的目标总数”，并不保证最终严格等于输入值。

### 18.3 I/O tensor 假设比报错文字更宽松

构造函数遍历所有 I/O，把最后看到的输入名和输出名保存下来，然后只检查二者非空。报错写的是“expected exactly one input and one output”，但代码没有显式验证数量恰好各一个。对本课 YOLO engine 假设成立；若换多输入/多输出 engine，需要改成计数和明确选择 tensor。

### 18.4 `metrics.json` 是手写 JSON

GPU 名称、策略名等字符串直接写入流，没有通用 JSON escaping。常见 NVIDIA GPU 名称不会触发问题，但这不是可处理任意字符串的 JSON 库。

这些不一定都要在本课立即修复，但理解边界比把教学代码当成无条件完备的生产框架更重要。

## 19. 错误注入为什么值得看

环境变量不是业务功能，而是用来强制走难触发的清理路径：

- `LESSON21_FAIL_SOURCE_FRAME`：采集线程异常能否传回主线程。
- `LESSON21_FAIL_SUBMIT_BATCH`：提交前/中失败能否停止流水线。
- `LESSON21_FAIL_INSUFFICIENT_CAPACITY`：容量准备失败。
- `LESSON21_FAIL_TENSOR_ADDRESS`：绑定 tensor 失败。
- `LESSON21_FAIL_ENQUEUE`：TensorRT enqueue 失败。
- `LESSON21_FAIL_POSTPROCESS_BATCH`：GPU 已完成但 CPU 后处理失败。
- `LESSON21_ABORT_AFTER_SUBMISSIONS`：已有 GPU in-flight 时中止。

生产代码不能只验证成功路径。尤其异步 GPU 程序中，“CPU 抛异常了”不代表 GPU 不再运行，清理顺序直接决定是否 use-after-free 或进程卡死。

## 20. 测试代码在证明什么

### 20.1 CPU 核心测试

`test_pipeline_core.cpp` 检查：

- 两个 slot 不重复分配；
- slot 满时 `try_reserve()` 为空；
- 逆序回收仍保留正确身份；
- Failed slot 不能 release；
- drop-oldest 的弹出结果和计数；
- close(false) 会 drain，close(true) 会 discard；
- 对账恒等式能拒绝不一致计数。

### 20.2 scheduler 测试

`test_frame_scheduler.cpp` 检查多源唯一身份、batch_index、队列峰值和 source 异常传播。该测试为了紧凑写成一行较多，不适合作为 C++ 排版范例，但测试意图有效。

### 20.3 GPU 测试

- batch 1/2/4 与双 slot 冒烟；
- 超过动态 profile 的 batch 5 必须失败；
- 输出身份、环境、阶段计时和 batch 分布；
- batch 1 与 batch 4 检测结果近似一致；
- overload/latest-first 必须明确计入 dropped；
- 长一些的运行只保留有界统计，且稳定阶段不再扩容；
- 多种错误注入必须非零退出且不能挂死。

GPU 测试需要第 17 课生成的动态 engine、NVIDIA GPU 和指定容器，CPU 测试通过不能代替这些证据。

## 21. 建议你实际跟代码练习的顺序

### 练习 1：只画状态，不跑 GPU

在纸上跟踪 `main.cpp`：

```text
slot 0: Free → Reserved → Submitted ─────────────→ Completing → Free
slot 1: Free → Reserved → Submitted → Completing → Free
结果顺序: stream 1，然后 stream 0
```

回答：为什么结果没有串流？因为 stream/frame identity 存在 batch metadata 中。

### 练习 2：手推 drop-oldest

容量 2，依次 push 1、2、3：

```text
[1] → [1,2] → 弹 1 → [2,3]
evicted=1, peak=2
```

再对照 `test_pipeline_core.cpp:34-38`。

### 练习 3：手推 round-robin batch

假设 Q0 有 `a0,a1`，Q1 有 `b0,b1`，cursor=0，maximum=3：结果应接近 `a0,b0,a1`，下次 cursor 从 Q1 开始。然后把策略改成 latest-first，观察每路旧帧为什么会计入 stale。

### 练习 4：只运行 backend smoke

先不要上完整流水线。按本课 README 在容器中运行 `integrated_tensorrt_gpu_smoke` 的 batch 1，再运行 batch 4 和 `--two-slots`。重点看输出元素数、各 GPU 阶段时间以及逆序 collect。

### 练习 5：画一个 slot 的时间线

```text
slot.stream:
| H2D | memset+NPP+kernel | TensorRT | D2H | done |

CPU:
submit 返回 ---------------------- ready? ------- collect
```

再增加第二条 slot stream，思考两行哪些区间可能重叠。

### 练习 6：观察过载策略

分别运行 block 和 drop-oldest。不要只比较 FPS，还要对比：

- `captured/processed/dropped`；
- `queue_peak`；
- p50/p90/p99；
- batch distribution。

block 倾向于不丢帧但把背压传给采集；drop-oldest 倾向于保持新鲜度但牺牲完整性，没有一个策略对所有业务都“更好”。

## 22. 最终应掌握的检查清单

读完并动手后，应能独立回答：

1. 为什么队列必须有容量上限？
2. `close(false)` 和 `close(true)` 的行为有什么不同？
3. 为什么多源调度使用 `try_pop()` 而不是对某一路 `pop()`？
4. 为什么 `stream_id + frame_id` 才是帧身份，slot 不是？
5. 为什么共享 engine，但每个 slot 独占 context、stream 和 buffer？
6. pinned host memory 对异步 H2D/D2H 有什么作用？
7. 为什么同一 stream 的 NPP、kernel 和 TensorRT 之间不需要逐段 CPU synchronize？
8. `ready()` 和 `collect()` 分别是非阻塞查询还是阻塞等待？
9. 为什么不能用 `enqueueV3()` 的 CPU 调用耗时表示 GPU inference 耗时？
10. letterbox 的 `scale/pad_x/pad_y` 为什么必须随帧 metadata 保存？
11. 异常发生时，为什么要先停止采集，再等待已经提交的 CUDA 工作？
12. 正常结束时 `captured == completed + evicted + aborted` 为什么应该成立？

如果这些问题能不看答案讲清楚，你已经抓住第 21 课的主体。CUDA kernel 的优化细节反而可以后续再深入；这一课更重要的是先建立**异步执行、资源所有权、背压和可验证对账**这四个工程模型。
