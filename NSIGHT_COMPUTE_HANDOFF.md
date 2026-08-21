# Nsight Compute 专项课程 Hand-off

## 结论

不将 Nsight Compute 纳入核心路径；建议作为课程 20 的高级扩展（例如 `20A`）。课程目标不是保证 kernel 加速，而是用证据判断优化是否值得。

## 计划

1. 先用 Nsight Systems 区分 NPP resize、fused kernel、内存拷贝和 staging 成本。
2. 为 `bgr_to_rgb_nchw` 增加独立 CUDA event、NVTX 标记和 standalone benchmark，建立固定输入、warmup、迭代次数的基线。
3. 用 Nsight Compute 分析 occupancy、内存吞吐、global load/store efficiency、寄存器使用和 warp stall reasons。
4. 对比少量可解释的变体：block 配置、访问方式、向量化读取，以及融合前后的实现。
5. 分别报告 kernel、GPU preprocessing 和端到端 pipeline 的时间、正确性误差与环境信息。

## 验收

- 能从 Nsight Systems 选择一个有依据的 kernel 热点。
- 有可复现的 `.ncu-rep`、指标摘要和基线/变体对比。
- 数值误差满足课程 20 的约束。
- 明确说明 kernel 指标改善但端到端无改善时，优化不能算部署收益。
- 允许得出“已接近内存带宽上限，继续优化收益很小”的结论。

## 边界

- 不要求预设的性能提升百分比。
- 不把所有 TensorRT 内部 kernel 作为分析对象。
- 课程 26 的 plugin kernel 可作为后续更复杂案例；课程 13 是前置条件。
