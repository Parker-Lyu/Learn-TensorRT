# 19 - ONNX Graph Surgery And TensorRT Plugin

Goal: learn the escalation path for unsupported operators and TensorRT parser failures.

Topics:

- Unsupported operator diagnosis
- PyTorch equivalent replacement
- ONNX GraphSurgeon
- Constant folding
- Node replacement and splitting
- TensorRT plugin strategy
- `IPluginV2DynamicExt`
- Plugin serialization and deserialization

Acceptance criteria:

- You can explain model rewrite, ONNX graph surgery, and TensorRT plugin as three escalation levels.
- You can edit a small ONNX graph with GraphSurgeon.
- You can describe `getOutputDimensions`, `configurePlugin`, `enqueue`, `serialize`, and `clone`.
