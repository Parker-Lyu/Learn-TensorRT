# 06a - Polygraphy Precision Alignment

Goal: use Polygraphy to debug numerical differences between ONNX Runtime and TensorRT.

Why it matters:

- A TensorRT engine that builds successfully can still be wrong enough to hurt detection quality.
- Precision alignment gives you evidence when debugging preprocessing, export settings, FP16, INT8, or unsupported graph behavior.

Topics:

- Polygraphy model inspection
- ONNX Runtime versus TensorRT comparison
- Saving input and output tensors
- FP32, FP16, and INT8 drift analysis
- Absolute and relative tolerance selection
- First-mismatch debugging workflow
- Reproducible command logs for benchmark reports

Acceptance criteria:

- Polygraphy can run the YOLO ONNX model with ONNX Runtime.
- Polygraphy can run the same model or engine with TensorRT.
- ONNX Runtime and TensorRT outputs are compared using the same input tensor.
- Any mismatch is summarized with max error, mean error, tolerance, and likely cause.
- The final note explains whether the observed drift is acceptable for the deployment target.
