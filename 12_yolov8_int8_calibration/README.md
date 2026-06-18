# 12 - YOLOv8 INT8 Calibration

Goal: build INT8 TensorRT engines and understand quantization trade-offs with both speed and
accuracy evidence.

This lesson extends the single-input precision alignment from `06a_polygraphy_precision_alignment`
into a small representative-image regression workflow. Tensor-level drift is still useful for
debugging, but release decisions should also compare decoded detections across many images.

Topics:

- Calibration image set
- PTQ
- Entropy calibration and KL divergence intuition
- `IInt8EntropyCalibrator2`
- Calibration table
- INT8 engine build
- FP32, FP16, and INT8 output comparison
- FP16 versus INT8 latency
- Multi-image numerical drift summary
- Decoded box, class, and confidence comparison
- Mixed precision fallback
- Sensitive layer fallback to FP16 or FP32
- QAT as the fallback when PTQ fails
- Accuracy and detection quality comparison

Acceptance criteria:

- An INT8 engine is generated.
- A representative calibration set is documented.
- A small validation image set is documented separately from the calibration set.
- A short report compares FP32, FP16, and INT8 speed, tensor drift statistics, and detection quality.
- The report lists any high-drift or changed-detection examples that deserve visual inspection.
- You can explain what to do when INT8 causes a severe recall drop.
