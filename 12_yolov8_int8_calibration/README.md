# 12 - YOLOv8 INT8 Calibration

Goal: build INT8 TensorRT engines and understand quantization trade-offs.

Topics:

- Calibration image set
- PTQ
- Entropy calibration and KL divergence intuition
- `IInt8EntropyCalibrator2`
- Calibration table
- INT8 engine build
- FP32, FP16, and INT8 output comparison
- FP16 versus INT8 latency
- Mixed precision fallback
- Sensitive layer fallback to FP16 or FP32
- QAT as the fallback when PTQ fails
- Accuracy and detection quality comparison

Acceptance criteria:

- An INT8 engine is generated.
- A representative calibration set is documented.
- A short report compares FP32, FP16, and INT8 speed and detection quality.
- You can explain what to do when INT8 causes a severe recall drop.
