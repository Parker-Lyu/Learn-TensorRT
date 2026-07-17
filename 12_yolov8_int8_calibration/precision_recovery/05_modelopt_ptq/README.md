# Step 05 ModelOpt Candidate Notes

This directory contains the source and focused tests for ModelOpt explicit Q/DQ recovery. Generated
ONNX models, TensorRT engines, logs, timing captures, and evaluation reports remain under the
ignored `../../outputs/precision_recovery/05_modelopt_ptq/` directory.

## FP16 High-Precision Q/DQ Candidate

Status: **BUILD FAIL** on 2026-07-17. The validation gate was not run.

This predeclared candidate kept the coverage-aware 3,000-image calibration manifest, ModelOpt max
calibration, INT8 activation quantization, per-channel INT8 weight quantization, preprocessing, and
FP32 external I/O unchanged. The only intended precision change was:

- set `trt_high_precision_dtype` to `Half` for input and weight quantizers;
- convert the calibrated quantized model to FP16 before export;
- insert an FP32-to-FP16 cast after `images` and an FP16-to-FP32 cast before `output0`.

Candidate configuration ID:

```text
modelopt-int8-default-max-high-precision-fp16-v1
```

The 32-image smoke export and the formal 3,000-image export both succeeded. The formal ONNX graph
passed ONNX checker and retained the required boundary contract:

```text
images  FLOAT [1, 3, 640, 640]
output0 FLOAT [1, 84, 8400]
```

Graph evidence:

- 131 `QuantizeLinear` nodes;
- 131 `DequantizeLinear` nodes;
- 2 boundary `Cast` nodes;
- 135 FLOAT16 tensor constants;
- ONNX SHA-256:
  `3fad6a3dba71e4026c7e8036a413fbc027dd310cf15c91e72ccdda70e677dc90`.

Formal export command:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/modelopt_ptq.py \
    --candidate-kind formal \
    --calibration-images 3000 \
    --batch-size 4 \
    --high-precision fp16 \
    --name yolov8n_modelopt_int8_max_hp_fp16_train3000
'
```

TensorRT 8.6.1 rejected the graph before tactic selection at the first per-channel weight
`QuantizeLinear` node:

```text
Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"
```

The failed node was:

```text
/model/model.0/conv/weight_quantizer/QuantizeLinear
```

An independent ONNX audit checked the scale tensors used by all 262 Q/DQ nodes. It found zero
non-positive scales and zero positive FP16 subnormal scales. For the rejected first weight
quantizer, the 16 scales ranged from approximately `0.0007939` to `0.0040207`. The evidence therefore
indicates a TensorRT 8.6 parser compatibility limitation for FLOAT16 Q/DQ scales rather than an
invalid calibrated range.

The same ONNX model was then checked in the existing TensorRT 10.14.1 ModelOpt container with a
non-persistent optimization-level-0 build:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  trtexec \
    --onnx=12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/\
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx \
    --stronglyTyped \
    --builderOptimizationLevel=0 \
    --skipInference
'
```

TensorRT 10.14 parsed the graph and generated an in-memory engine successfully in approximately
`16.98` seconds. This confirms that the graph is valid for a newer strongly-typed TensorRT parser.
It does not convert this TRT8.6 candidate into a passing result: no optimized TRT10 engine was saved,
no matched TRT10 FP32/FP16 references were established, and no validation gate was run.

Build command:

```bash
docker exec trt_dev bash -lc '
  cd /workspace/Projects/Learn-TensorRT &&
  output=12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq &&
  python3 \
    12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/build_trt86_qdq_engine.py \
    --onnx "$output/yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx" \
    --engine "$output/yolov8n_modelopt_int8_max_hp_fp16_train3000_trt86.engine" \
    --timing-cache "$output/modelopt_qdq_hp_fp16_trt86.timing.cache"
'
```

Failure evidence:

- `../../outputs/precision_recovery/05_modelopt_ptq/`
  `yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx.json`
- `../../outputs/precision_recovery/05_modelopt_ptq/`
  `yolov8n_modelopt_int8_max_hp_fp16_train3000_trt86.build.log`

No TensorRT engine was produced, no validation labels were consulted, and the 5,000-image gate was
not run. The existing FP32-high-precision Q/DQ candidate remains the passing quality result. A future
experiment may promote this graph to a formal TensorRT 10 candidate or use narrowly scoped TensorRT
precision constraints on the TRT8.6 FP32-scale graph, but both are separate candidates with new
engine identities and matched-runtime evidence.
