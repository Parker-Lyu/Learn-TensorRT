# 05 - Torch To ONNX

This lesson exports YOLOv8n from PyTorch to a simplified ONNX graph and validates that ONNX Runtime
produces the same raw model output as PyTorch for the same preprocessed tensor.

Goal: build a trustworthy, TensorRT-ready ONNX artifact before using `trtexec` and TensorRT C++
runtime code.

Topics:

- Ultralytics YOLO export
- ONNX opset
- Static shape and dynamic shape
- ONNX Runtime validation
- `onnxsim` or Ultralytics graph simplification
- Netron graph inspection
- Input and output tensor names
- Raw output comparison before NMS

## Why This Matters

TensorRT starts from an ONNX graph. If the ONNX model is already wrong, the TensorRT engine will
only make the wrong result faster.

This lesson validates the boundary between training-framework code and deployment-framework code:

```text
YOLOv8n .pt weights
  -> export to simplified ONNX
  -> inspect input/output names and shapes
  -> run ONNX Runtime on the same NCHW float32 tensor as PyTorch
  -> compare raw [1, 84, 8400] outputs before decode and NMS
```

The comparison happens before postprocessing so later lessons can separate model-export problems
from YOLO decode, NMS, coordinate mapping, TensorRT precision, or C++ buffer bugs.

## Directory Layout

- `export_yolov8_onnx.py`: downloads or uses `assets/yolov8n.pt`, then exports simplified ONNX
  artifacts by default.
- `inspect_onnx.py`: checks the ONNX model and records tensor names, shapes, opset, and operator
  inventory.
- `validate_onnx_runtime.py`: runs PyTorch and ONNX Runtime on the same preprocessed image tensor.
- `outputs/`: generated simplified ONNX models, optional raw ONNX comparisons, tensor dumps, and
  validation reports. This folder is ignored by git.
- `../assets/yolov8n.pt`: shared YOLOv8n weights used by later lessons.
- `../assets/img.jpeg`: shared sample image.

## Export

Export a simplified static-shape ONNX model:

```bash
python3 export_yolov8_onnx.py
```

The default command writes:

```text
outputs/yolov8n.onnx
```

Export a simplified dynamic-shape ONNX model for later TensorRT optimization-profile experiments:

```bash
python3 export_yolov8_onnx.py --dynamic
```

The dynamic command writes:

```text
outputs/yolov8n_dynamic.onnx
```

Graph simplification is enabled by default because lesson 06 uses these files as the canonical
TensorRT build inputs. Export an unsimplified graph only when you want to compare tool behavior or
debug a simplifier issue:

```bash
python3 export_yolov8_onnx.py --no-simplify --output outputs/yolov8n_raw.onnx
```

The default simplified artifacts are the handoff contract to lesson 06:

```text
outputs/yolov8n.onnx
outputs/yolov8n_dynamic.onnx
```

The default weights path is:

```text
../assets/yolov8n.pt
```

If it is missing, the script downloads the official YOLOv8n weights into the root `assets` folder so
later lessons can reuse the same file.

## Inspect

Check the simplified static model and write a JSON report:

```bash
python3 inspect_onnx.py
```

The report is written to:

```text
outputs/onnx_inspection.json
```

For YOLOv8n static export, the important tensors should look like:

```text
input:  images  FLOAT [1, 3, 640, 640]
output: output0 FLOAT [1, 84, 8400]
```

The `84` dimension is `4` box values plus `80` COCO class scores. The `8400` dimension is the number
of candidate boxes across the detection feature maps.

Open the ONNX file in Netron and confirm:

- The model has one image input.
- The output tensor name and shape match the inspection report.
- The graph does not contain unexpected preprocessing or NMS nodes.

Inspect the dynamic model separately when preparing for TensorRT optimization profiles:

```bash
python3 inspect_onnx.py \
  --onnx outputs/yolov8n_dynamic.onnx \
  --report outputs/onnx_dynamic_inspection.json
```

## Validate

Compare PyTorch raw output with the simplified static ONNX Runtime raw output:

```bash
python3 validate_onnx_runtime.py
```

The script uses the same preprocessing convention as the C++ preprocessing lesson:

```text
BGR image
  -> letterbox to 640 x 640 with padding value 114
  -> BGR to RGB
  -> normalize to float32 [0, 1]
  -> HWC to CHW
  -> add batch dimension, shape [1, 3, 640, 640]
```

The validation command writes:

- `outputs/input_nchw_float32.npy`
- `outputs/pytorch_raw_output.npy`
- `outputs/onnxruntime_raw_output.npy`
- `outputs/validation_preview.txt`
- `outputs/validation_report.json`

Use a different image:

```bash
python3 validate_onnx_runtime.py --image /path/to/image.jpg
```

Use a different tolerance:

```bash
python3 validate_onnx_runtime.py --rtol 1e-3 --atol 1e-3
```

The command exits with status `2` if the outputs are not close enough. That is intentional: a failed
validation should stop the deployment chain until the mismatch is understood.

## Expected Report Fields

`validation_report.json` records:

- weights path
- ONNX path
- image path
- letterbox scale and padding
- ONNX Runtime input/output names
- output shape and dtype
- max absolute error
- mean absolute error
- P99 absolute error
- index and values at the largest mismatch
- `np.allclose` result

These fields become the first accuracy-alignment evidence before Polygraphy and TensorRT enter the
project.

## Checkpoints

- Export both simplified static and dynamic ONNX models and compare their input shapes in the
  inspection reports.
- Change `--imgsz` to `320` and explain why the output candidate count changes.
- Open the model in Netron and find the first `Conv` node and the final output tensor.
- Delete `outputs/yolov8n.onnx`, rerun the export, and confirm the command is reproducible.
- Export `outputs/yolov8n_raw.onnx` with `--no-simplify` and compare its node count with the
  simplified graph.
- Intentionally use a loose and a strict tolerance in validation and explain the difference between
  acceptable float drift and a real export bug.

Acceptance criteria:

- Simplified `outputs/yolov8n.onnx` and `outputs/yolov8n_dynamic.onnx` are generated.
- Input and output tensor names are recorded.
- ONNX checker passes.
- Static and dynamic inspection reports show the expected static tensor shapes and dynamic
  `batch`/`height`/`width` axes.
- ONNX Runtime output from the simplified static ONNX is compared with PyTorch output on the same
  image tensor.
- The validation report explains the numerical difference and tolerance.
