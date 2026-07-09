#!/usr/bin/env python3
"""YOLOv8 TensorRT Python reference pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class LetterboxInfo:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    scale: float


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: list[float]


@dataclass
class TensorBinding:
    name: str
    mode: str
    dtype: np.dtype
    shape: tuple[int, ...]
    host: np.ndarray
    device: int


class CudaStream:
    def __init__(self) -> None:
        self.stream = check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate")

    @property
    def handle(self) -> int:
        return int(self.stream)

    def synchronize(self) -> None:
        check_cuda(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def close(self) -> None:
        if self.stream is not None:
            check_cuda(cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")
            self.stream = None

    def __enter__(self) -> "CudaStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DeviceAllocation:
    def __init__(self, byte_count: int) -> None:
        if byte_count <= 0:
            raise ValueError("device allocation byte count must be positive")
        self.ptr = check_cuda(cudart.cudaMalloc(byte_count), "cudaMalloc")

    def close(self) -> None:
        if self.ptr is not None:
            check_cuda(cudart.cudaFree(self.ptr), "cudaFree")
            self.ptr = None


def check_cuda(result: tuple, operation: str):
    status = result[0]
    if status != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(status)
        _, message = cudart.cudaGetErrorString(status)
        raise RuntimeError(f"{operation} failed: {name.decode()} ({message.decode()})")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def parse_shape(text: str) -> tuple[str, tuple[int, ...]]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("shape must look like NAME:D0xD1x...")
    name, shape_text = text.split(":", 1)
    if not name or not shape_text:
        raise argparse.ArgumentTypeError("shape must look like NAME:D0xD1x...")
    try:
        dims = tuple(int(part) for part in shape_text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {text}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f"all shape dimensions must be positive: {text}")
    return name, dims


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image: {path}")
    return image


def letterbox(image_bgr: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    input_width, input_height = input_size
    if input_width <= 0 or input_height <= 0:
        raise ValueError("input size must be positive")

    original_height, original_width = image_bgr.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    resized_width = max(1, min(input_width, int(round(original_width * scale))))
    resized_height = max(1, min(input_height, int(round(original_height * scale))))

    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    pad_right = pad_width - pad_left
    pad_bottom = pad_height - pad_top

    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    output[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized

    info = LetterboxInfo(
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        scale=scale,
    )
    return output, info


def preprocess(image_bgr: np.ndarray, input_shape: tuple[int, ...]) -> tuple[np.ndarray, LetterboxInfo]:
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"expected a single NCHW RGB input, got shape {input_shape}")
    input_height, input_width = input_shape[2], input_shape[3]
    letterboxed, info = letterbox(image_bgr, (input_width, input_height))
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), info


def load_engine(path: Path) -> trt.ICudaEngine:
    if not path.is_file():
        raise FileNotFoundError(f"engine not found: {path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
    return engine


def tensor_names(engine: trt.ICudaEngine) -> Iterable[str]:
    if hasattr(engine, "num_io_tensors"):
        for index in range(engine.num_io_tensors):
            yield engine.get_tensor_name(index)
    else:
        for index in range(engine.num_bindings):
            yield engine.get_binding_name(index)


def tensor_mode(engine: trt.ICudaEngine, name: str) -> str:
    if hasattr(engine, "get_tensor_mode"):
        return "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"
    return "input" if engine.binding_is_input(engine.get_binding_index(name)) else "output"


def tensor_dtype(engine: trt.ICudaEngine, name: str) -> np.dtype:
    if hasattr(engine, "get_tensor_dtype"):
        return trt.nptype(engine.get_tensor_dtype(name))
    return trt.nptype(engine.get_binding_dtype(engine.get_binding_index(name)))


def tensor_shape(engine: trt.ICudaEngine, context: trt.IExecutionContext, name: str) -> tuple[int, ...]:
    if hasattr(context, "get_tensor_shape"):
        shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
    else:
        shape = tuple(int(dim) for dim in context.get_binding_shape(engine.get_binding_index(name)))
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"tensor {name} has unresolved shape {shape}; pass --input-shape")
    return shape


def set_input_shape(context: trt.IExecutionContext, name: str, shape: tuple[int, ...]) -> None:
    if hasattr(context, "set_input_shape"):
        ok = context.set_input_shape(name, shape)
    else:
        ok = context.set_binding_shape(context.engine.get_binding_index(name), shape)
    if not ok:
        raise RuntimeError(f"failed to set input shape for {name}: {shape}")


def allocate_bindings(engine: trt.ICudaEngine,
                      context: trt.IExecutionContext) -> tuple[list[TensorBinding], list[int], list[DeviceAllocation]]:
    bindings: list[TensorBinding] = []
    pointer_table: list[int] = []
    allocations: list[DeviceAllocation] = []
    for name in tensor_names(engine):
        shape = tensor_shape(engine, context, name)
        dtype = tensor_dtype(engine, name)
        host = np.empty(int(np.prod(shape)), dtype)
        allocation = DeviceAllocation(host.nbytes)
        binding = TensorBinding(
            name=name,
            mode=tensor_mode(engine, name),
            dtype=np.dtype(dtype),
            shape=shape,
            host=host,
            device=allocation.ptr,
        )
        bindings.append(binding)
        allocations.append(allocation)
        pointer_table.append(int(allocation.ptr))
        if hasattr(context, "set_tensor_address"):
            if not context.set_tensor_address(name, int(allocation.ptr)):
                raise RuntimeError(f"failed to bind tensor address for {name}")
    return bindings, pointer_table, allocations


def execute(context: trt.IExecutionContext,
            bindings: list[TensorBinding],
            pointer_table: list[int],
            input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    with CudaStream() as stream:
        for binding in bindings:
            if binding.mode != "input":
                continue
            if input_tensor.shape != binding.shape:
                raise ValueError(f"input tensor shape {input_tensor.shape} != engine shape {binding.shape}")
            np.copyto(binding.host.reshape(binding.shape), input_tensor.astype(binding.dtype, copy=False))
            check_cuda(
                cudart.cudaMemcpyAsync(
                    binding.device,
                    binding.host.ctypes.data,
                    binding.host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream.stream,
                ),
                "cudaMemcpyAsync(host-to-device)",
            )

        if hasattr(context, "execute_async_v3"):
            ok = context.execute_async_v3(stream_handle=stream.handle)
        else:
            ok = context.execute_async_v2(bindings=pointer_table, stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async failed")

        outputs: dict[str, np.ndarray] = {}
        for binding in bindings:
            if binding.mode != "output":
                continue
            check_cuda(
                cudart.cudaMemcpyAsync(
                    binding.host.ctypes.data,
                    binding.device,
                    binding.host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream.stream,
                ),
                "cudaMemcpyAsync(device-to-host)",
            )
            outputs[binding.name] = binding.host.reshape(binding.shape).copy()
        stream.synchronize()
        return outputs


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(boxes_xywh, dtype=np.float32)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5
    return xyxy


def map_boxes_to_original(boxes_xyxy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_left) / info.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_top) / info.scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, info.original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, info.original_height)
    return boxes


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.maximum(union, 1.0e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = compute_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def decode_yolov8(output: np.ndarray,
                  info: LetterboxInfo,
                  confidence_threshold: float,
                  iou_threshold: float,
                  max_detections: int) -> list[Detection]:
    prediction = np.squeeze(output)
    if prediction.ndim != 2:
        raise ValueError(f"expected YOLO output rank 2 after squeeze, got {prediction.shape}")
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise ValueError(f"expected boxes + class scores, got {prediction.shape}")

    boxes_xywh = prediction[:, :4]
    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= confidence_threshold
    if not np.any(mask):
        return []

    boxes_xyxy = map_boxes_to_original(xywh_to_xyxy(boxes_xywh[mask]), info)
    scores = scores[mask]
    class_ids = class_ids[mask]

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]
        class_keep = nms(boxes_xyxy[class_indices], scores[class_indices], iou_threshold)
        keep.extend(int(class_indices[index]) for index in class_keep)

    keep = sorted(keep, key=lambda index: float(scores[index]), reverse=True)[:max_detections]
    detections: list[Detection] = []
    for index in keep:
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
                confidence=float(scores[index]),
                box_xyxy=[float(value) for value in boxes_xyxy[index]],
            )
        )
    return detections


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in det.box_xyxy)
        color = (37 * (det.class_id + 3) % 255, 17 * (det.class_id + 7) % 255, 29 * (det.class_id + 11) % 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(0, y1 - label_size[1] - baseline - 4)
        cv2.rectangle(output, (x1, y_label), (x1 + label_size[0] + 4, y_label + label_size[1] + baseline + 4), color, -1)
        cv2.putText(output, label, (x1 + 2, y_label + label_size[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run_ultralytics_reference(weights: Path,
                              image: Path,
                              confidence_threshold: float,
                              iou_threshold: float,
                              max_detections: int) -> dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.predict(
        source=str(image),
        imgsz=640,
        conf=confidence_threshold,
        iou=iou_threshold,
        max_det=max_detections,
        verbose=False,
    )[0]
    boxes = result.boxes
    count = 0 if boxes is None else int(len(boxes))
    top = None
    if count:
        top_index = int(np.argmax(boxes.conf.cpu().numpy()))
        class_id = int(boxes.cls[top_index].item())
        top = {
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
            "confidence": float(boxes.conf[top_index].item()),
            "box_xyxy": [float(value) for value in boxes.xyxy[top_index].cpu().numpy()],
        }
    return {"count": count, "top_detection": top}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp32.engine"))
    parser.add_argument("--image", type=Path, default=Path("../assets/img2.jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--input-shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--reference", action="store_true", help="Run Ultralytics reference for a top detection check.")
    parser.add_argument("--weights", type=Path, default=Path("../assets/yolov8n.pt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.confidence <= 1.0):
        raise ValueError("--confidence must be in [0, 1]")
    if not (0.0 <= args.iou <= 1.0):
        raise ValueError("--iou must be in [0, 1]")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(args.image)
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    input_shapes = dict(args.input_shape)
    input_names = [name for name in tensor_names(engine) if tensor_mode(engine, name) == "input"]
    if len(input_names) != 1:
        raise RuntimeError(f"expected one input tensor, got {input_names}")
    input_name = input_names[0]
    if input_name in input_shapes:
        set_input_shape(context, input_name, input_shapes[input_name])

    input_shape = tensor_shape(engine, context, input_name)
    t0 = time.perf_counter()
    input_tensor, letterbox_info = preprocess(image, input_shape)
    t1 = time.perf_counter()
    bindings, pointer_table, allocations = allocate_bindings(engine, context)
    try:
        outputs = execute(context, bindings, pointer_table, input_tensor)
    finally:
        for allocation in allocations:
            allocation.close()
    t2 = time.perf_counter()
    output_name, output_tensor = next(iter(outputs.items()))
    detections = decode_yolov8(output_tensor, letterbox_info, args.confidence, args.iou, args.max_detections)
    t3 = time.perf_counter()

    annotated = draw_detections(image, detections)
    output_image = args.output_dir / f"{args.image.stem}_yolov8_trt_python.jpg"
    output_json = args.output_dir / "detections.json"
    if not cv2.imwrite(str(output_image), annotated):
        raise RuntimeError(f"failed to write output image: {output_image}")

    report: dict[str, object] = {
        "engine": str(args.engine),
        "image": str(args.image),
        "input_name": input_name,
        "input_shape": list(input_shape),
        "output_name": output_name,
        "output_shape": list(output_tensor.shape),
        "letterbox": asdict(letterbox_info),
        "thresholds": {"confidence": args.confidence, "iou": args.iou},
        "latency_ms": {
            "preprocess": (t1 - t0) * 1000.0,
            "inference": (t2 - t1) * 1000.0,
            "postprocess": (t3 - t2) * 1000.0,
            "total": (t3 - t0) * 1000.0,
        },
        "detections": [asdict(det) for det in detections],
        "output_image": str(output_image),
    }
    if args.reference:
        report["ultralytics_reference"] = run_ultralytics_reference(
            args.weights,
            args.image,
            args.confidence,
            args.iou,
            args.max_detections,
        )

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Engine: {args.engine}")
    print(f"Image: {args.image}")
    print(f"Input: {input_name} {input_shape}")
    print(f"Output: {output_name} {output_tensor.shape}")
    print(f"Detections: {len(detections)}")
    for det in detections[:10]:
        print(f"  {det.class_name:>12s} {det.confidence:.3f} box={det.box_xyxy}")
    print(f"Latency ms: preprocess={report['latency_ms']['preprocess']:.2f}, "
          f"inference={report['latency_ms']['inference']:.2f}, "
          f"postprocess={report['latency_ms']['postprocess']:.2f}, "
          f"total={report['latency_ms']['total']:.2f}")
    print(f"Output image: {output_image}")
    print(f"Detections JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
