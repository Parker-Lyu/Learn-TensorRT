# 03 - OpenCV Preprocess

This lesson implements YOLO-style image preprocessing in C++ with OpenCV.

Goal: understand exactly what happens before an image tensor is passed into a TensorRT engine.

This lesson establishes the input contract used by later TensorRT 10.14 lessons: batch is an
explicit tensor dimension, the host buffer is contiguous, and the engine's input tensor metadata—not
TensorRT itself—determines the required shape, data type, layout, color order, and normalization.

Topics:

- Resize
- Letterbox
- BGR to RGB
- Normalize to `float32`
- HWC to CHW
- Batch input buffer layout
- Coordinate mapping from network input space back to the original image

The runnable example deliberately uses the common YOLO contract `float32` RGB NCHW in `[0, 1]`.
Do not assume that contract for an arbitrary engine; inspect the exported model and TensorRT 10.14
engine I/O tensor metadata first.

## Why This Matters

TensorRT engines do not know how your input image was resized or padded. If preprocessing is wrong,
the model may still run, but the detections will be inaccurate or the boxes will map back to the
wrong image coordinates.

This lesson keeps preprocessing on the CPU so the data layout and math are visible before later
lessons move memory copies and preprocessing work closer to CUDA.

## Data Flow

```text
image file
  -> cv::imread BGR uint8 HWC
  -> letterbox resize to 640 x 640 with gray padding
  -> BGR to RGB
  -> normalize uint8 [0, 255] to float32 [0, 1]
  -> HWC to NCHW
  -> contiguous float buffer shaped [1, 3, 640, 640]
```

The output tensor is laid out as:

```text
NCHW index = n * C * H * W + c * H * W + y * W + x
```

For a single image, the first `H * W` values are the red channel, the next `H * W` values are the
green channel, and the final `H * W` values are the blue channel.

## Coordinate Mapping

Letterbox keeps the original aspect ratio:

```text
scale = min(input_width / original_width, input_height / original_height)
resized_width = round(original_width * scale)
resized_height = round(original_height * scale)
pad_left = (input_width - resized_width) / 2
pad_top = (input_height - resized_height) / 2
```

To map a model box from network input coordinates back to original image coordinates:

```text
original_x = (input_x - pad_left) / scale
original_y = (input_y - pad_top) / scale
```

The code clamps mapped boxes to the original image boundary.

## Directory Layout

- `CMakeLists.txt`: target-based build file for this runnable lesson.
- `include/preprocess.hpp`: small data structures and preprocessing function declarations.
- `src/main.cpp`: command-line entry point, image loading, output writing, and printed diagnostics.
- `src/preprocess.cpp`: letterbox, normalization, HWC-to-CHW conversion, and coordinate mapping.
- `tests/preprocess_tests.cpp`: focused checks for layout, mapping, invalid inputs, odd padding, and
  extreme aspect ratios.
- `../assets/img.jpeg`: shared sample image stored at the repository root.

The lesson keeps preprocessing declarations, implementation, and command-line wiring separate so
each part can evolve without turning the example into one large source file.

## Build

Run the lesson commands from its directory:

```bash
cd 03_opencv_preprocess
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Run

Use the shared sample image:

```bash
./build/opencv_preprocess
```

Use your own image:

```bash
./build/opencv_preprocess /path/to/your/image.jpg
```

Choose a different input size and output folder:

```bash
./build/opencv_preprocess ../assets/img.jpeg 640 384 outputs_640x384
```

## Outputs

The program writes:

- `outputs/letterbox_debug.jpg`: the resized and padded image that visually confirms letterbox behavior.
- `outputs/input_tensor_nchw_float32.bin`: the full contiguous `float32` tensor buffer.
- `outputs/input_tensor_preview.txt`: shape metadata and the first tensor values for quick inspection.

The program also prints:

- original image size
- network input size
- resized image size
- scale
- padding
- tensor shape and byte size
- one sample box mapped from network input coordinates back to original image coordinates

Acceptance criteria:

- The program reads an image and writes a debug image or tensor dump.
- The scale and padding values are printed for later box coordinate recovery.
- The focused preprocessing tests pass from a clean build directory.
