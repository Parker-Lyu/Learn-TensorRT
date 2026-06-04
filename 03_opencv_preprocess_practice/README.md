# 03 Practice - OpenCV Preprocess

This is the hand-written practice version of `../03_opencv_preprocess`.

The goal is to rebuild the same preprocessing pipeline step by step without copying the complete
lesson implementation. This folder is intentionally kept on `main` as a public exercise module, so
learners can write their own practice solution here directly without switching branches. If you want
to preserve your work separately, commit your changes locally or keep a personal copy; compare
against `../03_opencv_preprocess` only after finishing each step.

## Final Target

Build a runnable C++17 OpenCV program that:

- reads an image from disk
- letterboxes it to a network input size
- converts BGR uint8 HWC pixels into RGB float32 NCHW tensor layout
- saves a debug letterbox image
- saves a small tensor preview text file
- prints scale, padding, tensor shape, and one sample mapped box

The final data flow should be:

```text
image file
  -> cv::imread BGR uint8 HWC
  -> letterbox resize to 640 x 640 with gray padding
  -> BGR to RGB
  -> normalize uint8 [0, 255] to float32 [0, 1]
  -> HWC to NCHW
  -> contiguous float buffer shaped [1, 3, 640, 640]
```

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/opencv_preprocess_practice
```

Use your own image and input shape:

```bash
./build/opencv_preprocess_practice ../assets/dog.webp 640 384 outputs_640x384
```

## Step-by-step Practice Plan

### Step 0 - Run the starter

Do not edit code yet. Build and run the starter.

Expected result:

```text
Image path:     ../assets/dog.webp
Network input:  640 x 640
Output dir:     outputs
Loaded image:   <width> x <height>

TODO 1: implement letterbox_image(...)
TODO 2: implement append_nchw_rgb_float(...)
TODO 3: implement preprocess_batch_to_nchw(...)
TODO 4: implement map_box_to_original_image(...)
```

This confirms your CMake target, OpenCV link, argument parsing, image loading, and output directory
creation are working before you touch preprocessing.

### Step 1 - Implement `letterbox_image`

Edit `src/preprocess.cpp`.

Write:

- input validation for empty images, non-3-channel images, and non-positive target sizes
- `scale = min(input_width / original_width, input_height / original_height)`
- rounded resized width and height
- left, top, right, and bottom padding
- `cv::resize`
- a gray `CV_8UC3` canvas with value `(114, 114, 114)`
- `resized.copyTo(letterboxed(roi))`

Then update `src/main.cpp` so it calls `letterbox_image` and writes:

```text
outputs/letterbox_debug.jpg
```

Expected result after build/run:

```text
Letterbox image: outputs/letterbox_debug.jpg
Scale:           <number>
Padding:         left=<n>, top=<n>, right=<n>, bottom=<n>
```

Open the debug image and check that the original image keeps its aspect ratio.

### Step 2 - Implement `append_nchw_rgb_float`

Edit the helper inside `src/preprocess.cpp`.

Write nested loops over `y` and `x`:

- read `cv::Vec3b` as BGR
- compute `hw_index = y * width + x`
- write red into channel 0
- write green into channel 1
- write blue into channel 2
- normalize each value with `1.0F / 255.0F`

Expected result after build/run:

```text
TODO 3: implement preprocess_batch_to_nchw(...)
```

At this point the helper may be complete, but the public batch function still controls whether the
program can produce a tensor.

### Step 3 - Implement `preprocess_batch_to_nchw`

Edit `src/preprocess.cpp`.

Write:

- validation for an empty image list and invalid input size
- result metadata: batch size, channels, height, width
- `input_tensor.resize(batch * channels * height * width)`
- loop over each image
- call `letterbox_image`
- save each `LetterboxInfo`
- call `append_nchw_rgb_float`

Then update `src/main.cpp` so it writes:

```text
outputs/input_tensor_preview.txt
```

Expected result after build/run:

```text
Tensor shape:    [1, 3, 640, 640]
Tensor values:   1228800 float32 values
Tensor preview:  outputs/input_tensor_preview.txt
```

The preview file should start with:

```text
shape: [1, 3, 640, 640]
layout: NCHW
dtype: float32
color: RGB
range: [0, 1]
```

### Step 4 - Implement `map_box_to_original_image`

Edit `src/preprocess.cpp`.

Write:

- subtract letterbox padding from `x1`, `y1`, `x2`, `y2`
- divide by `scale`
- clamp coordinates to the original image boundary
- return `cv::Rect2f(x1, y1, x2 - x1, y2 - y1)`

Then update `src/main.cpp` so it prints a sample box mapping.

Expected result after build/run:

```text
Sample box in network input: x=220, y=180, w=160, h=120
Mapped back to original:     x=<number>, y=<number>, w=<number>, h=<number>
```

### Step 5 - Compare with the reference lesson

After your practice version runs end to end, compare it with:

```text
../03_opencv_preprocess
```

Useful checks:

- Are your scale and padding values the same?
- Is your tensor shape the same?
- Are your first preview values close?
- Does your mapped box match?

Small formatting differences are fine. Different preprocessing math is not.

## TODO Map

- `include/preprocess.hpp`: read the data structures and function contracts first.
- `src/preprocess.cpp`: implement preprocessing math.
- `src/main.cpp`: wire each completed function into the runnable artifact.

Acceptance criteria:

- The starter builds before any TODO is implemented.
- Each step can be verified by rebuilding and running the executable.
- The final practice version produces a debug image, tensor preview, and coordinate mapping output.
