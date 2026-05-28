# 03 - OpenCV Preprocess

Goal: implement YOLO-style image preprocessing in C++.

Topics:

- Resize
- Letterbox
- BGR to RGB
- Normalize to `float32`
- HWC to CHW
- Batch input buffer layout

Acceptance criteria:

- The program reads an image and writes a debug image or tensor dump.
- The scale and padding values are printed for later box coordinate recovery.
