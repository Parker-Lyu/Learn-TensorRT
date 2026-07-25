# 02 - OpenCV Read And Inspect Image

This lesson reads an image from disk with OpenCV and prints its decoded metadata.

Goal: learn image loading, `cv::Mat` metadata, and basic OpenCV project setup.

Topics:

- OpenCV image loading
- Image width and height
- Channel count
- OpenCV data type and per-channel element type
- OpenCV linking with CMake

## Build

```bash
cd 02_opencv_read_image_info
cmake -S . -B build
cmake --build build --parallel
```

## Run

Use the sample image:

```bash
./build/opencv_read_image_info
```

By default, the program reads the shared sample image from `../assets/img.jpeg`.

Use your own image:

```bash
./build/opencv_read_image_info /path/to/your/image.jpg
```

Example output:

```text
Loaded image: ../assets/img.jpeg
Width: 800
Height: 1067
Channels: 3
Data type: CV_8UC3 (uint8 per channel)
```

The program uses `cv::IMREAD_UNCHANGED` so OpenCV preserves the decoded channel count and element
depth instead of forcing every input to an 8-bit, three-channel BGR image. For example, an image may
be reported as `CV_8UC3`, meaning three channels whose elements are unsigned 8-bit integers.

The program returns a nonzero exit status for an unreadable image or invalid command-line
arguments.

## Acceptance criteria:

- You can load an image and explain its dimensions, channel count, and OpenCV data type.
- You can link OpenCV with CMake.
