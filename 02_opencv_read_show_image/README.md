# 02 - OpenCV Read And Show Image

This lesson reads an image from disk with OpenCV and displays it in a window.

Goal: learn image loading and basic OpenCV project setup.

Topics:

- OpenCV image loading
- Image dimensions and channel inspection
- Basic OpenCV display flow
- OpenCV linking with CMake

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

Use the sample image:

```bash
./build/opencv_read_show_image
```

Use your own image:

```bash
./build/opencv_read_show_image /path/to/your/image.jpg
```

Acceptance criteria:

- You can load an image, check dimensions, and display or save output.
- You can link OpenCV with CMake.
