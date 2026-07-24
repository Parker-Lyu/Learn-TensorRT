# 02 - OpenCV Read And Show Image

This lesson reads an image from disk with OpenCV and displays it in a window.

Goal: learn image loading and basic OpenCV project setup.

Topics:

- OpenCV image loading
- Image dimensions and channel inspection
- Basic OpenCV display flow
- OpenCV linking with CMake

This lesson does not call TensorRT or CUDA, so it has no TensorRT API compatibility surface. It
should still be built in the course development image so the compiler and dependency environment
remain reproducible.

## Prerequisites

Use the repository development image derived from `nvcr.io/nvidia/pytorch:25.11-py3`. Its
`libopencv-dev` package provides the OpenCV C++ headers, libraries, image codecs, and HighGUI module
used by this lesson. See the root `README.md` and `00_environment_check/agent_env_setup.md` for the
container setup.

## Build

```bash
cd 02_opencv_read_show_image
cmake -S . -B build
cmake --build build --parallel
```

## Run

Use the sample image:

```bash
./build/opencv_read_show_image
```

By default, the program reads the shared sample image from `../assets/dog.webp`.

Use your own image:

```bash
./build/opencv_read_show_image /path/to/your/image.jpg
```

The default run opens an OpenCV HighGUI window and waits for a key press. The container must have
access to a graphical display server for this mode; merely setting a stale `DISPLAY` value does not
guarantee that the server is reachable.

For a headless container or CI environment, verify image decoding and metadata without opening a
window:

```bash
./build/opencv_read_show_image --no-display
```

The program prints the decoded width, height, and channel count. It returns a nonzero exit status
for an unreadable image, invalid command-line arguments, or an unavailable requested display.

Acceptance criteria:

- You can load an image, check dimensions, and display or save output.
- You can link OpenCV with CMake.
