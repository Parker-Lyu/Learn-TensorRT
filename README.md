# Learn TensorRT

A small C++17 learning repository for TensorRT.

Each lesson is stored in its own folder. The folder name starts with a lesson number, followed by a short lesson topic.

## Lessons

### 01 - Hello World

Folder:

```bash
01_hello_world
```

Build and run:

```bash
cd 01_hello_world
cmake -S . -B build
cmake --build build
./build/hello_world
```

### 02 - OpenCV Read And Show Image

Folder:

```bash
02_opencv_read_show_image
```

Build and run:

```bash
cd 02_opencv_read_show_image
cmake -S . -B build
cmake --build build
./build/opencv_read_show_image
```

## General Build Pattern

For each lesson, enter the lesson folder first:

```bash
cd <lesson_folder>
```

Configure the CMake project:

```bash
cmake -S . -B build
```

Build the executable:

```bash
cmake --build build
```

Run the lesson executable from the `build` folder or by using its path:

```bash
./build/<executable_name>
```

## Requirements

- A C++17 compiler
- CMake 3.10 or newer
