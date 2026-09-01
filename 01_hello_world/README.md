# 01 - Hello World

## Purpose

- Build confidence with C++17 and CMake.

## Prerequisites

- Complete `00_environment_check` and enter the pinned development container.
- No generated artifact from an earlier lesson is required.

## Deliverables

- `hello_world` CMake executable
- `CMakeLists.txt` enforcing ISO C++17
- `README.md` with clean build and run commands

## What the CMake settings guarantee

- `target_compile_features(... cxx_std_17)` and `CXX_STANDARD_REQUIRED ON` require C++17 for this
  executable rather than permitting fallback to an older standard.
- `CXX_EXTENSIONS OFF` disables compiler-specific GNU language extensions, keeping the lesson on
  ISO C++17.
- `CMAKE_EXPORT_COMPILE_COMMANDS` writes `compile_commands.json` for editor and analysis tooling.

This lesson has no TensorRT API usage or linkage.
Its relevant compatibility requirement is that the host code continues to compile as ISO C++17.

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 01_hello_world -B 01_hello_world/build
cmake --build 01_hello_world/build --parallel
```

The generated build directory is ignored.

## Run

Run the executable produced by the build. This starts the program and writes its greeting and
success message to standard output:

```bash
./01_hello_world/build/hello_world
```

```text
Hello World
Congratulations! You have successfully compiled and run your first C++ program.
```

## Outputs

- `01_hello_world/build/hello_world`: ignored build artifact.
- The program prints the documented greeting and success message to standard output.

## Checkpoints

- You can configure, build, and run a small C++ executable.
- You understand `CMakeLists.txt`, target creation, and C++ standard settings.
