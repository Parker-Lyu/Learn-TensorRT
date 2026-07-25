# 01 - Hello World

Goal: build confidence with a minimal C++17 and CMake project.

Topics:

- C++17 executable
- Basic `CMakeLists.txt`
- Target creation
- C++ standard settings
- Command-line configure, build, and run flow

## Build and run

Start in the repository root. Remove the ignored build directory first when you want a clean,
from-scratch build:

```bash
rm -rf 01_hello_world/build
cmake -S 01_hello_world -B 01_hello_world/build
cmake --build 01_hello_world/build --parallel
```

Run the resulting artifact:

```bash
./01_hello_world/build/hello_world
```

Expected output:

```text
Hello World
Congratulations! You have successfully compiled and run your first C++ program.
```

## What the CMake settings guarantee

- `target_compile_features(... cxx_std_17)` and `CXX_STANDARD_REQUIRED ON` require C++17 for this
  executable rather than permitting fallback to an older standard.
- `CXX_EXTENSIONS OFF` disables compiler-specific GNU language extensions, keeping the lesson on
  ISO C++17.
- `CMAKE_EXPORT_COMPILE_COMMANDS` writes `compile_commands.json` for editor and analysis tooling.

No TensorRT migration is required in this lesson because it has no TensorRT API usage or linkage.
Its relevant compatibility requirement is that the host code continues to compile as ISO C++17.

## Acceptance criteria

- You can configure, build, and run a small C++ executable.
- You understand `CMakeLists.txt`, target creation, and C++ standard settings.
