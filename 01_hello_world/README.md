# 01 - Hello World

Goal: build confidence with a minimal C++17 and CMake project.

Topics:

- C++17 executable
- Basic `CMakeLists.txt`
- Target creation
- C++ standard settings
- Command-line configure, build, and run flow

Build:

```bash
cmake -S . -B build
cmake --build build
```

Run:

```bash
./build/hello_world
```

Acceptance criteria:

- You can configure, build, and run a small C++ executable.
- You understand `CMakeLists.txt`, target creation, and C++ standard settings.
