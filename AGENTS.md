# Project Instructions For Codex

This repository is a C++17 learning project for TensorRT deployment. It should teach good
engineering habits from the beginning, not quick demo shortcuts.

## Course Style

- Each lesson should produce one runnable artifact and one concise README.
- Lesson code should be easy to read, but still structured like code that can evolve.
- Prefer small, focused files over one large file when a lesson has multiple concepts.
- Shared images, models, and reusable resources belong in the root `assets/` directory.
- Generated outputs should go to local output folders and stay out of git.

## Practice Modules

- For lessons that need hand-written practice, keep both a reference lesson and a practice lesson
  on the main branch.
- The reference lesson folder, such as `03_opencv_preprocess/`, contains the complete runnable
  implementation.
- The practice folder, such as `03_opencv_preprocess_practice/`, contains a buildable starter,
  TODO comments, and a step-by-step README.
- The reference and practice folders for the same lesson should mirror each other's directory
  structure whenever practical, for example `CMakeLists.txt`, `README.md`, `include/*.hpp`, and
  `src/*.cpp`.
- Practice code should build and run before the TODOs are implemented, so learners start from a
  working project skeleton.
- Practice READMEs should explain what to implement step by step, which files to edit, and what
  output to expect after each build/run checkpoint.
- Do not replace a complete lesson with a TODO-only version unless explicitly requested.
- Do not create, switch, or push solution branches for the user unless explicitly requested.

## C++ Style

- Use modern C++17 and target-based CMake.
- Prefer RAII, standard containers, and clear ownership over manual memory management.
- Validate inputs in public helper functions, not only in `main`.
- Check file and resource errors explicitly.
- Use `const`, `static_cast`, `<algorithm>` utilities, and standard library facilities where appropriate.
- Keep comments useful for learning intent or non-obvious logic; avoid comments that restate the code.
- Avoid silent failures, hidden assumptions, and clever code that hurts readability.

## CMake Style

- Each C++ lesson should have its own `CMakeLists.txt`.
- Use `target_compile_features(<target> PRIVATE cxx_std_17)`.
- Prefer target-specific include paths, libraries, and properties.
- Keep build artifacts in ignored build directories.

## Verification

Before finishing code changes, whenever practical:

- Build the touched lesson.
- Run the lesson executable or a focused smoke test.
- Run `git diff --check`.
- Tell the user what was verified and what was not.
