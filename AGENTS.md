# Project Instructions For Codex

This repository is a C++17 learning project for TensorRT deployment. It should teach good
engineering habits from the beginning, not quick demo shortcuts.

## Course Style

- Each lesson should produce one runnable artifact and one concise README.
- Lesson code should be easy to read, but still structured like code that can evolve.
- Prefer small, focused files over one large file when a lesson has multiple concepts.
- Shared images, models, and reusable resources belong in the root `assets/` directory.
- Generated outputs should go to local output folders and stay out of git.

## Lesson Modules

- Keep lessons as complete runnable implementations on the main branch.
- Do not add separate `_practice` lesson folders or TODO-only starter copies unless explicitly
  requested.
- When a lesson benefits from hands-on guidance, put concise checkpoints or experiments in that
  lesson's README without duplicating the lesson directory.
- Do not replace a complete lesson with a TODO-only version unless explicitly requested.
- Do not create, switch, or push solution branches for the user unless explicitly requested.

## Industrial Code Expectations

- Treat lesson code as production-style teaching code, not throwaway demos.
- Prefer explicit ownership, error handling, and resource lifetime boundaries.
- Keep APIs small, testable, and reusable across later TensorRT lessons.
- Make assumptions visible in names, validation, comments, or README notes.
- Avoid global mutable state, hidden side effects, magic constants, and hard-coded local paths.
- Structure code so later lessons can extend it toward long-running inference services.

## C++ Style

- Use modern C++17 and target-based CMake.
- Prefer RAII, standard containers, and clear ownership over manual memory management.
- Validate inputs in public helper functions, not only in `main`.
- Check file and resource errors explicitly.
- Use `const`, `static_cast`, `<algorithm>` utilities, and standard library facilities where appropriate.
- Add suitable comments for teaching code, especially around learning intent, data layout, coordinate
  transforms, resource ownership, synchronization, and non-obvious API choices.
- Keep comments concise and useful; avoid line-by-line narration or comments that merely restate the
  code.
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
