# Project Instructions For Codex

This repository is a TensorRT deployment learning project. C++17 is the primary implementation
language for inference and systems lessons, while Python and shell scripts support model export,
validation, profiling, and report generation. It should teach good engineering habits from the
beginning, not quick demo shortcuts.

## Course Style

- Each implementation lesson should produce a runnable artifact and one concise README. Reporting
  checkpoints should produce a reproducible report and document how it was generated.
- Lesson code should be easy to read, but still structured like code that can evolve.
- Prefer small, focused files over one large file when a lesson has multiple concepts.
- Shared images, models, and reusable resources belong in the root `assets/` directory.
- Transient build products, TensorRT engines, profiling captures, generated images, and local
  benchmark outputs should go to ignored output directories.
- Curated reports, small test fixtures, manifests, and reproducibility metadata may be committed
  when they are intentional lesson deliverables.

## Lesson Modules

- Keep lesson directories as complete runnable implementations in the normal course history.
- Design each lesson so a third party can reproduce it from scratch using only the repository at
  that revision and its documented external prerequisites. A lesson may depend on earlier lessons,
  but must not depend on files, generated artifacts, undocumented local state, or other resources
  that existed during development and were later deleted. Document any cross-lesson dependencies
  and the commands needed to reproduce the lesson in its README.
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
- When a lesson introduces reusable deployment logic, prefer library-style modules with clear
  headers, source files, and narrow public APIs before wiring them into `main`.
- Apply production-style practices proportionally to the lesson objective. Do not introduce
  abstractions, dependencies, or framework layers that are not yet needed.
- Make assumptions visible in names, validation, comments, or README notes.
- Avoid global mutable state, hidden side effects, magic constants, and hard-coded local paths.
- Structure code so later lessons can extend it toward long-running inference services.
- Preserve a path from lesson code to a final portfolio project with separable preprocessing,
  inference, postprocessing, pipeline, and reporting components.

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
- For lessons with reusable components, build those components as explicit library targets and link
  a small executable target on top.
- Prefer modular CMake that can later grow into root-level `cmake/`, `src/`, and `tests/`
  organization without rewriting the lesson code.
- Keep build artifacts in ignored build directories.

## Testing Style

- For C++ lessons that implement reusable algorithms or resource wrappers, add focused tests when
  practical, especially for preprocessing, postprocessing, queues, batching, and RAII behavior.
- Prefer Google Test for multi-case C++ tests once a lesson grows beyond a single smoke-test
  executable.
- Include defensive cases for invalid inputs, empty data, extreme image aspect ratios, boundary
  coordinates, overlapping boxes, and resource-initialization failures where relevant.
- Keep tests runnable from the lesson build directory and document the command in that lesson's
  README.
- Do not claim coverage percentages unless the repository actually measures them.

## Container And Delivery Style

- Use the pinned TensorRT development container from `00_environment_check` as the default build and
  test environment.
- Do not install CUDA, TensorRT, or OpenCV directly on the host unless the user explicitly asks for a
  host-native experiment.
- Add Dockerfiles only when a lesson explicitly targets packaging or runtime delivery; keep early
  lessons focused on container usage rather than image-authoring ceremony.
- When Docker packaging is introduced, separate development images from lean runtime images and
  document what files must be copied into the runtime image.

## Dependency And Compatibility Style

- Preserve compatibility with the pinned TensorRT development container unless a lesson explicitly
  studies a newer version.
- Do not silently upgrade TensorRT, CUDA, OpenCV, ONNX, or Python package versions.
- Avoid adding third-party dependencies when the standard library or an existing project dependency
  is sufficient.
- Treat serialized TensorRT engines as environment-specific generated artifacts; do not commit them
  unless explicitly requested.

## Verification

Before finishing code changes, whenever practical:

- Build the touched lesson.
- Run the lesson executable or a focused smoke test.
- Run GPU-, CUDA-, and TensorRT-dependent checks inside the pinned development container.
- If the required container, GPU, model, or dependency is unavailable, run the strongest available
  static or CPU-only checks and state the exact limitation.
- Never claim that an executable, test, or benchmark passed unless it was actually run.
- Run `git diff --check`.
- Tell the user what was verified and what was not.
