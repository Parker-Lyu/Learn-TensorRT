# Project Instructions For Codex

This repository is a TensorRT deployment learning project. ISO C++17 is the primary implementation
language for inference and systems lessons, while Python and shell scripts support model export,
validation, profiling, and report generation. It should teach good engineering habits from the
beginning, not quick demo shortcuts.

## Course Baseline

- Use `nvcr.io/nvidia/pytorch:25.11-py3` as the single upstream development image.
- Target TensorRT 10.14 (`10.14.1.48` in the pinned image) and CUDA Toolkit 13.0.
- Compile host C++ and CUDA C++ as ISO C++17 without GNU language extensions.
- Use the PyTorch and ModelOpt stack supplied by the development image for model export and
  explicit-Q/DQ workflows.
- Build engines, timing caches, references, and benchmark evidence with TensorRT 10.14 in the pinned
  development environment.

## Course Style

- Keep `README.md`, `docs/learning_roadmap.md`, and `docs/coverage_matrix.md` written for third-party
  learners taking the course; keep agent-only implementation instructions in `AGENTS.md`.
- Each implementation lesson should produce a runnable artifact and one concise README. Reporting
  checkpoints should produce a reproducible report and document how it was generated.
- Lesson code should be easy to read, but still structured like code that can evolve.
- Prefer small, focused files over one large file when a lesson has multiple concepts.
- Shared images, models, and reusable resources belong in the root `assets/` directory.
- When a lesson needs an input image, use `assets/img.jpeg` by default.
- Do not display images in GUI windows (for example, with `cv::imshow`); save images learners need
  to inspect in the lesson's `output/` directory instead.
- Transient build products, TensorRT engines, profiling captures, generated images, and local
  benchmark outputs should go to ignored output directories.
- Curated reports, small test fixtures, manifests, and reproducibility metadata may be committed
  when they are intentional lesson deliverables.

## Lesson Modules

- Keep lesson directories as complete runnable implementations.
- Course 00 documents the shared runtime environment; do not repeat it in every later lesson.
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

## Lesson README Structure

- Treat `docs/learning_roadmap.md` as the course-level contract. A lesson README must implement the
  roadmap's purpose, deliverables, and acceptance boundary without silently expanding or narrowing
  them.
- Use these learner-facing sections in this order when they apply:
  1. `Purpose`
  2. `Prerequisites`
  3. `Deliverables`
  4. `Build` for compiled lessons or `Setup` for dependency preparation
  5. `Run` for executable lessons or `Generate the Report` for reporting checkpoints
  6. `Outputs`
  7. `Tests`
  8. `Checkpoints`
- `Purpose`, `Prerequisites`, `Deliverables`, the primary execution section, `Outputs`, and
  `Checkpoints` are expected in every lesson README. Use `None` with a brief explanation when a
  prerequisite, build step, generated output, or automated test genuinely does not exist; do not
  invent an empty command merely to satisfy the format.
- Keep lesson-specific explanations under descriptive optional headings such as `Design`, `Data
  Flow`, `Experiments`, `Failure Semantics`, `Troubleshooting`, or `Appendix`. These headings do not
  replace the standard execution sections.
- Put commands in the section that owns them: dependency and cross-lesson preparation under
  `Prerequisites` or `Setup`, compilation under `Build`, the main learner workflow under `Run`, and
  automated checks under `Tests`.
- `Outputs` must distinguish committed deliverables from ignored, environment-specific generated
  artifacts. Never imply that an engine, benchmark, server run, sanitizer run, or target-hardware
  result exists unless it was actually produced.
- `Checkpoints` are learner exercises and review questions. Do not use them as a substitute for
  objective roadmap acceptance criteria or automated tests.

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

- Use ISO C++17 and target-based CMake. Do not raise the repository-wide language standard without
  an explicit course decision.
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
- Set `CXX_EXTENSIONS OFF`. For targets that compile `.cu` sources, also require
  `CUDA_STANDARD 17`, `CUDA_STANDARD_REQUIRED ON`, and `CUDA_EXTENSIONS OFF`.
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

- Use the development environment derived from `nvcr.io/nvidia/pytorch:25.11-py3` as the default
  build and test environment.
- Do not install CUDA, TensorRT, or OpenCV directly on the host unless the user explicitly asks for a
  host-native experiment.
- A repository development Dockerfile may add course dependencies such as OpenCV, ONNX Runtime, and
  test tools, but it must not silently replace the base image's CUDA, TensorRT, or PyTorch stack.
- Add runtime-delivery Dockerfiles only when a lesson explicitly targets packaging; keep early
  lessons focused on using the reproducible development environment.
- When Docker packaging is introduced, separate development images from lean runtime images and
  document what files must be copied into the runtime image.

## Dependency And Compatibility Style

- Preserve compatibility with TensorRT 10.14, CUDA Toolkit 13.0, and ISO C++17 in the pinned
  development image unless a lesson explicitly studies portability to another environment.
- Do not silently upgrade TensorRT, CUDA, OpenCV, ONNX, or Python package versions.
- Avoid adding third-party dependencies when the standard library or an existing project dependency
  is sufficient.
- Treat serialized TensorRT engines as environment-specific generated artifacts; do not commit them
  unless explicitly requested.
- Generate engines, timing caches, golden outputs, and performance baselines in the pinned
  development environment, and record the runtime, CUDA, GPU, driver, and container identity.

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
