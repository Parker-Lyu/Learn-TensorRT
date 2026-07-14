# 23 - C++ Deployment Interview Katas

This lesson collects compact C++17 implementations that commonly appear in CV deployment
interviews: IoU, class-aware NMS, bilinear sampling, letterbox coordinate mapping, HWC-to-CHW,
stable Top-K, a ring buffer, a closeable bounded queue, and a move-only CUDA buffer.

```bash
cmake -S 23_cpp_interview_katas -B 23_cpp_interview_katas/build
cmake --build 23_cpp_interview_katas/build -j
ctest --test-dir 23_cpp_interview_katas/build --output-on-failure
./23_cpp_interview_katas/build/katas_demo
```

Tests cover empty NMS, degenerate boxes, overlapping boxes from different classes, interpolation
boundaries, extreme letterbox clamping, Top-K ties, ring wrap/full/empty behavior, queue close waking
a blocked producer, and CUDA ownership transfer. Practice rewriting one group from memory after its
related core lesson; explain validation and complexity before optimizing syntax.
