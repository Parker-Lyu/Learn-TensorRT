# 12a - Precision and performance report

This checkpoint consumes Lesson 12's TensorRT 10.14 FP32, FP16, and quality-passing Q/DQ INT8
engines. It records at least 100 synchronized `trtexec` samples, wall-time throughput, detection
metrics, release-gate state, engine hashes, dataset identity, and profiler context without copying
numbers by hand.

Run in the pinned course container from the repository root:

```bash
(cd 11_nsight_performance_diagnosis && python3 profile_yolov8_cpp.py)
python3 -m unittest discover -s 12a_precision_performance_report/tests -v
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

Complete `12_yolov8_int8_quantization_engineering/docs/reproduction.md` first. The generator rejects
an INT8 candidate that failed its gate and rejects mismatched manifest, engine, runtime, or sample
identities. Raw timing and profiler captures remain in ignored output directories.

## Decision policy

1. Evaluate PyTorch FP32/FP16 and TensorRT FP32/FP16 on the fixed validation split.
2. Evaluate Q/DQ INT8 with the unchanged quality contract.
3. Benchmark INT8 only after it passes both PyTorch-FP32-relative and TensorRT-FP16-relative gates.
4. Select INT8 only when matched measurements show a meaningful benefit over FP16.
5. Regenerate every affected artifact after a model, dataset, preprocessing, runtime, or engine
   identity changes.
