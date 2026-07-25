# 12a - Precision and performance report

This checkpoint consumes Lesson 12's TensorRT 10.14 FP32, FP16, and quality-passing Q/DQ INT8
engines. It records at least 100 synchronized `trtexec` samples, wall-time throughput, detection
metrics, release-gate state, engine hashes, dataset identity, and TensorRT 10.14 Engine Inspector evidence without copying numbers by
hand.

Run in the pinned course container from the repository root:

```bash
python3 -m unittest discover -s 12a_precision_performance_report/tests -v
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

Complete `12_yolov8_int8_quantization_engineering/docs/reproduction.md` first. The generator rejects
an INT8 candidate that failed its gate and rejects mismatched manifest, engine, runtime, or sample
identities. The report also consumes Lesson 12's TensorRT 10.14 Engine Inspector audit. Raw timing
captures remain in ignored output directories.

An evidence-backed run from the pinned environment is committed at
[`reports/12a_precision_performance.md`](../reports/12a_precision_performance.md). It records
TensorRT 10.14.1, the matched engine identities, quality results, and the measured FP16 deployment
decision. Learners must regenerate it when their GPU, driver, model, or dataset identity differs.

## Decision policy

1. Evaluate PyTorch FP32/FP16 and TensorRT FP32/FP16 on the fixed validation split.
2. Evaluate Q/DQ INT8 with the unchanged quality contract.
3. Benchmark INT8 only after it passes both PyTorch-FP32-relative and TensorRT-FP16-relative gates.
4. Select INT8 only when matched measurements show a meaningful benefit over FP16.
5. Regenerate every affected artifact after a model, dataset, preprocessing, runtime, or engine
   identity changes.
