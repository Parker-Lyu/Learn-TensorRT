# 17a - Pipeline Performance and Reliability Report

This checkpoint converts lessons 13–17 into reproducible load, latency, fairness, memory, fault,
and sanitizer evidence. It generates `reports/17a_pipeline_performance.md`; it does not silently
mark a short smoke run as a completed 30-minute soak.

Build lessons 13, 15, 16, and 17 first, then run from the repository root:

```bash
python3 -m unittest discover -s 17a_pipeline_performance_report/tests -v
python3 17a_pipeline_performance_report/collect_pipeline_evidence.py
python3 17a_pipeline_performance_report/generate_report.py
```

For the formal checkpoint:

```bash
python3 17a_pipeline_performance_report/collect_pipeline_evidence.py \
  --soak-minutes 30 \
  --restart-cycles 100
python3 17a_pipeline_performance_report/generate_report.py
```

The collector runs 100 lifecycle cycles, four fault cases, CUDA memcheck, and the available TSAN
binary. Host RSS and device memory are sampled around the single-stream load. The evidence also
records
the pinned development image, RTX GPU identity, compute capability, driver, memory, and TensorRT
version so performance results remain tied to the measured platform. Generated raw evidence
stays ignored; the concise report is committed.
