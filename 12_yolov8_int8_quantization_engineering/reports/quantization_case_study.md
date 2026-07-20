# YOLOv8n Quantization Engineering Case Study

Status: **COMPLETE — DEPLOYMENT RETAINS FP16**

This case study follows one fixed quality contract from TensorRT legacy calibration through ModelOpt
explicit Q/DQ. Candidate speed never overrides a failed quality gate, and a quality-passing INT8
candidate is deployed only when it also beats a version-matched FP16 reference.

## Data Qualification

- Dataset: `coco2017-yolov8n-calibration-v3-val5000-human-labels-v1`.
- Candidate pool: 5000 fixed COCO train2017 images.
- Calibration: 3000 independently coverage-selected images.
- Validation: all 5000 COCO val2017 images with human labels.
- Calibration manifest SHA-256: `38c88bff89757dba6e22c44d30398ae0d17f8bd11ec2c09a867b3e975d339a50`.
- Selection metadata SHA-256: `58fa6d629136ea73e1d94f13b8b115a32f783ade0e5ea8c68090a4ca732b480e`.
- Historical 1,000-image members selected again: 828; old membership was not forced.
- Preprocessing parity: PASS for all 3,000 calibration images with byte-identical FP32 tensors.

## Quantization Evolution

| Candidate | Runtime | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Legacy Entropy INT8+FP16 | 8.6.1 | 0.3150 | 0.4499 | 0.0432 | 0.7834 | FAIL |
| Legacy MinMax INT8+FP16 | 8.6.1 | 0.3439 | 0.4889 | 0.0430 | 0.7930 | FAIL |
| MinMax + complete detection head FP16 | 8.6.1 | 0.3463 | 0.4895 | 0.0427 | 0.7930 | FAIL |
| ModelOpt Q/DQ INT8+FP16 | 8.6.1 | 0.3452 | 0.4946 | 0.0452 | 0.7990 | PASS |
| ModelOpt native-FP16 Q/DQ INT8+FP16 | 10.14.1.48 | 0.3454 | 0.4946 | 0.0459 | 0.7998 | PASS |

Entropy loses too much task accuracy. MinMax recovers most of it but misses the fixed mAP50 gate.
Moving the complete detection head to FP16 improves mAP50-95 but still misses mAP50, so the remaining
regression is not explained by detection-head INT8 alone. Explicit Q/DQ passes the unchanged gate in
both TensorRT 8.6 and TensorRT 10.14.

## Matched TensorRT 10 Performance

Methodology: 500 ms warmup, 120 measured iterations, one inference stream, and synchronized
`trtexec --exportTimes` evidence.

| Engine | Mean latency (ms) | P90 (ms) | GPU compute mean (ms) | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 5.161 | 5.186 | 3.982 | 248.961 |
| FP16 | 2.750 | 2.765 | 1.556 | 636.729 |
| Q/DQ INT8+FP16 | 3.089 | 3.103 | 1.898 | 522.188 |

Against FP16, Q/DQ INT8+FP16 has `-18.0%` throughput,
`+21.9%` GPU compute time, and
`+12.3%` mean latency. It is quality-eligible but slower.

## Why INT8 Can Be Slower

Inspector evidence reports 44 INT8-output
compute layers, 49 FP16-output compute layers,
and 90 reformats, including 41 with Q/DQ
origin. Likely contributors include incomplete INT8 kernel coverage, Q/DQ conversion boundaries,
layout changes, memory traffic, tactic selection, and batch-1 overhead. A deeper investigation could
use per-layer `trtexec` profiles, Engine Inspector formats, Nsight Systems, and Nsight Compute, but
that root-cause study is intentionally outside this lesson.

## Deployment Decision

Retain FP16. Both explicit-Q/DQ candidates prove that INT8 quality can be recovered without changing
the gate, but the final TensorRT 10 candidate does not provide a matched performance benefit. The
engineering lesson is that INT8 is a candidate technology, not an automatic deployment decision.
