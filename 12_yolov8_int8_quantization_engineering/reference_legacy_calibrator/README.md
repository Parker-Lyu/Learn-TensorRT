# TensorRT legacy entropy calibrator reference

This directory demonstrates `IInt8EntropyCalibrator2` with TensorRT 10.14 for API comparison. The
interface drives implicit INT8 calibration and is not the recommended Lesson 12 deployment path.
Use `../modelopt/export_qdq.py` and an explicit Q/DQ graph for the course implementation.

The builder consumes the same calibration manifest and preprocessing code as the Q/DQ workflow. Its
engine can be passed to `../compare_engines.py` with experiment ID
`reference_entropy_calibrator`, so any comparison uses the unchanged validation contract. Generated
engines and caches belong under `../outputs/legacy_entropy/` and must not be committed.
