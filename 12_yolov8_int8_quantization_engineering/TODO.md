# Lesson 12 Follow-up

## Completed In Code

- `configs/quality_contract.json` is now the executable source for evaluation settings and release
  thresholds; reports and reference reuse bind its SHA-256 identity.
- Candidate evaluation now validates the declared experiment, engine build metadata, source Q/DQ
  model metadata, TensorRT runtime, dataset manifest, and engine hash.
- Candidate-only evaluation now reuses a validated reference bundle instead of an unbound report.
- `detection_head_fp16` now follows the complete 67-layer detection-head data path through the
  prediction towers, reshape/concat, DFL, decoding, sigmoid, and output assembly.
- Engine Inspector evidence now rejects any non-FP16 internal detection-head output while allowing
  only the explicit external FP32 output boundary.
- The rebuilt TensorRT 8.6 full-head engine passed metadata validation, Inspector validation, and a
  `trtexec` inference smoke test. Lesson CPU tests pass in `trt_dev`.

## TODO Before Publishing New Results

- [ ] In `trt_dev`, rerun the complete TensorRT 8.6 reference evaluation and create the new
  reference bundle. The attempted rerun was stopped before completion and did not overwrite the old
  report.
- [ ] In `trt_dev`, reevaluate Legacy MinMax, the rebuilt complete-head FP16 candidate, and the TRT8
  ModelOpt Q/DQ candidate against the new bundle.
- [ ] In `learn-tensorrt-modelopt`, rerun the TensorRT 10 quality evaluation with the new contract and
  experiment identities, then recreate its reference bundle. Existing engines and performance
  captures may be reused only after their identities pass the new validators.
- [ ] Regenerate `reports/quantization_results.json` and `reports/quantization_case_study.md` from the
  new evidence. The currently committed metrics, especially the old tower-only mixed-precision
  result, must be treated as stale until this step completes.
- [ ] Update the README recorded outcome if the complete-head candidate changes the quality result.
