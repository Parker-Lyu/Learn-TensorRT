# 24 - Final Portfolio Case Study

This final synthesis links checkpoint evidence, runs the available local test matrix, generates the
five-minute case study, and provides a multi-stage deployment Dockerfile.

```bash
python3 24_final_portfolio_case_study/run_local_checks.py
python3 24_final_portfolio_case_study/generate_case_study.py
python3 -m unittest discover -s 24_final_portfolio_case_study/tests -v
```

The report is written to `reports/24_final_portfolio_case_study.md`. It preserves failed and
incomplete gates instead of converting course completion into a false production-readiness claim.

On a Docker-capable NVIDIA host, build and measure the runtime image:

```bash
./24_final_portfolio_case_study/measure_images.sh
docker run --rm --gpus all -v "$PWD/outputs:/outputs" learn-tensorrt-runtime
```

The runtime image intentionally excludes source code, compilers, CMake, Python tooling, profiling
captures, and calibration data. The TensorRT engine is still environment-specific and must be built
for the deployment GPU/software combination.
