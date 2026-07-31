ARG DEVELOPMENT_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
ARG RUNTIME_IMAGE=nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04

FROM ${DEVELOPMENT_IMAGE} AS builder
WORKDIR /src
COPY 11_yolov8_trt_cpp/ 11_yolov8_trt_cpp/
RUN cmake -S 11_yolov8_trt_cpp -B /tmp/build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/build --target yolov8_trt_cpp -j"$(nproc)"

FROM ${RUNTIME_IMAGE} AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopencv-core406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-imgproc406t64 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /tmp/build/yolov8_trt_cpp /app/yolov8_trt_cpp
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.14.1 /usr/lib/x86_64-linux-gnu/
COPY 32_final_portfolio_case_study/outputs/yolov8n_static_autocast_fp16.engine /app/model.engine
COPY assets/img.jpeg /app/input.jpeg
RUN ln -s libnvinfer.so.10.14.1 /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 && \
    ldconfig && \
    mkdir -p /outputs

ENTRYPOINT ["/app/yolov8_trt_cpp", "--engine", "/app/model.engine", "--image", "/app/input.jpeg", "--output-dir", "/outputs"]
