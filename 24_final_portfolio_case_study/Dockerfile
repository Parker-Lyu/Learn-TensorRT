ARG TRT_IMAGE=nvcr.io/nvidia/tensorrt:23.10-py3
FROM ${TRT_IMAGE} AS builder
WORKDIR /src
COPY 10_yolov8_trt_cpp/ 10_yolov8_trt_cpp/
RUN cmake -S 10_yolov8_trt_cpp -B /build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /build -j

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/yolov8_trt_cpp /app/yolov8_trt_cpp
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnvinfer.so.8* /usr/lib/x86_64-linux-gnu/
COPY 06_trtexec_engine/outputs/yolov8n_static_fp16.engine /app/model.engine
COPY assets/img2.jpeg /app/input.jpeg
RUN mkdir -p /outputs
ENTRYPOINT ["/app/yolov8_trt_cpp", "--engine", "/app/model.engine", "--image", "/app/input.jpeg", "--output-dir", "/outputs"]
