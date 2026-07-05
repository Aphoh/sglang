# syntax=docker/dockerfile:1.7
FROM lmsysorg/sglang:v0.5.14-runtime

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG CRIU_REPOSITORY=https://github.com/galletas1712/criu.git
ARG CRIU_REF=00b4a49d17a0bb7d9f69c0164148c0c46f0ca928
ARG CUDA_CHECKPOINT_REF=1c71b048c779d30921f01172dda96e04033bb8ac

RUN apt-get update && apt-get install -y --no-install-recommends \
    asciidoctor build-essential ca-certificates git-core iproute2 iptables jq libaio-dev \
    libbsd-dev libcap-dev libdrm-dev libelf-dev libgnutls28-dev libnet1-dev \
    libnl-3-dev libnl-route-3-dev libnftables-dev libprotobuf-c-dev \
    libprotobuf-dev libselinux-dev libtraceevent-dev libtracefs-dev \
    pkg-config protobuf-c-compiler protobuf-compiler python3-dev \
    python3-protobuf python3-yaml uuid-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init /tmp/criu \
    && git -C /tmp/criu remote add origin "${CRIU_REPOSITORY}" \
    && git -C /tmp/criu fetch --depth 1 origin "${CRIU_REF}" \
    && git -C /tmp/criu checkout FETCH_HEAD \
    && make -C /tmp/criu -j"$(nproc)" \
    && make -C /tmp/criu install PREFIX=/usr/local \
    && mkdir -p /var/empty/criu \
    && rm -rf /tmp/criu

RUN git init /tmp/cuda-checkpoint \
    && git -C /tmp/cuda-checkpoint remote add origin https://github.com/NVIDIA/cuda-checkpoint.git \
    && git -C /tmp/cuda-checkpoint fetch --depth 1 origin "${CUDA_CHECKPOINT_REF}" \
    && git -C /tmp/cuda-checkpoint checkout FETCH_HEAD \
    && install -m 0755 /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint \
      /usr/local/sbin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --force-reinstall cmake distro uv
COPY sgl-kernel /tmp/sgl-kernel
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/tmp/sgl-kernel-build \
    uv build --wheel --out-dir /tmp/wheels \
      -Cbuild-dir=/tmp/sgl-kernel-build \
      -Ccmake.define.ENABLE_BELOW_SM90=OFF \
      -Ccmake.define.SGL_KERNEL_ENABLE_FA3=OFF \
      /tmp/sgl-kernel --no-build-isolation \
    && python3 -m pip install --force-reinstall --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/sgl-kernel /tmp/wheels

COPY examples/experimental/criu/nvidia_criu_plugin.c /tmp/nvidia_criu_plugin.c
RUN cc -O2 -Wall -Wextra -Werror -shared -fPIC \
      /tmp/nvidia_criu_plugin.c \
      -o /usr/local/lib/criu/nvidia_plugin.so \
    && rm /tmp/nvidia_criu_plugin.c

COPY python /opt/sglang/python
COPY examples/experimental/criu /opt/sglang/examples/experimental/criu
ENV PYTHONPATH=/opt/sglang/python \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    UV_USE_IO_URING=0 \
    USE_LIBUV=0
WORKDIR /opt/sglang
