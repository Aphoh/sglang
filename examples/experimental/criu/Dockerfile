FROM nvcr.io/nvidia/ai-dynamo/sglang-runtime@sha256:ca083d53a569cde5d97e451b876f600b5c96a44c8bea82b05717a9c03234015b

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG CRIU_REPOSITORY=https://github.com/galletas1712/criu.git
ARG CRIU_REF=00b4a49d17a0bb7d9f69c0164148c0c46f0ca928
ARG CUDA_CHECKPOINT_REF=1c71b048c779d30921f01172dda96e04033bb8ac
ARG DYNAMO_REF=eab08859f1f48f3ff7dceab57c849caf105a275c

RUN apt-get update && apt-get install -y --no-install-recommends \
    asciidoctor bash bsdmainutils build-essential ca-certificates curl git-core \
    iproute2 iptables jq kmod libaio-dev libbsd-dev libcap-dev libdrm-dev \
    libelf-dev libgnutls28-dev libnet1-dev libnl-3-dev libnl-route-3-dev \
    libnftables-dev libperl-dev libprotobuf-c-dev libprotobuf-dev \
    libselinux-dev libtraceevent-dev libtracefs-dev pkg-config \
    protobuf-c-compiler protobuf-compiler python3-dev python3-protobuf \
    python3-yaml time util-linux uuid-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git init /tmp/criu \
    && cd /tmp/criu \
    && git remote add origin "${CRIU_REPOSITORY}" \
    && git fetch --depth 1 origin "${CRIU_REF}" \
    && git checkout FETCH_HEAD \
    && make -j"$(nproc)" \
    && make install PREFIX=/usr/local \
    && rm -rf /tmp/criu

RUN git init /tmp/cuda-checkpoint \
    && cd /tmp/cuda-checkpoint \
    && git remote add origin https://github.com/NVIDIA/cuda-checkpoint.git \
    && git fetch --depth 1 origin "${CUDA_CHECKPOINT_REF}" \
    && git checkout FETCH_HEAD \
    && install -m 0755 bin/x86_64_Linux/cuda-checkpoint \
      /usr/local/sbin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint \
    && mkdir -p /var/empty/criu

RUN curl -LsSf \
      "https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REF}/deploy/snapshot/cmd/cuda-checkpoint-helper/main.c" \
      -o /tmp/cuda-checkpoint-helper.c \
    && gcc -O2 -Wall -Wextra \
      -I/usr/local/cuda/include \
      -L/usr/local/cuda/lib64/stubs \
      /tmp/cuda-checkpoint-helper.c -lcuda \
      -o /usr/local/bin/cuda-checkpoint-helper \
    && rm /tmp/cuda-checkpoint-helper.c

RUN curl -LsSf https://astral.sh/uv/0.11.16/install.sh | \
      env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv pip install --system --break-system-packages --upgrade \
      'nixl-cu13==1.1.0' \
      'sglang-kernel==0.4.3' \
    && criu --version \
    && uv --version \
    && python -c 'import importlib.metadata as m; print(m.version("sglang-kernel"))'

WORKDIR /workspace/sglang
