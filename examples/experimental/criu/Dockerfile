FROM nvcr.io/nvidia/ai-dynamo/sglang-runtime@sha256:ca083d53a569cde5d97e451b876f600b5c96a44c8bea82b05717a9c03234015b

USER root

ARG DEBIAN_FRONTEND=noninteractive
ARG CRIU_REPOSITORY=https://github.com/galletas1712/criu.git
ARG CRIU_REF=00b4a49d17a0bb7d9f69c0164148c0c46f0ca928
ARG CUDA_CHECKPOINT_REF=1c71b048c779d30921f01172dda96e04033bb8ac
ARG DYNAMO_REF=eab08859f1f48f3ff7dceab57c849caf105a275c
ARG FLASHINFER_REPOSITORY=https://github.com/Aphoh/flashinfer.git
ARG FLASHINFER_REF=3709f4ce776e954891384220c028bb31bff1d217
ARG RUST_TOOLCHAIN=1.90.0

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

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" \
    && ln -s /root/.cargo/bin/cargo /usr/local/bin/cargo \
    && ln -s /root/.cargo/bin/rustc /usr/local/bin/rustc \
    && cargo --version \
    && rustc --version

RUN curl -LsSf https://astral.sh/uv/0.11.16/install.sh | \
      env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv pip install --system --break-system-packages --no-deps \
      'flashinfer_cubin==0.6.13' \
      'nixl-cu13==1.1.0' \
      'sglang-kernel==0.4.3' \
    && git init /tmp/flashinfer \
    && git -C /tmp/flashinfer remote add origin "${FLASHINFER_REPOSITORY}" \
    && git -C /tmp/flashinfer fetch --depth 1 origin "${FLASHINFER_REF}" \
    && git -C /tmp/flashinfer checkout FETCH_HEAD \
    && git -C /tmp/flashinfer submodule update --init --recursive \
    && uv pip install --system --break-system-packages --no-deps \
      /tmp/flashinfer \
    && rm -rf /tmp/flashinfer \
    && uv pip uninstall --system --break-system-packages flashinfer-jit-cache \
    && criu --version \
    && uv --version \
    && python -c 'import flashinfer, importlib.metadata as m, torch; assert m.version("torch").startswith("2.11.0"); assert torch.version.cuda == "13.0"; assert m.version("flashinfer-python") == "0.6.13"; assert flashinfer.__git_version__ == "3709f4ce776e954891384220c028bb31bff1d217"; assert m.version("flashinfer-cubin") == "0.6.13"; assert m.version("nixl-cu13") == "1.1.0"; assert m.version("sglang-kernel") == "0.4.3"'

WORKDIR /workspace/sglang
