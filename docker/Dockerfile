ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG BUILD_TYPE=all
ARG BRANCH_TYPE=remote
ARG GRACE_BLACKWELL=0
ARG HOPPER_SBO=0

ARG GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
ARG HOPPER_SBO_DEEPEP_COMMIT=9f2fc4b3182a51044ae7ecb6610f7c9c3258c4d6
ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG BUILD_AND_DOWNLOAD_PARALLEL=8
ARG SGL_KERNEL_VERSION=0.3.19
ARG SGL_VERSION=0.5.6.post2
ARG USE_LATEST_SGLANG=0
ARG GDRCOPY_VERSION=2.5.1
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG INSTALL_FLASHINFER_JIT_CACHE=0
ARG FLASHINFER_VERSION=0.5.3

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Python setup (combined with apt update to reduce layers)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt update && apt install -y --no-install-recommends \
    wget \
    python3.12-full \
    python3.12-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    # Allow pip to install packages globally (PEP 668 workaround for Ubuntu 24.04)
    && python3 -m pip config set global.break-system-packages true

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    tzdata \
    ca-certificates \
    software-properties-common \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials (needed for framework stage)
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global cache if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## Local Source Stage #########################
########################################################

FROM scratch AS local_src
COPY . /src

########################################################
########## PARALLEL BUILDER STAGES ####################
########################################################
#
# These stages run IN PARALLEL via BuildKit:
#
#   base
#     |
#     +-- rust_builder -----> gateway_builder (needs Rust + source)
#     |
#     +-- torch_deps ------> deepep_builder (needs torch)
#     |                  \-> flashinfer_cache (needs flashinfer)
#     |
#     +-- devtools_builder (independent)
#     |
#     v
#   framework (combines all artifacts)
#

########################################################
# PARALLEL STAGE 1: Rust Builder (starts from base)
########################################################
FROM base AS rust_builder

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin for Python wheel building
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install maturin

########################################################
# PARALLEL STAGE 2: Torch/Deps Builder (starts from base)
########################################################
FROM base AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE
ARG SGL_KERNEL_VERSION
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace

# Install sgl-kernel (from pre-built wheel)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel html5lib six \
    && case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && if [ "$CUDA_VERSION" = "12.6.1" ]; then \
        python3 -m pip install https://${GITHUB_ARTIFACTORY}/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    elif [ "$CUDA_VERSION" = "12.8.1" ] || [ "$CUDA_VERSION" = "12.9.1" ]; then \
        python3 -m pip install sgl-kernel==${SGL_KERNEL_VERSION} \
    ; \
    elif [ "$CUDA_VERSION" = "13.0.1" ]; then \
        python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps \
    ; \
    else \
        echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
    ; \
    fi

# Copy ONLY dependency specification file (for better caching)
COPY --from=local_src /src/python/pyproject.toml /tmp/sglang_deps/python/pyproject.toml

# Install sglang dependencies (torch, transformers, etc.)
# This layer is cached unless pyproject.toml changes
# Generate constraints.txt to prevent reinstalling these deps in later stages
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && echo '__version__ = "0.0.0"' > sglang/version.py \
    && touch README.md \
    && touch LICENSE \
    && python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} . \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && pip freeze | grep -v "^sglang==" > /sgl-workspace/constraints.txt

########################################################
# PARALLEL STAGE 3: DeepEP Builder (needs torch_deps)
########################################################
FROM torch_deps AS deepep_builder

ARG CUDA_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL
ARG GRACE_BLACKWELL
ARG GRACE_BLACKWELL_DEEPEP_BRANCH
ARG HOPPER_SBO
ARG HOPPER_SBO_DEEPEP_COMMIT
ARG DEEPEP_COMMIT
ARG GITHUB_ARTIFACTORY

WORKDIR /build

# Clone DeepEP
RUN set -eux; \
    if [ "$GRACE_BLACKWELL" = "1" ]; then \
      git clone https://github.com/fzyzcjy/DeepEP.git && \
      cd DeepEP && \
      git checkout ${GRACE_BLACKWELL_DEEPEP_BRANCH} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    elif [ "$HOPPER_SBO" = "1" ]; then \
      git clone https://github.com/deepseek-ai/DeepEP.git -b antgroup-opt && \
      cd DeepEP && \
      git checkout ${HOPPER_SBO_DEEPEP_COMMIT} && \
      sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    else \
        curl --retry 3 --retry-delay 2 -fsSL -o ${DEEPEP_COMMIT}.zip \
            https://${GITHUB_ARTIFACTORY}/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.zip && \
        unzip -q ${DEEPEP_COMMIT}.zip && rm ${DEEPEP_COMMIT}.zip && mv DeepEP-${DEEPEP_COMMIT} DeepEP && \
        cd DeepEP && \
        sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh; \
    fi

# Build DeepEP wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /build/DeepEP && \
    case "$CUDA_VERSION" in \
        12.6.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' ;; \
        12.8.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0' ;; \
        12.9.1|13.0.1) CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" MAX_JOBS=${BUILD_AND_DOWNLOAD_PARALLEL} \
        python3 setup.py bdist_wheel -d /wheels

########################################################
# PARALLEL STAGE 4: FlashInfer Cache (needs torch_deps)
########################################################
FROM torch_deps AS flashinfer_cache

ARG CUDA_VERSION
ARG INSTALL_FLASHINFER_JIT_CACHE
ARG FLASHINFER_VERSION
ARG BUILD_AND_DOWNLOAD_PARALLEL

# Download flashinfer cubins
RUN --mount=type=cache,target=/root/.cache/pip \
    case "$CUDA_VERSION" in \
        12.6.1) CUINDEX=126 ;; \
        12.8.1) CUINDEX=128 ;; \
        12.9.1) CUINDEX=129 ;; \
        13.0.1) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
        python3 -m pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX} ; \
    fi \
    && FLASHINFER_CUBIN_DOWNLOAD_THREADS=${BUILD_AND_DOWNLOAD_PARALLEL} FLASHINFER_LOGGING_LEVEL=warning python3 -m flashinfer --download-cubin

########################################################
# PARALLEL STAGE 5: Gateway Builder (needs rust_builder)
########################################################
FROM rust_builder AS gateway_builder

ARG BRANCH_TYPE
ARG SGL_VERSION
ARG USE_LATEST_SGLANG

WORKDIR /build

# Copy only sgl-model-gateway source (not all of sglang)
COPY --from=local_src /src/sgl-model-gateway /build/sgl-model-gateway

# Build sgl-model-gateway wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 \
    && maturin build --release --features vendored-openssl --out /wheels

# Build sgl-model-gateway binary
RUN cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /usr/local/bin/sglang-router

########################################################
# PARALLEL STAGE 6: Dev Tools Builder (starts from base)
########################################################
FROM base AS devtools_builder

ARG GITHUB_ARTIFACTORY

WORKDIR /tools

# Install development apt packages
RUN --mount=type=cache,target=/var/cache/apt,id=devtools-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    openssh-server \
    gnuplot \
    infiniband-diags \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install NVIDIA development tools
RUN --mount=type=cache,target=/var/cache/apt,id=devtools-apt \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# Download CLI tools
RUN curl --retry 3 --retry-delay 2 -LSso /tools/diff-so-fancy \
        https://${GITHUB_ARTIFACTORY}/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy \
    && chmod +x /tools/diff-so-fancy

RUN curl --retry 3 --retry-delay 2 -LSso /tools/clang-format \
        https://${GITHUB_ARTIFACTORY}/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 \
    && chmod +x /tools/clang-format

RUN curl --retry 3 --retry-delay 2 -fsSL -o /tmp/clangd.zip \
        https://${GITHUB_ARTIFACTORY}/clangd/clangd/releases/download/18.1.3/clangd-linux-18.1.3.zip \
    && unzip -q /tmp/clangd.zip -d /tmp \
    && cp /tmp/clangd_18.1.3/bin/* /tools/ \
    && mkdir -p /tools/lib && cp -r /tmp/clangd_18.1.3/lib/* /tools/lib/ \
    && rm -rf /tmp/clangd.zip /tmp/clangd_18.1.3

RUN CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "/tmp/${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "/tmp/${CMAKE_INSTALLER}.tar.gz" -C /tmp \
    && cp -r "/tmp/${CMAKE_INSTALLER}/bin/"* /tools/ \
    && mkdir -p /tools/share && cp -r "/tmp/${CMAKE_INSTALLER}/share/"* /tools/share/ \
    && rm -rf "/tmp/${CMAKE_INSTALLER}" "/tmp/${CMAKE_INSTALLER}.tar.gz"

RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://just.systems/install.sh | \
    sed "s|https://github.com|https://${GITHUB_ARTIFACTORY}|g" | \
    bash -s -- --tag 1.42.4 --to /tools

# Install oh-my-zsh
RUN sh -c "$(curl --retry 3 --retry-delay 2 -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git clone --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-/root/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

########################################################
########## Final Framework Image ######################
########################################################
#
# Combines all artifacts from parallel builder stages
#
FROM torch_deps AS framework

ARG BRANCH_TYPE
ARG BUILD_TYPE
ARG CUDA_VERSION
ARG SGL_VERSION
ARG USE_LATEST_SGLANG
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace

# =============================================================================
# Copy artifacts from parallel builders
# =============================================================================

# Copy DeepEP wheel and install
COPY --from=deepep_builder /wheels /tmp/wheels/deepep
COPY --from=deepep_builder /build/DeepEP /sgl-workspace/DeepEP
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/wheels/deepep/*.whl && rm -rf /tmp/wheels/deepep

# Copy flashinfer cache
COPY --from=flashinfer_cache /root/.cache/flashinfer /root/.cache/flashinfer

# Copy gateway wheel and binary, install
COPY --from=gateway_builder /wheels /tmp/wheels/gateway
COPY --from=gateway_builder /usr/local/bin/sglang-router /usr/local/bin/sglang-router
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/wheels/gateway/*.whl && rm -rf /tmp/wheels/gateway

# Copy dev tools
COPY --from=devtools_builder /tools/diff-so-fancy /usr/local/bin/
COPY --from=devtools_builder /tools/clang-format /usr/local/bin/
COPY --from=devtools_builder /tools/clangd /usr/local/bin/
COPY --from=devtools_builder /tools/lib /usr/local/lib/
COPY --from=devtools_builder /tools/cmake /usr/local/bin/
COPY --from=devtools_builder /tools/ctest /usr/local/bin/
COPY --from=devtools_builder /tools/cpack /usr/local/bin/
COPY --from=devtools_builder /tools/share/cmake-3.31 /usr/local/share/cmake-3.31
COPY --from=devtools_builder /tools/just /usr/local/bin/
COPY --from=devtools_builder /root/.oh-my-zsh /root/.oh-my-zsh

# Install dev apt packages (need to re-run since we're in a different stage)
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    gdb \
    ninja-build \
    vim \
    tmux \
    htop \
    zsh \
    tree \
    silversearcher-ag \
    cloc \
    pkg-config \
    bear \
    less \
    rdma-core \
    openssh-server \
    gnuplot \
    infiniband-diags \
    perftest \
    ibverbs-providers \
    libibumad3 \
    libibverbs1 \
    libnl-3-200 \
    libnl-route-3-200 \
    librdmacm1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install NVIDIA development tools
RUN --mount=type=cache,target=/var/cache/apt,id=framework-apt \
    apt update -y \
    && apt install -y --no-install-recommends gnupg \
    && echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list \
    && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub \
    && apt update -y \
    && apt install -y --no-install-recommends nsight-systems-cli \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Final setup (before source copy for better caching)
# =============================================================================

# Install essential Python packages (use constraints to prevent conflicts)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
    datamodel_code_generator \
    mooncake-transfer-engine==0.3.7.post2 \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    uv \
    wheel \
    scikit-build-core \
    nixl \
    py-spy \
    cubloaty \
    pandas \
    matplotlib \
    tabulate \
    termplotlib

# Patching packages for CUDA 12/13 compatibility
RUN --mount=type=cache,target=/root/.cache/pip if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
    python3 -m pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps ; \
    python3 -m pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps ; \
    python3 -m pip install nvidia-cutlass-dsl==4.3.0 --force-reinstall --no-deps ; \
elif [ "${CUDA_VERSION%%.*}" = "13" ]; then \
    python3 -m pip install nvidia-nccl-cu13==2.28.3 --force-reinstall --no-deps ; \
    python3 -m pip install nvidia-cublas==13.1.0.3 --force-reinstall --no-deps ; \
    python3 -m pip install nixl-cu13 ; \
    python3 -m pip install nvidia-cutlass-dsl==4.3.1 --force-reinstall --no-deps ; \
fi

# Add yank script
COPY --chown=root:root --chmod=755 docker/configs/yank /usr/local/bin/yank

# Configure development environment
COPY docker/configs/.vimrc /root/.vimrc
COPY docker/configs/.tmux.conf /root/.tmux.conf
COPY docker/configs/.gitconfig /tmp/.gitconfig
RUN cat /tmp/.gitconfig >> /root/.gitconfig && rm /tmp/.gitconfig
COPY docker/configs/.zshrc /root/.zshrc

# Fix Triton to use system ptxas for Blackwell (sm_103a) support (CUDA 13+ only)
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

# =============================================================================
# Copy sglang source and do editable install (LAST for better caching)
# =============================================================================

COPY --from=local_src /src /tmp/local_src
RUN if [ "$BRANCH_TYPE" = "local" ]; then \
        cp -r /tmp/local_src /sgl-workspace/sglang; \
    elif [ "$USE_LATEST_SGLANG" = "1" ]; then \
        git clone --depth=1 https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    else \
        git clone --depth=1 --branch v${SGL_VERSION} https://github.com/sgl-project/sglang.git /sgl-workspace/sglang; \
    fi \
    && rm -rf /tmp/local_src

# Editable install (fast - dependencies already installed)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /sgl-workspace/sglang \
    && python3 -m pip install --no-deps -e "python"

# Set workspace directory
WORKDIR /sgl-workspace/sglang

########################################################
########## Runtime Image ##############################
########################################################
#
# PURPOSE: Production runtime environment with JIT support
#
# This stage creates a production-ready image containing:
# - Pre-compiled SGLang and DeepEP components
# - Full CUDA toolchain for JIT compilation (DeepGEMM, Triton, FlashInfer)
# - Optimized for inference workloads and deployment
# - Smaller than framework (no dev tools like vim, tmux, nsight, etc.)
#
# Use this stage when you need:
# - Production deployment of SGLang
# - JIT compilation support for FP8/microscaling kernels
# - Ready-to-run inference server environment
#
# Note: Uses devel base for complete NVCC toolchain required by DeepGEMM JIT
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/

# Add GKE default lib and bin locations + CUDA compiler paths for FlashInfer JIT
ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Install runtime dependencies (devel base provides gcc/g++/build tools)
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    # Python runtime
    python3.12-full \
    python3.12-dev \
    wget \
    # Core system utilities
    tzdata \
    ca-certificates \
    netcat-openbsd \
    curl \
    git \
    # Runtime libraries
    libopenmpi3 \
    libnuma1 \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl3 \
    # RDMA runtime
    rdma-core \
    infiniband-diags \
    perftest \
    # Build tools for JIT compilation
    ninja-build \
    # NCCL packages needed for pynccl_allocator JIT compilation (-lnccl)
    libnccl2 \
    libnccl-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py --break-system-packages \
    && rm get-pip.py \
    # Allow pip to install packages globally (PEP 668 workaround for Ubuntu 24.04)
    && python3 -m pip config set global.break-system-packages true \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up locale
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Copy Python site-packages from framework (contains all built packages)
COPY --from=framework /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Copy SGLang workspace
COPY --from=framework /sgl-workspace /sgl-workspace

# Fix Triton to use system ptxas for Blackwell (sm_103a) support (CUDA 13+ only)
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ -d /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin ]; then \
        rm -f /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas && \
        ln -s /usr/local/cuda/bin/ptxas /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas; \
    fi

# Copy GDRCopy runtime libraries (but not the build artifacts)
COPY --from=framework /usr/lib/libgdrapi.so* /usr/lib/
COPY --from=framework /usr/bin/gdrcopy_* /usr/bin/
COPY --from=framework /usr/src/gdrdrv-2.5.1 /usr/src/gdrdrv-2.5.1

# Fix DeepEP IBGDA symlink in runtime
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

WORKDIR /sgl-workspace/sglang

# Default command
CMD ["/bin/bash"]
