# Build command (run from repo root):
#   docker build -t alpasim-base .
#
# For private dependencies (requires ~/.netrc with credentials):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base .
#
# Automatically detects architecture:
#   x86_64  -> nvidia/cuda base
#   aarch64 -> NGC PyTorch base (only CUDA-enabled PyTorch source on ARM)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-amd64
FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base-arm64
FROM nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.6.0-ubuntu22.04@sha256:b7a4241c608253aa829041cc3575ea57082491251a4a626bcdddc68eaf9a3101 AS dcgm-exporter
ARG TARGETARCH
FROM base-${TARGETARCH}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=dcgm-exporter /usr/bin/dcgm-exporter /usr/bin/dcgm-exporter
COPY --from=dcgm-exporter /etc/dcgm-exporter /etc/dcgm-exporter
RUN printf '%s\n' \
    'DCGM_FI_DEV_FB_TOTAL, gauge, Total framebuffer memory (in MiB).' \
    >> /etc/dcgm-exporter/default-counters.csv

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
    libgl1 \
    prometheus \
    prometheus-node-exporter \
    prometheus-process-exporter \
    datacenter-gpu-manager-4-cuda12 \
    && dcgm_lib="$(find /usr/lib /lib -name libdcgm.so.4 -print -quit)" \
    && ln -sf "$dcgm_lib" "$(dirname "$dcgm_lib")/libdcgm.so" \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain (required for utils_rs maturin build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /repo

# Configure uv
ENV UV_LINK_MODE=copy

# Compile protos
WORKDIR /repo/src/grpc
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync'
RUN uv run compile-protos --no-sync

WORKDIR /repo

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    sh -c 'if [ -f /root/.netrc ]; then export NETRC=/root/.netrc; fi && uv sync --extra all'

ARG PYTORCH_VERSION=2.8.0+cu128
ARG TORCH_CLUSTER_VERSION=1.6.3
ARG TORCH_SCATTER_VERSION=2.1.2
ARG TORCH_SPARSE_VERSION=0.6.18

# Install PyG compiled extensions (torch-cluster, torch-scatter, torch-sparse)
# from pre-built wheels matching the installed torch + CUDA versions.
RUN PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}.html" && \
    uv pip install \
        "torch-cluster==${TORCH_CLUSTER_VERSION}" \
        "torch-scatter==${TORCH_SCATTER_VERSION}" \
        "torch-sparse==${TORCH_SPARSE_VERSION}" \
        -f "$PYG_WHEEL_URL"

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_NO_SYNC=1
