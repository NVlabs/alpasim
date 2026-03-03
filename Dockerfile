# Build command (run from repo root):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim-base .
#
# Automatically detects architecture:
#   x86_64  -> nvidia/cuda base
#   aarch64 -> NGC PyTorch base (only CUDA-enabled PyTorch source on ARM)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-amd64
FROM nvcr.io/nvidia/pytorch:25.08-py3 AS base-arm64
ARG TARGETARCH
FROM base-${TARGETARCH}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    curl \
    build-essential \
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
    NETRC=/root/.netrc uv sync
RUN uv run compile-protos --no-sync

WORKDIR /repo

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    NETRC=/root/.netrc uv sync --all-packages

ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_NO_SYNC=1
