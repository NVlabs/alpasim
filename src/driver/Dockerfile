# Example command (run from repo root):
#   docker build --secret id=netrc,src=$HOME/.netrc -t alpasim_driver:latest -f src/driver/Dockerfile .
# Runtime stage with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Install git (needed for VAM repository)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy grpc and utils source code
COPY src/grpc /src/grpc
COPY src/utils /src/utils

# Configure uv
ENV UV_LINK_MODE=copy

# Compile protos
WORKDIR /src/grpc
RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    NETRC=/root/.netrc uv sync
RUN uv run compile-protos --no-sync

# Install driver dependencies and code
COPY src/driver/ /src/driver/
WORKDIR /src/driver

RUN --mount=type=secret,id=netrc,target=/root/.netrc \
    --mount=type=cache,target=/root/.cache/uv \
    NETRC=/root/.netrc uv sync

ENV UV_NO_SYNC=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# Default entrypoint
ENTRYPOINT ["uv", "run", "alpasim_driver"]
