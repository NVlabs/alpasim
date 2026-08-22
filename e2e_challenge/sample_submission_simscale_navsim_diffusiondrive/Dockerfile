ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE} AS grpc_builder

ARG DIFFUSIONDRIVE_SOURCE_DIR

USER root
WORKDIR /tmp/alpasim_grpc

COPY src/grpc /tmp/alpasim_grpc
COPY ${DIFFUSIONDRIVE_SOURCE_DIR}/navsim_diffusiondrive_challenge/alpasim_grpc_build_only.toml /tmp/alpasim_grpc/pyproject.toml
RUN pip install --no-cache-dir --disable-pip-version-check --no-deps \
      grpcio==1.74.0 \
      grpcio-tools==1.62.2 \
      protobuf==4.25.8 \
      setuptools==80.9.0 \
    && python -c "from scripts.compile_protos import compile_protos; compile_protos()"

FROM ${BASE_IMAGE}

ARG DIFFUSIONDRIVE_SOURCE_DIR
ARG DIFFUSIONDRIVE_ASSET_FILE=diffusiondrive_sim_navhard.ckpt

USER root
WORKDIR /app

ENV PATH=/opt/conda/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp/home \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    TORCH_HOME=/tmp/torch \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_CACHE_PATH=/tmp/nv \
    NUMBA_CACHE_DIR=/tmp/numba \
    ALPASIM_DRIVER_LOG_DIR=/run/alpasim-driver \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_NUM_THREADS=1 \
    TORCH_NUM_INTEROP_THREADS=1 \
    ALPASIM_DRIVER_HOST=0.0.0.0 \
    ALPASIM_DRIVER_PORT=6789 \
    ALPASIM_DRIVER_GRPC_WORKERS=4 \
    DIFFUSIONDRIVE_CHECKPOINT_PATH=/app/assets/diffusiondrive/diffusiondrive_sim_navhard.ckpt \
    DIFFUSIONDRIVE_DEVICE=cuda \
    DIFFUSIONDRIVE_MAX_BATCH_SIZE=1 \
    DIFFUSIONDRIVE_BATCH_WINDOW_MS=2

COPY ${DIFFUSIONDRIVE_SOURCE_DIR}/requirements.txt /tmp/requirements.txt
COPY --from=grpc_builder /tmp/alpasim_grpc /tmp/alpasim_grpc
RUN pip install -r /tmp/requirements.txt \
    && pip install --no-deps --no-build-isolation /tmp/alpasim_grpc \
    && pip check \
    && python -c "import alpasim_grpc, cv2, dataclasses_json, grpc, numpy, PIL, timm, torch, torchvision; assert alpasim_grpc.__version__ == (0, 54, 0); assert torch.__version__.startswith('2.6.0'); assert torchvision.__version__.startswith('0.21.0')" \
    && rm -rf /tmp/alpasim_grpc /tmp/requirements.txt

COPY ${DIFFUSIONDRIVE_SOURCE_DIR}/assets/diffusiondrive/${DIFFUSIONDRIVE_ASSET_FILE} /app/assets/diffusiondrive/diffusiondrive_sim_navhard.ckpt
COPY ${DIFFUSIONDRIVE_SOURCE_DIR}/navsim_diffusiondrive_challenge /app/navsim_diffusiondrive_challenge
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 alpasim; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp/home --shell /usr/sbin/nologin alpasim; fi \
    && chmod -R a+rX /app

EXPOSE 6789
STOPSIGNAL SIGTERM
USER 10001:10001
CMD ["python", "-m", "navsim_diffusiondrive_challenge.driver"]
