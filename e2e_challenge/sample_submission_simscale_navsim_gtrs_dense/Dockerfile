ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG INSTALL_DEPENDENCIES=1
ARG GTRS_SOURCE_DIR
ARG GTRS_ASSET_FILE
ARG GTRS_CHECKPOINT_NAME
ARG GTRS_BACKBONE=resnet
ARG GTRS_SERVICE_VERSION

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
    GTRS_CHECKPOINT_PATH=/app/assets/gtrs_dense/${GTRS_CHECKPOINT_NAME} \
    GTRS_BACKBONE=${GTRS_BACKBONE} \
    GTRS_VOCAB_PATH=/app/assets/gtrs_dense/navsim_16384.npy \
    GTRS_SERVICE_VERSION=${GTRS_SERVICE_VERSION} \
    GTRS_DEVICE=cuda \
    GTRS_SPEED_ENHANCEMENT=1 \
    GTRS_MAX_BATCH_SIZE=1 \
    GTRS_BATCH_WINDOW_MS=2

COPY ${GTRS_SOURCE_DIR}/requirements.txt /tmp/requirements.txt
COPY src/grpc /tmp/alpasim_grpc
RUN case "${INSTALL_DEPENDENCIES}" in \
      1) pip install -r /tmp/requirements.txt && pip install /tmp/alpasim_grpc ;; \
      0) ;; \
      *) echo "INSTALL_DEPENDENCIES must be 0 or 1" >&2; exit 2 ;; \
    esac \
    && pip check \
    && python -c "import alpasim_grpc, cv2, grpc, timm, torch, torchvision; assert alpasim_grpc.__version__ == (0, 54, 0); assert torch.__version__.startswith('2.6.0'); assert torchvision.__version__.startswith('0.21.0')" \
    && rm -rf /tmp/alpasim_grpc /tmp/requirements.txt

COPY ${GTRS_SOURCE_DIR}/assets/gtrs_dense/${GTRS_ASSET_FILE} /app/assets/gtrs_dense/${GTRS_CHECKPOINT_NAME}
COPY ${GTRS_SOURCE_DIR}/assets/gtrs_dense/navhard_8192.npy /app/assets/gtrs_dense/navhard_8192.npy
COPY ${GTRS_SOURCE_DIR}/assets/gtrs_dense/navsim_16384.npy /app/assets/gtrs_dense/navsim_16384.npy
COPY ${GTRS_SOURCE_DIR}/navsim_gtrs_dense_challenge /app/navsim_gtrs_dense_challenge
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 alpasim; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp/home --shell /usr/sbin/nologin alpasim; fi \
    && chmod -R a+rX /app

EXPOSE 6789
STOPSIGNAL SIGTERM
USER 10001:10001
CMD ["python", "-m", "navsim_gtrs_dense_challenge.driver"]
