ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG LTF_ASSET_FILE=ltf_sim_navtest.ckpt

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
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
    LTF_CHECKPOINT_PATH=/app/assets/ltf/ltf_sim_navtest.ckpt \
    LTF_DEVICE=cuda \
    LTF_MAX_BATCH_SIZE=2 \
    LTF_BATCH_WINDOW_MS=2

WORKDIR /app

COPY e2e_challenge/sample_submission_simscale_navsim_transfuser/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY src/grpc /tmp/alpasim_grpc
RUN pip install /tmp/alpasim_grpc \
    && pip check \
    && python -c "import alpasim_grpc, torch, torchvision; assert alpasim_grpc.__version__ == (0, 54, 0); assert torch.__version__.startswith('2.6.0'); assert torchvision.__version__.startswith('0.21.0')" \
    && rm -rf /tmp/alpasim_grpc

COPY e2e_challenge/sample_submission_simscale_navsim_transfuser/assets/ltf/${LTF_ASSET_FILE} /app/assets/ltf/ltf_sim_navtest.ckpt
COPY e2e_challenge/sample_submission_simscale_navsim_transfuser/navsim_transfuser_challenge /app/navsim_transfuser_challenge
RUN groupadd --gid 10001 alpasim \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp/home \
        --shell /usr/sbin/nologin alpasim \
    && chmod -R a+rX /app

EXPOSE 6789
STOPSIGNAL SIGTERM
USER 10001:10001
CMD ["python", "-m", "navsim_transfuser_challenge.driver"]
