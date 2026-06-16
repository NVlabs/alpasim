FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV XDG_CACHE_HOME=/tmp/.cache
ENV TORCH_HOME=/tmp/torch
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV CUDA_CACHE_PATH=/tmp/nv
ENV NUMBA_CACHE_DIR=/tmp/numba
ENV ALPASIM_DRIVER_LOG_DIR=/run/alpasim-driver
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV TORCH_NUM_THREADS=1
ENV TORCH_NUM_INTEROP_THREADS=1
ENV ALPASIM_DRIVER_GRPC_WORKERS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/grpc /tmp/alpasim_grpc
RUN pip install /tmp/alpasim_grpc && rm -rf /tmp/alpasim_grpc

COPY e2e_challenge/sample_submission_vavam/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

COPY e2e_challenge/sample_submission_vavam/assets/vavam /app/assets/vavam
COPY e2e_challenge/sample_submission_vavam/vavam_challenge /app/vavam_challenge

ENV ALPASIM_DRIVER_HOST=0.0.0.0
ENV ALPASIM_DRIVER_PORT=6789
ENV VAVAM_CHECKPOINT_PATH=/app/assets/vavam/VAM_width_1024_pretrained_139k.pt
ENV VAVAM_TOKENIZER_PATH=/app/assets/vavam/VQ_ds16_16384_llamagen_encoder.jit
EXPOSE 6789

CMD ["python", "-m", "vavam_challenge.driver"]
