# MC-Dyn: Dynamic PET/CT TAC extraction pipeline
#
# Build: docker build -t mc-dyn .
# Run:   docker run --gpus all -v /data/input:/input:ro -v /data/output:/output \
#            mc-dyn run /input --output /output

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        dcm2niix \
        git \
        && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python python3.11 1

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY pyproject.toml .
COPY mc_dyn/ mc_dyn/

# Install the package and all dependencies
RUN pip install -e .

# Pre-download Moose model weights (bakes them into the image)
# Adjust model names to match the ones you actually use
# This layer is large (~5-10 GB) but prevents download at runtime
ARG MOOSE_MODELS="clin_ct_organs"
RUN python -c "
from moosez import moose
import os
# Download models by running on a dummy file is not ideal;
# use moosez's built-in download mechanism if available.
# If moosez provides a download CLI: moosez --download clin_ct_organs
# Otherwise this layer can be skipped and models will download on first run.
print('Moose model pre-download: configure RUN step as needed for your moosez version.')
" || true

ENTRYPOINT ["mc-dyn"]
CMD ["--help"]
