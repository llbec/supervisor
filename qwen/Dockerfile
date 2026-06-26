FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/qwen

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /srv/qwen/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /srv/qwen/requirements.txt

COPY service /srv/qwen/service

EXPOSE 9001

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "9001"]
