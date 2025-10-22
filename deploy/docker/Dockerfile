FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev libpng-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /workspace

ENV PYTHONPATH=/workspace:${PYTHONPATH}

CMD ["deepspeed", "OmniMoE/training/train.py", "--help"]
