FROM python:3.14.4-slim-bookworm

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

# Environment setup
COPY requirements.txt .
COPY *.py .
RUN apt update && apt install -y git build-essential wget
RUN apt clean
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/wenet-e2e/wespeaker.git

# Model download
WORKDIR /app/models
RUN wget https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/resolve/main/voxceleb_resnet293_LM.onnx?download=true
WORKDIR /app
