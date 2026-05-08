FROM python:3.14.4-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

WORKDIR /app

# Environment setup
RUN apt update && apt install -y git build-essential wget ffmpeg
RUN apt clean
RUN pip install --upgrade pip
RUN pip install --no-cache-dir git+https://github.com/wenet-e2e/wespeaker.git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model download
WORKDIR /app/models
RUN wget -O voxceleb_resnet293_LM.onnx \
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/resolve/main/voxceleb_resnet293_LM.onnx?download=true"
WORKDIR /app

# Copy the code
COPY src/ /app/src/

CMD ["python", "-m", "src.api"]