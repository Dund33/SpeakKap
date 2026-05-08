FROM python:3.14.4-slim-bookworm

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
COPY requirements.txt .
RUN apt update && apt install -y git
RUN apt clean
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/wenet-e2e/wespeaker.git