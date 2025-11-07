FROM python:3.10-bookworm

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        wget \
        curl \
        git \
        gnupg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""
ENV YOLO_MODEL_PATH="/app/yolo/weight/best.pt"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

