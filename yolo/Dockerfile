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

EXPOSE 8000
CMD ["uvicorn", "main_yolo:app", "--host", "0.0.0.0", "--port", "8000"]