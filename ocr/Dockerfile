FROM python:3.10-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
ENV OPENAI_API_KEY=""

CMD ["uvicorn", "main_ocr:app", "--host", "0.0.0.0", "--port", "8001"]
