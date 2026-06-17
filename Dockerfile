FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY aetherstream ./aetherstream

CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "3002", "--timeout-keep-alive", "600", "--workers", "1"]
