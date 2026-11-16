FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cie ./cie
RUN pip install --no-cache-dir . redis qdrant-client

ENV CIE_DATA_DIR=/data
VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "cie.api:app", "--host", "0.0.0.0", "--port", "8000"]
