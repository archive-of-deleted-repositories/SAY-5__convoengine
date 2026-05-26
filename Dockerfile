FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install .

FROM python:3.12-slim AS runtime
WORKDIR /app
RUN useradd -m -u 10001 convo
COPY --from=builder /install /usr/local/lib/python3.12/site-packages
COPY src /app/src
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
USER convo
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "convoengine.api:app", "--host", "0.0.0.0", "--port", "8000"]
