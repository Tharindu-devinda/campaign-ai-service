FROM ollama/ollama:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-venv \
       python3-pip \
       curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY converter ./converter
COPY docker/start.sh ./start.sh

# Remove Windows CRLF characters and make executable
RUN sed -i 's/\r$//' ./start.sh && chmod +x ./start.sh

EXPOSE 8000 11434

ENTRYPOINT ["/bin/bash", "/app/start.sh"]