#!/bin/bash

# 1. Start Ollama in the background
echo "Starting Ollama server..."
ollama serve &

# 2. Wait for Ollama server to accept connections on port 11434
echo "Waiting for Ollama to wake up..."
until curl -s http://127.0.0.1:11434 > /dev/null; do
    sleep 2
done
echo "Ollama is ready!"

# 3. Pull the configured model only if not already present
MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
if ! ollama list | grep -q "$MODEL"; then
    echo "Pulling $MODEL model (first run only)..."
    ollama pull "$MODEL"
else
    echo "Model $MODEL is already installed."
fi

# 4. Start the FastAPI application using Uvicorn
echo "Starting FastAPI App on port 8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000