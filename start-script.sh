#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/version > /dev/null; do
  sleep 1
done

# Pull the required models
echo "Pulling required Ollama models..."
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# Start Streamlit
echo "Starting Streamlit application..."
streamlit run app.py --server.address=0.0.0.0
