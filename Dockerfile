# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /app

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Download dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Create start.sh script if it doesn't exist in the project
RUN if [ ! -f start.sh ]; then \
    echo '#!/bin/bash\n\n# Start Ollama in the background\nollama serve &\n\n# Wait for Ollama to start\necho "Waiting for Ollama to start..."\nuntil curl -s http://localhost:11434/api/version > /dev/null; do\n  sleep 1\ndone\n\n# Start Streamlit\necho "Starting Streamlit application..."\nstreamlit run app.py --server.port=8501 --server.address=0.0.0.0' > start.sh; \
    chmod +x start.sh; \
    fi

# Expose ports for Streamlit and Ollama
EXPOSE 8501 11434

# Run the application
CMD ["./start.sh"]