FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py logger.py ./

# Create logs directory
RUN mkdir -p /app/logs

# Don't set environment variables here - they come from docker-compose
# which reads from .env file

# Expose proxy port (will be overridden by docker-compose if needed)
EXPOSE 4001

# Run the application - port will be set via environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PROXY_PORT:-4001}"]