FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RUNNING_INSIDE_DOCKER=1

WORKDIR /app

# Copy code
COPY . /app

# Install debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --upgrade pip \
    && pip install .

# Start server
CMD ["python", "server.py"]
