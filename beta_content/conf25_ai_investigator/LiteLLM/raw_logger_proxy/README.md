# Raw Logger Proxy

A transparent HTTP proxy that logs raw request and response data while forwarding traffic to LiteLLM.

## Features

- **Transparent Proxy**: Forwards all HTTP methods and headers without modification
- **Raw Logging**: Captures complete request and response data
- **Streaming Support**: Handles SSE/streaming responses properly
- **Configurable**: Environment variables for all settings
- **Organized Logs**: Separate directories for requests and responses

## Configuration

Environment variables:
- `LITELLM_HOST`: Target LiteLLM host (default: localhost)
- `LITELLM_PORT`: Target LiteLLM port (default: 4000)
- `PROXY_PORT`: Proxy listen port (default: 4001)
- `LOG_DIR`: Directory for log files (default: ./logs)
- `MAX_BODY_LOG_SIZE`: Maximum body size to log in bytes (default: 1048576)
- `LOG_FORMAT`: Log format - "json" or "raw" (default: json)

## Usage

### Standalone

```bash
pip install -r requirements.txt
python main.py
```

### Docker

```bash
docker build -t raw-logger-proxy .
docker run -p 4001:4001 -v $(pwd)/logs:/app/logs raw-logger-proxy
```

### Docker Compose

```bash
sudo docker compose up -d
```

## API Usage

Simply point your API clients to `http://localhost:4001` instead of `http://localhost:4000`. All requests will be transparently forwarded to LiteLLM while being logged.

## Log Structure

```
logs/
├── requests/
│   ├── <request_id>_request.json
│   └── <request_id>_request_body.raw (if LOG_FORMAT=raw)
└── responses/
    ├── <request_id>_response.json
    └── <request_id>_response_body.raw (if LOG_FORMAT=raw)
```

## Health Check

```bash
curl http://localhost:4001/health
```