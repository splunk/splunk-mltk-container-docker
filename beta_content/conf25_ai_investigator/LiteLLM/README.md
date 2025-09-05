# LiteLLM Integration

This directory contains the LiteLLM proxy service configuration, which is integrated with the main RAG project.

## Overview

LiteLLM provides a unified API for accessing multiple LLM providers (OpenAI, Anthropic, etc.) with:
- Single API endpoint for all LLM providers
- Request/response logging
- Model fallback support
- Cost tracking
- Rate limiting

## Architecture

The LiteLLM setup includes:
- **LiteLLM Proxy**: Main API service (port 4000)
- **PostgreSQL Database**: Stores model configurations and usage data (port 5434)
- **Raw Logger Proxy**: Optional logging proxy for debugging (port 4001)

## Integration with Main Project

This LiteLLM setup is integrated with the main docker-compose.yml using Docker Compose's `include` feature. All services can be managed from the project root using the `./dock` script.

### Start all services (RAG + LiteLLM):
```bash
./dock up
```

### Stop all services:
```bash
./dock down
```

### Check status:
```bash
./dock status
```

### View logs:
```bash
./dock logs litellm
```

## Configuration

### Environment Variables
All LiteLLM configuration is done through the main `.env` file in the project root. Key variables:
- `LITELLM_PORT`: External API port (default: 4000)
- `LITELLM_MASTER_KEY`: Master API key for admin access
- `LITELLM_DB_PORT`: PostgreSQL port (default: 5434)
- `RAW_LOGGING_PORT`: Raw logging proxy port (default: 4001)

### Model Configuration
Edit `litellm_config.yaml` to configure available models and their settings.

## Usage

### Direct API Access (No Logging)
Connect to `http://localhost:4000`

### With Raw Request/Response Logging
Connect to `http://localhost:4001`

### Test the Setup
```bash
./LiteLLM/testrun
```

## Directory Structure
```
LiteLLM/
├── docker-compose-litellm.yml  # Service definitions
├── litellm_config.yaml         # Model configurations
├── raw_logger_proxy/           # Raw logging proxy code
├── SCRIPTS/                    # Helper scripts
└── testrun                     # Test script
```

## Troubleshooting

### Check service health:
```bash
./dock status
```

### View LiteLLM logs:
```bash
./dock logs litellm
```

### Access LiteLLM database:
```bash
sudo docker exec -it armd_finder__litellm-postgres psql -U llmproxy -d litellm
```

### Reset LiteLLM (preserves main RAG data):
```bash
cd LiteLLM
./SCRIPTS/reset0
```