# AInvestigator

We're building an intelligent system that uncovers early warnings and hidden threats in vast, unfamiliar data - turning complexity into clarity before disaster strikes.
Project leverages natural language queries to unlock critical insights from custom datasets and uncharted knowledge domains.

Our solution is adaptable to any use case or industry, bridging the gap between less technical business users and the complexity of underlying technologies.

Based on real-world public datasets(not synthetic), we will provably demonstrate the system’s ability to detect dangerous fraud and criminal intent early - before harm is done.

While rooted in cybersecurity applications, the system is multi-domain and has already proven effective in detecting deadly forms of healthcare fraud, such as the illegal and excessive prescription of controlled substances and opioids without medical justification.

Our solution will bridge together 2 custom MCPs + general purpose LLMs + Vector DB storage and agentic workflows to surface critical insights from less-familiar datasets and non-standard business domains.


## Overview

### Core Components

1. **AInvestigator SPL Agent** (Port 2993)
   - Main FastAPI application that generates SPL queries from natural language
   - Uses FAISS vector database with GPU acceleration for semantic search
   - Supports multiple LLM providers through direct integration and LiteLLM proxy
   - Provides web interface for query generation and vector database management

2. **MCP (Model Context Protocol) Servers**
   - **Custom Query Agent** (Ports 7006-7007): Converts natural language to SPL via MCP protocol
   - **Splunk MCP Lite** (Ports 7008-7009/8052): Executes SPL queries against Splunk instances
   - Both support SSE and Streamable HTTP transports with Cloudflare tunnel access

3. **LiteLLM Service** (Ports 7010-7011)
   - Unified proxy for multiple LLM providers (OpenAI, Ollama, OpenRouter, Groq, etc.)
   - PostgreSQL database for configuration and logging
   - Raw logger proxy for request/response debugging

## Features

- **Natural Language to SPL**: Convert plain English questions into complex Splunk queries
- **Vector Database**: Store and retrieve similar query examples and data descriptions using semantic search
- **Multiple LLM Support**: Compatible with various LLM providers including OpenAI, Ollama, and more
- **Specialized Tools**: Pre-built tools for common security use cases like anomalous IP detection
- **Web Interface**: User-friendly interface for query generation and management
- **API Compatibility**: Ollama-compatible API endpoints for integration with other tools
- **GPU Acceleration**: FAISS vector database with GPU support for faster similarity search
- **Query Validation**: Comprehensive validation of generated SPL queries
- **MCP Protocol Support**: Multiple transport options (SSE, HTTP streaming, stdio)
- **Docker-based Architecture**: All services containerized for easy deployment

## Complete Setup Instructions

### Prerequisites

- Linux/Unix system (tested on Ubuntu)
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Conda or Miniconda installed
- Git
- At least 16GB RAM recommended

### 1. Clone Repository

```bash
git clone https://github.com/splunk/splunk-mltk-container-docker.git 
cd splunk-mltk-container-docker/beta_content/conf25_ai_investigator
```

### 2. Environment Setup

Initialize and activate the conda environment:

```bash
source ./init
```

This creates/activates the "AInv2" conda environment with Python 3.11, PyTorch, FAISS-GPU, and all dependencies.

### 3. Configure API Keys

Create a configuration file for your API keys:

```bash
mkdir -p config
echo "OPENAI_API_KEY=your-openai-key-here" > config/api_keys.txt
# Add other API keys as needed
```

### 4. Build and Start Docker Services

#### Start All Services

```bash
# Full reset and rebuild (first time setup)
./dock reset0

# Or just reload if containers exist
./dock reload
```

This will:
- Build all Docker images
- Start LiteLLM service with PostgreSQL
- Start both MCP servers (custom-query-agent and splunk-mcp-lite)
- Set up Cloudflare tunnels for external access

#### Check Service Status

```bash
./dock status
```

This shows the health status of all components with colored output.

### 5. Initialize Vector Database

Load the initial vector database content:

```bash
./start
# Or manually:
python app.py loaddata
```

### 6. Access the Application

- **Main Web Interface**: http://localhost:2993
- **API Documentation**: http://localhost:2993/docs
- **Vector DB Management**: http://localhost:2993/manage

## Service Architecture

### Port Mapping

| Service | Port | Description |
|---------|------|-------------|
| AInvestigator Main | 2993 | Main SPL query generation API and web UI |
| Custom Query Agent SSE | 7006 | MCP server SSE transport |
| Custom Query Agent HTTP | 7007 | MCP server HTTP streaming |
| Splunk MCP Lite SSE | 7008 | Splunk execution SSE transport |
| Splunk MCP Lite HTTP | 7009/8052 | Splunk execution HTTP streaming |
| LiteLLM Proxy | 7010 | Unified LLM provider interface |
| LiteLLM Raw Logger | 7011 | Request/response logging proxy |
| PostgreSQL (LiteLLM) | 5435 | LiteLLM configuration database |

### Docker Containers

#### LiteLLM Service
- `ainv__litellm-main`: Main LiteLLM proxy server
- `ainv__litellm-postgres`: PostgreSQL database for LiteLLM
- `ainv__raw_logger`: HTTP proxy for logging LLM requests/responses

#### MCP Custom Query Agent
- `ainv__mcp_custom-query-agent-sse-int`: SSE transport (internal)
- `ainv__mcp_custom-query-agent-stream-http-int`: HTTP streaming (internal)
- `ainv__mcp_custom-query-agent-stream-http-ext`: HTTP streaming (external via tunnel)
- `ainv__mcp_custom-query-agent-cloudflare`: Cloudflare tunnel for external access

#### MCP Splunk Lite
- `ainv__mcp_splunk-lite-sse-int`: SSE transport (internal)
- `ainv__mcp_splunk-lite-stream-http-int`: HTTP streaming (internal)
- `ainv__mcp_splunk-lite-stream-http-ext`: HTTP streaming (external via tunnel)
- `ainv__mcp_splunk-lite-cloudflare`: Cloudflare tunnel for external access

## Docker Management

### Global Commands (from project root)

```bash
./dock status     # Show status of all services
./dock reset0     # Full cleanup and rebuild of all components
./dock reload     # Rebuild and restart all components
./dock stop       # Stop all components
./dock restart    # Restart all components
```

### Individual Service Management

```bash
# LiteLLM service
cd LiteLLM && ./dock status

# Custom Query Agent
cd mcp.custom-query-agent && ./dock status

# Splunk MCP Lite
cd mcp.splunk-mcp-lite && ./dock status
```

## Usage Examples

### Web Interface

1. Navigate to http://localhost:2993
2. Enter a natural language query (e.g., "Show me all failed login attempts in the last hour")
3. Select LLM provider and parameters
4. Click "Generate Query" to get the SPL

### API Usage

#### Generate Query API

```bash
curl -X POST http://localhost:2993/generate_query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find top 10 IP addresses with most traffic",
    "llm_provider": "ollama_3.1_70b",
    "q_samples": 5
  }'
```

#### Ollama-Compatible API

```bash
curl -X POST http://localhost:2993/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b-instruct-q8_0",
    "prompt": "Show me all errors in the last day",
    "stream": false
  }'
```

### Vector Database Management

Add new data source descriptions or query examples via the web interface at http://localhost:2993/manage

## Development

### Running Tests

```bash
# Test Custom Query Agent
cd mcp.custom-query-agent/tests
./testall

# Test Splunk MCP Lite
cd mcp.splunk-mcp-lite/tests
python test_server.py

# Test LiteLLM
cd LiteLLM
./testall
```

### Git Workflow

```bash
# Create backup with automatic commit
./bu

# Create backup with custom message
./bu "Added new feature"

# Create new working branch
./bu w!

# Show recent working branches
./show_working_branches
```

### Project Structure

```
conf25_ai_investigator/
├── app.py                  # Main application entry point
├── api.py                  # FastAPI routes and handlers
├── query_engine.py         # Query generation logic
├── vector_store.py         # FAISS vector database
├── constants.py            # SPL templates and constants
├── models.py              # Pydantic data models
├── VECTOR_DB_CONTENT/     # Vector database content
│   ├── DATASTORES/        # Data source descriptions
│   └── QUERY_SAMPLES/     # Example queries
├── mcp.custom-query-agent/ # MCP server for query generation
├── mcp.splunk-mcp-lite/   # MCP server for Splunk execution
├── LiteLLM/               # LiteLLM proxy service
├── templates/             # Jinja2 HTML templates
├── static/                # JavaScript and CSS files
└── dock                   # Docker management script
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure no other services are using the configured ports
2. **GPU not detected**: Check CUDA installation and GPU drivers
3. **Docker permission errors**: Run Docker commands with `sudo` if needed
4. **API key errors**: Verify API keys in `config/api_keys.txt`

### Checking Logs

```bash
# View logs for specific service
docker logs ainv__litellm-main
docker logs ainv__mcp_custom-query-agent-stream-http-int

# Follow logs in real-time
docker logs -f ainv__litellm-main
```

### Service Health Checks

```bash
# Check all services
./dock status

# Test MCP endpoints (use timeout to avoid blocking)
timeout 5 curl -s http://localhost:7007/sse

# Check LiteLLM
curl http://localhost:7010/health
```

## Security Considerations

- API keys stored in `config/api_keys.txt` (not in version control)
- Environment variables loaded via python-dotenv
- MCP servers include guardrails for Splunk query validation
- External access controlled via Cloudflare tunnels
- No hardcoded credentials in the codebase

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is proprietary software. All rights reserved.

## Contact

Gleb Esman  
gesman@cisco.com