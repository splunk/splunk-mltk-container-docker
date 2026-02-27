# Custom Query Agent MCP Server

An MCP (Model Context Protocol) server that converts natural language queries to Splunk SPL queries by leveraging the AInvestigator SPL Vectors Agent API.

## Features

- **Natural Language to SPL**: Convert plain English queries to Splunk Search Processing Language
- **Multiple Transport Modes**: Supports both SSE (Server-Sent Events) and stdio transports
- **Docker Support**: Fully containerized deployment option
- **Configurable LLM**: Use different LLM providers (default: ollama_3.3_70b)
- **Context Control**: Optionally include system prompts and context in responses

## Prerequisites

- Python 3.11+
- Running AInvestigator SPL Vectors Agent (default: http://localhost:2993)
- MCP-compatible client (Claude Desktop, mcp-cli, etc.)

## Installation

### Local Installation

1. Clone or navigate to the directory:
```bash
cd mcp.custom-query-agent
```

2. Install dependencies:
```bash
pip install -e .
```

3. Configure environment variables in `.env`:
```env
MCP_SERVER_NAME=Custom Query Agent
MCP_SERVER_DESCRIPTION=MCP server to convert natural language queries to Splunk SPL
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=7007
CUSTOM_QUERY_AGENT_HOST=http://localhost:2993
DEFAULT_LLM_PROVIDER=ollama_3.3_70b
DEFAULT_Q_SAMPLES=10
TRANSPORT=sse
```

### Docker Installation

1. Build the Docker image:
```bash
docker-compose build
```

2. Run the container:
```bash
docker-compose up -d
```

### CloudFlare Tunnel Support

The Docker configuration now supports CloudFlare tunnels for external access:

1. Configure your CloudFlare tunnel token in `.env`:
```env
CLOUDFLARE_TOKEN=your_cloudflare_tunnel_token_here
```

2. The docker-compose.yml includes three services:
   - `custom-query-agent-int`: Internal service accessible via port (default: 7007)
   - `custom-query-agent-ext`: External service for CloudFlare tunnel
   - `custom-query-agent-cloudflare`: CloudFlare tunnel container

3. Start all services:
```bash
docker-compose up -d
```

## Usage

### Running the Server

#### SSE Mode (default)
```bash
./start_sse
# or
python server.py
```

#### stdio Mode
```bash
TRANSPORT=stdio python server.py
```

#### Docker Mode
```bash
docker-compose up
```

### Available Tools

#### 1. generate_spl_query
Converts natural language queries to Splunk SPL.

**Parameters:**
- `query` (required): Natural language query to convert
- `llm_provider` (optional): LLM provider to use (default: ollama_3.3_70b)
- `q_samples` (optional): Number of query samples for context (default: 10)
- `include_context` (optional): Include system prompt and context in response (default: false)

**Example:**
```json
{
  "tool": "generate_spl_query",
  "arguments": {
    "query": "Show me all failed login attempts in the last 24 hours",
    "include_context": false
  }
}
```

**Response:**
```json
{
  "spl_query": "index=web_traffic earliest=-24h latest=now status=401 OR status=403 | stats count by src_ip, user | sort -count",
  "viz_type": "table",
  "tool_used": "general",
  "tool_selection_reason": "No specific tool matched",
  "earliest": "-24h",
  "latest": "now"
}
```

#### 2. get_config
Returns current server configuration and connection status.

**Example:**
```json
{
  "tool": "get_config",
  "arguments": {}
}
```

**Response:**
```json
{
  "name": "Custom Query Agent",
  "description": "MCP server to convert natural language queries to Splunk SPL",
  "transport": "sse",
  "custom_query_agent_host": "http://localhost:2993",
  "agent_connected": true,
  "default_llm_provider": "ollama_3.3_70b",
  "default_q_samples": 10
}
```

## Integration with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "custom-query-agent": {
      "command": "python",
      "args": ["/path/to/mcp.custom-query-agent/server.py"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_SERVER_NAME` | Server display name | Custom Query Agent |
| `MCP_SERVER_DESCRIPTION` | Server description | MCP server to convert natural language queries to Splunk SPL |
| `MCP_SERVER_HOST` | Server host | 0.0.0.0 |
| `MCP_SERVER_PORT` | Server port | 8051 |
| `TRANSPORT` | Transport mode (sse/stdio) | sse |
| `LOG_LEVEL` | Logging level | INFO |
| `CUSTOM_QUERY_AGENT_HOST` | URL of Custom Query Agent Host | http://localhost:2993 |
| `DEFAULT_LLM_PROVIDER` | Default LLM provider | ollama_3.3_70b |
| `DEFAULT_Q_SAMPLES` | Default query samples | 10 |

## Testing

Run the test scripts in the `tests/` directory:

```bash
# Test connection
python tests/test_connection.py

# Test query generation
python tests/test_query_generation.py

# Run all tests
python tests/testall.py
```

## Deployment Scenarios

### 1. Local Development
Best for testing and development:
```bash
./start_sse
```

### 2. Production with systemd
Create `/etc/systemd/system/custom-query-agent.service`:
```ini
[Unit]
Description=Custom Query Agent MCP Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/mcp.custom-query-agent
Environment="TRANSPORT=sse"
ExecStart=/usr/bin/python3 /path/to/mcp.custom-query-agent/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable custom-query-agent
sudo systemctl start custom-query-agent
```

### 3. Docker Production
For containerized deployments:
```bash
docker-compose up -d
```

## Troubleshooting

1. **Connection Failed**: Ensure the SPL Vectors Agent is running at the configured URL
2. **Timeout Errors**: The API has a 60-second timeout; complex queries may need adjustment
3. **Docker Networking**: The Docker container uses `host.docker.internal` to connect to the host

## License

This project follows the same license as the AInvestigator project.