# Splunk MCP Lite Server

A lightweight MCP (Model Context Protocol) server focused on executing Splunk SPL search queries with built-in validation and index discovery.

## Features

- **FastMCP Framework**: Simplified server implementation
- **Splunk Integration**: Direct REST API integration (no SDK dependency)
- **Multiple Transports**: Supports both stdio and SSE
- **Docker Support**: Ready for containerized deployment
- **SPL Query Validation**: Built-in guardrails to detect risky, inefficient, or destructive queries
- **Output Sanitization**: Automatic masking of sensitive data (credit cards, SSNs)
- **Multiple Output Formats**: JSON, Markdown, CSV, and Summary formats
- **Search-Focused Tools**:
  - `validate_spl`: Validate SPL queries for risks and inefficiencies
  - `search_oneshot`: Run blocking search queries
  - `search_export`: Stream search results immediately
  - `get_indexes`: List available Splunk indexes
  - `get_config`: Get server configuration

## Quick Start

### 1. Setup Environment

```bash
cd /path/to/splunk-mcp-lite
cp .env.example .env
# Edit .env with your Splunk connection details
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Run the Server

**SSE Mode (default):**
```bash
python server.py
```

**Stdio Mode:**
Configure via environment or let the client spawn the server.

### 4. Test with Example Clients

```bash
cd tests

# For SSE transport
python test_sse_transport.py

# For stdio transport
python test_stdio_transport.py

# Test SPL validation interactively
python validate_spl_test.py

# Interactive SPL search
python splunk_sse_search.py

# Or use the comprehensive test menu
./testall
```

## Docker Deployment

```bash
# Using the dock script for common operations:

# Build and start containers
./dock rebuild

# Start containers
./dock up   # or ./dock start

# Stop containers
./dock down # or ./dock stop

# Restart containers
./dock restart

# Manual docker commands:
sudo docker compose build --no-cache
sudo docker compose up -d
```

## Client Configuration

### Claude Desktop

**SSE Mode Configuration:**

Edit your Claude Desktop configuration file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "splunk-mcp-lite": {
      "url": "http://localhost:8052/sse"
    }
  }
}
```

**Stdio Mode Configuration:**

```json
{
  "mcpServers": {
    "splunk-mcp-lite": {
      "command": "python",
      "args": ["/path/to/splunk-mcp-lite/server.py"],
      "env": {
        "TRANSPORT": "stdio",
        "SPLUNK_HOST": "your-splunk-host",
        "SPLUNK_USERNAME": "your-username",
        "SPLUNK_PASSWORD": "your-password"
      }
    }
  }
}
```

### Claude Code

**SSE Mode Configuration:**

Start the server:
```bash
cd /path/to/splunk-mcp-lite
python server.py
```

Add to Claude Code:
```bash
claude mcp add --transport sse --scope project splunk-mcp-lite http://localhost:8052/sse
```

**Stdio Mode Configuration:**

```bash
cd /path/to/splunk-mcp-lite
claude mcp add splunk-mcp-lite -e TRANSPORT=stdio --scope project -e SPLUNK_HOST=your-host -e SPLUNK_USERNAME=your-user -e SPLUNK_PASSWORD=your-pass -- python server.py

# claude mcp remove splunk-mcp-lite [--scope project]
```

## Available Tools

### validate_spl
Validate an SPL query for potential risks and inefficiencies before execution.

Parameters:
- `query`: The SPL query to validate

Returns:
- `risk_score`: Risk score from 0-100
- `risk_message`: Detailed explanation of risks found with suggestions
- `risk_tolerance`: Current risk tolerance setting  
- `would_execute`: Whether this query would execute or be blocked
- `execution_note`: Clear message about execution status

Example:
```
validate_spl("index=* | delete")
```

### search_oneshot
Run a blocking search query and return results.

Parameters:
- `query`: Splunk search query (e.g., "index=main | head 10")
- `earliest_time`: Start time (default: "-24h")
- `latest_time`: End time (default: "now")
- `max_count`: Maximum results (default: 100 or SPL_MAX_EVENTS_COUNT)
- `output_format`: Format for results - json, markdown/md, csv, or summary (default: "json")
- `risk_tolerance`: Override risk tolerance level (default: SPL_RISK_TOLERANCE)
- `sanitize_output`: Override output sanitization (default: SPL_SANITIZE_OUTPUT)

Example:
```
search_oneshot("index=_internal | head 5", earliest_time="-1h", output_format="markdown")
```

### search_export
Stream search results immediately without creating a job.

Parameters:
- `query`: Splunk search query
- `earliest_time`: Start time (default: "-24h")
- `latest_time`: End time (default: "now")
- `max_count`: Maximum results (default: 100 or SPL_MAX_EVENTS_COUNT)
- `output_format`: Format for results - json, markdown/md, csv, or summary (default: "json")
- `risk_tolerance`: Override risk tolerance level (default: SPL_RISK_TOLERANCE)
- `sanitize_output`: Override output sanitization (default: SPL_SANITIZE_OUTPUT)

### get_indexes
List all available Splunk indexes with properties.

### get_config
Get current server configuration (excludes sensitive data).

## Configuration

Edit `.env` file to customize:

```bash
# Server Configuration
MCP_SERVER_NAME=Splunk MCP Lite
MCP_SERVER_DESCRIPTION=Lite MCP server for executing Splunk SPL search queries
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8052
TRANSPORT=sse
LOG_LEVEL=INFO

# Splunk Configuration
SPLUNK_HOST=localhost
SPLUNK_PORT=8089
SPLUNK_USERNAME=admin
SPLUNK_PASSWORD=changeme
# Optional: Use token instead of username/password
SPLUNK_TOKEN=
# SSL verification
VERIFY_SSL=false

# Search Configuration
# Maximum number of events to return from searches (0 = unlimited)
SPL_MAX_EVENTS_COUNT=100000

# Risk tolerance level for SPL query validation (0 = reject all risky queries, 100 = allow all)
SPL_RISK_TOLERANCE=75

# Safe time range for searches - queries within this range get no time penalty
SPL_SAFE_TIMERANGE=24h

# Enable output sanitization to mask sensitive data (credit cards, SSNs)
SPL_SANITIZE_OUTPUT=false
```

## Common Splunk Search Patterns

### Get recent errors:
```
index=main level=ERROR | head 20
```

### Get statistics by source:
```
index=_internal | stats count by source
```

### Time-based searches:
```
index=main earliest=-1h latest=now | timechart count
```

### Field extraction:
```
index=web_logs | rex field=_raw "status=(?<status_code>\d+)" | stats count by status_code
```

## SPL Query Validation (Guardrails)

The server includes built-in validation to detect potentially risky or inefficient SPL queries before execution. This helps prevent:

- **Destructive Operations**: Commands like `delete` that permanently remove data
- **Performance Issues**: Unbounded searches, expensive commands, or missing time ranges
- **Resource Consumption**: Queries that could overwhelm system resources  
- **Security Risks**: External script execution or unsafe operations

### Risk Scoring

Each query is analyzed and assigned a risk score from 0-100:
- **0-30**: Low risk - Query is generally safe
- **31-60**: Medium risk - Query may have performance implications
- **61-100**: High risk - Query could be destructive or severely impact performance

### Common Risk Factors

1. **Destructive Commands** (High Risk):
   - `delete` command (+80 points)
   - `collect` with `override=true` (+25 points)
   - `outputlookup` with `override=true` (+20 points)

2. **Time Range Issues**:
   - All-time searches or missing time constraints (+50 points)
   - Time ranges exceeding safe threshold (+20 points)

3. **Performance Concerns**:
   - `index=*` without constraints (+35 points)
   - Expensive commands like `transaction`, `map`, `join` (+20 points each)
   - Missing index specification (+20 points)
   - Subsearches without limits (+20 points)

4. **Security Risks**:
   - External script execution (+40 points)

### Configuration

Configure validation behavior in your `.env` file:
- `SPL_RISK_TOLERANCE`: Set threshold for blocking queries (default: 75)
- `SPL_SAFE_TIMERANGE`: Define safe time range (default: 24h)
- `SPL_SANITIZE_OUTPUT`: Enable/disable output sanitization

### Output Sanitization

When `SPL_SANITIZE_OUTPUT=true`, the server automatically masks sensitive data:
- **Credit Cards**: Shows only last 4 digits (e.g., ****-****-****-1234)
- **Social Security Numbers**: Completely masked (e.g., ***-**-****)

### Testing Validation

Use the interactive validation tool to test queries:

```bash
cd tests
python validate_spl_test.py
```

## Security Notes

- Store credentials securely in `.env` file
- Use token authentication when possible
- Enable SSL verification in production (`VERIFY_SSL=true`)
- Never commit `.env` file to version control

## Troubleshooting

### Connection Issues
- Verify Splunk is running and accessible
- Check firewall rules for port 8089
- Ensure credentials are correct
- Try disabling SSL verification for testing

### Search Issues
- Verify user has search permissions
- Check index access rights
- Use `| head` to limit results during testing
- Check Splunk search job limits

## Project Structure

```
splunk-mcp-lite/
├── .env                  # Environment configuration
├── server.py             # Main server implementation (lite version)
├── splunk_client.py      # Splunk REST API client
├── helpers.py            # Formatting utilities
├── guardrails.py         # SPL query validation and sanitization
├── spl_risk_rules.py     # Configurable risk rules for validation
├── pyproject.toml        # Python project configuration
├── Dockerfile            # Docker deployment configuration
├── docker-compose.yml    # Docker compose configuration
├── dock                  # Docker management script
├── start_sse             # SSE server startup script
├── README.md             # This file
└── tests/
    ├── test_sse_transport.py    # SSE transport tests
    ├── test_stdio_transport.py  # Stdio transport tests
    ├── validate_spl_test.py     # Interactive validation tool
    ├── splunk_sse_search.py     # Interactive search tool
    ├── testall                  # Comprehensive test menu
    └── quick_test.py            # Quick smoke test
```