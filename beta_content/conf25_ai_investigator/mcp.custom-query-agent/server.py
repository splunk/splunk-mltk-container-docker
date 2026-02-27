#!/usr/bin/env python3
"""
MCP server to convert natural language queries to Splunk SPL
Supports stdio, SSE, and Streamable HTTP transports
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx
import re
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP, Context

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("MCP_SERVER_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration from environment
SERVER_NAME = os.getenv("MCP_SERVER_NAME", "Custom Query Agent")
SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "0.1.0")
HTTP_HOST = os.getenv("MCP_SERVER_HTTP_HOST", "0.0.0.0")
SSE_PORT = int(os.getenv("MCP_SERVER_SSE_PORT", "7006"))
STREAMABLE_HTTP_PORT = int(os.getenv("MCP_SERVER_STREAMABLE_HTTP_PORT", "7007"))

@dataclass
class AppContext:
    """Application context for the server."""
    config: dict
    http_client: httpx.AsyncClient

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manages the application lifecycle."""
    config = {
        "name": SERVER_NAME,
        "description": os.getenv("MCP_SERVER_DESCRIPTION", "MCP server to convert natural language queries to Splunk SPL"),
        "transport": os.getenv("MCP_SERVER_TRANSPORT", "sse"),
        "spl_vectors_agent": os.getenv("CUSTOM_QUERY_AGENT_HOST"),
        "default_llm_provider_model": os.getenv("DEFAULT_LLM_PROVIDER_MODEL"),
        "default_q_samples": int(os.getenv("DEFAULT_Q_SAMPLES", "10"))
    }
    
    # Create async HTTP client for API calls
    http_client = httpx.AsyncClient(timeout=60.0)
    
    try:
        # Test connection to SPL Vectors Agent
        try:
            response = await http_client.get(f"{config['spl_vectors_agent']}/")
            config["agent_connected"] = response.status_code == 200
        except Exception:
            config["agent_connected"] = False
            
        yield AppContext(config=config, http_client=http_client)
    finally:
        await http_client.aclose()

# Create FastMCP server with settings
mcp = FastMCP(SERVER_NAME, description=os.getenv("MCP_SERVER_DESCRIPTION"), lifespan=app_lifespan)

# Configure settings based on transport
transport = os.getenv("MCP_SERVER_TRANSPORT", "stdio").lower()
if transport == "sse":
    mcp.settings.host = HTTP_HOST
    mcp.settings.port = SSE_PORT
elif transport == "streamable-http":
    mcp.settings.host = HTTP_HOST
    mcp.settings.port = STREAMABLE_HTTP_PORT

# Add authentication middleware if configured
if os.getenv("RUNNING_INSIDE_DOCKER") and os.getenv("MCP_SERVER_API_KEY"):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class AuthMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, api_key: str):
            super().__init__(app)
            self.api_key = api_key

        async def dispatch(self, request, call_next):
            # Skip auth for health checks and OAuth discovery endpoints
            skip_auth_paths = [
                "/",
                "/health",
                "/.well-known/oauth-authorization-server",
                "/oauth/authorize",
                "/oauth/token",
                "/mcp/",  # MCP streamable HTTP endpoint
            ]
            
            # Check if the path should skip authentication
            if any(request.url.path.startswith(path) for path in skip_auth_paths):
                return await call_next(request)
            
            # Check both Authorization header and MCP_SERVER_API_KEY header for compatibility
            auth_header = request.headers.get("Authorization")
            mcp_key_header = request.headers.get("MCP_SERVER_API_KEY")
            
            # Check if the key matches either format
            valid_auth = False
            if auth_header and auth_header == f"Bearer {self.api_key}":
                valid_auth = True
            elif mcp_key_header and mcp_key_header == self.api_key:
                valid_auth = True
                
            if not valid_auth:
                logger.warning(f"Unauthorized access from {request.client.host} - Path: {request.url.path}")
                return JSONResponse(
                    {"error": "Unauthorized", "message": "Invalid or missing API key"},
                    status_code=401,
                    headers={"Content-Type": "application/json"}
                )
            
            logger.info(f"Correct key. Access authorized. From: {request.client.host} - Path: {request.url.path}")
            
            # Ensure we properly await and return the response
            try:
                response = await call_next(request)
                return response
            except Exception as e:
                logger.error(f"Error in middleware chain: {e}")
                return JSONResponse(
                    {"error": "Internal Server Error", "message": str(e)},
                    status_code=500,
                    headers={"Content-Type": "application/json"}
                )

    # We need to add the middleware before running the server
    # The issue is that FastMCP creates the app lazily, so we need to override the app method
    api_key = os.getenv("MCP_SERVER_API_KEY")
    
    if transport == "streamable-http":
        # Get the app and add middleware
        original_streamable_http_app = mcp.streamable_http_app
        
        def streamable_http_app_with_auth(*args, **kwargs):
            app = original_streamable_http_app(*args, **kwargs)
            app.add_middleware(AuthMiddleware, api_key=api_key)
            logger.info("API key authentication enabled for Streamable HTTP transport")
            return app
        
        # Replace the method
        mcp.streamable_http_app = streamable_http_app_with_auth
        
    elif transport == "sse":
        # Get the app and add middleware
        original_sse_app = mcp.sse_app
        
        def sse_app_with_auth(*args, **kwargs):
            app = original_sse_app(*args, **kwargs)
            app.add_middleware(AuthMiddleware, api_key=api_key)
            logger.info("API key authentication enabled for SSE transport")
            return app
        
        # Replace the method
        mcp.sse_app = sse_app_with_auth


@mcp.tool()
async def generate_spl_query(
    ctx: Context, 
    query: str, 
    llm_provider_model: Optional[str] = None,
    q_samples: Optional[int] = None,
    include_context: bool = False
) -> Dict[str, Any]:
    """
    Convert a natural language query to Splunk SPL query, ready to be executed by Splunk MCP server, one shot query tool
    
    Args:
        query: Natural language query to convert to SPL
        llm_provider_model: LLM provider/model to use (default: llama3.3:70b-instruct-q4_K_M)
        q_samples: Number of query samples to use for context (default: 10)
        include_context: Include the system prompt and context in response
        
    Returns:
        Dictionary containing:
        - spl_query: The generated SPL query
        - viz_type: Visualization type (table, chart, etc.)
        - tool_used: Which tool was used (general, find_anomalous_ip_addresses, etc.)
        - tool_selection_reason: Why the tool was selected
        - earliest: Time range start
        - latest: Time range end
        - system_prompt: System prompt used (if include_context=True)
        - llm_query: Full query sent to LLM (if include_context=True)
    """
    app_ctx = ctx.request_context.lifespan_context
    config = app_ctx.config
    http_client = app_ctx.http_client
    
    # Use defaults if not provided
    if llm_provider_model is None:
        llm_provider_model = config["default_llm_provider_model"]
    if q_samples is None:
        q_samples = config["default_q_samples"]
    
    # Prepare request payload
    query_cleaned = re.sub(r'^\s*use\s+splunk\s+to\s+', '', query, flags=re.IGNORECASE)
    payload = {
        "query": query_cleaned,
        "llm_provider": llm_provider_model,
        "q_samples": q_samples,
        "include_context": include_context
    }
    
    try:
        # Call the SPL Vectors Agent API
        response = await http_client.post(
            f"{config['spl_vectors_agent']}/generate_query",
            json=payload
        )
        
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get("error", error_detail)
            except:
                pass
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": error_detail
            }
        
        result = response.json()
        
        # Remove context fields if not requested
        if not include_context:
            result.pop("system_prompt", None)
            result.pop("llm_query", None)
        
        if 'spl_query' in result: 
            result['spl_query'] = result['spl_query'].replace('\\"', '"')
            # Replace actual newlines with spaces for cleaner SPL
            result['spl_query'] = result['spl_query'].replace('\n', ' ')
            # Also replace literal \n if any exist
            result['spl_query'] = result['spl_query'].replace('\\n', ' ')
            # Clean up multiple spaces
            result['spl_query'] = ' '.join(result['spl_query'].split())
        return result
        
    except httpx.TimeoutException:
        return {
            "error": "Request timed out",
            "details": "The SPL Vectors Agent took too long to respond"
        }
    except httpx.ConnectError:
        return {
            "error": "Connection failed",
            "details": f"Could not connect to SPL Vectors Agent at {config['spl_vectors_agent']}"
        }
    except Exception as e:
        return {
            "error": "Unexpected error",
            "details": str(e)
        }


@mcp.tool()
async def get_config(ctx: Context) -> Dict[str, Any]:
    """
    Get current server configuration and status.
    
    Returns:
        Dictionary containing:
        - name: Server name
        - description: Server description
        - transport: Current transport mode (sse/stdio)
        - spl_vectors_agent: URL of the SPL Vectors Agent
        - agent_connected: Connection status to SPL Vectors Agent
        - default_llm_provider_model: Default LLM provider/model
        - default_q_samples: Default number of query samples
    """
    config = ctx.request_context.lifespan_context.config.copy()
    
    # Test current connection status
    http_client = ctx.request_context.lifespan_context.http_client
    try:
        response = await http_client.get(
            f"{config['spl_vectors_agent']}/",
            timeout=5.0
        )
        config["agent_connected"] = response.status_code == 200
    except Exception:
        config["agent_connected"] = False
    
    # Return relevant configuration
    return {
        "name": config.get("name"),
        "description": config.get("description"),
        "version": SERVER_VERSION,
        "transport": config.get("transport"),
        "spl_vectors_agent": config.get("spl_vectors_agent"),
        "agent_connected": config.get("agent_connected"),
        "default_llm_provider_model": config.get("default_llm_provider_model"),
        "default_q_samples": config.get("default_q_samples")
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_SERVER_TRANSPORT", "stdio").lower()
    
    try:
        logger.info(f"Starting server with {transport} transport")
        # Cast to proper literal type for type checking
        if transport in ["stdio", "sse", "streamable-http"]:
            mcp.run(transport=transport)  # type: ignore
        else:
            logger.error(f"Invalid transport: {transport}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)