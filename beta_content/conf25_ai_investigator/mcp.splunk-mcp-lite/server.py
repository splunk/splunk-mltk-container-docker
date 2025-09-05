#!/usr/bin/env python3
"""
MCP server for executing Splunk SPL search queries
Supports stdio, SSE, and Streamable HTTP transports
"""
import os
import sys
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
import re
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from helpers import format_events_as_markdown, format_events_as_csv, format_events_as_summary
from splunk_client import SplunkClient, SplunkAPIError
from guardrails import validate_spl_query, sanitize_output

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("MCP_SERVER_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration from environment
SERVER_NAME = os.getenv("MCP_SERVER_NAME", "Splunk MCP Lite")
SERVER_VERSION = os.getenv("MCP_SERVER_VERSION", "0.1.0")
HTTP_HOST = os.getenv("MCP_SERVER_HTTP_HOST", "0.0.0.0")
SSE_PORT = int(os.getenv("MCP_SERVER_SSE_PORT", "7008"))
STREAMABLE_HTTP_PORT = int(os.getenv("MCP_SERVER_STREAMABLE_HTTP_PORT", "7009"))

@dataclass
class AppContext:
    """Application context for the server."""
    config: dict
    splunk_client: Optional[SplunkClient] = None

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manages the application lifecycle."""
    # Check if running inside Docker and use appropriate Splunk host
    if os.getenv("RUNNING_INSIDE_DOCKER") == "1":
        print ("---> Running inside Docker, using host.docker.internal for Splunk host")
        splunk_host = os.getenv("SPLUNK_HOST_FOR_DOCKER", "host.docker.internal")
    else:
        print ("---> Running outside Docker, using localhost for Splunk host")
        splunk_host = os.getenv("SPLUNK_HOST", "localhost")
    
    config = {
        "name": SERVER_NAME,
        "description": os.getenv("MCP_SERVER_DESCRIPTION", "Lite MCP server for executing Splunk SPL search queries"),
        "transport": os.getenv("MCP_SERVER_TRANSPORT", "sse"),
        "splunk_host": splunk_host,
        "splunk_port": int(os.getenv("SPLUNK_PORT", "8089")),
        "splunk_username": os.getenv("SPLUNK_USERNAME"),
        "splunk_password": os.getenv("SPLUNK_PASSWORD"),
        "splunk_token": os.getenv("SPLUNK_TOKEN"),
        "verify_ssl": os.getenv("VERIFY_SSL", "false").lower() == "true",
        "spl_max_events_count": int(os.getenv("SPL_MAX_EVENTS_COUNT", "100000")),
        "spl_risk_tolerance": int(os.getenv("SPL_RISK_TOLERANCE", "75")),
        "spl_safe_timerange": os.getenv("SPL_SAFE_TIMERANGE", "24h"),
        "spl_sanitize_output": os.getenv("SPL_SANITIZE_OUTPUT", "false").lower() == "true"
    }
    
    # Create Splunk client
    splunk_client = SplunkClient(config)
    try:
        await splunk_client.connect()
        yield AppContext(config=config, splunk_client=splunk_client)
    finally:
        await splunk_client.disconnect()

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
            # Skip auth for health checks
            if request.url.path == "/" or request.url.path == "/health":
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
                    status_code=401
                )
            logger.info(f"Correct key. Access authorized. From: {request.client.host} - Path: {request.url.path}")
            return await call_next(request)

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
async def validate_spl(ctx: Context, query: str) -> Dict[str, Any]:
    """
    Validate an SPL query for potential risks and inefficiencies.
    
    Args:
        query: The SPL query to validate
        
    Returns:
        Dictionary containing:
        - risk_score: Risk score from 0-100
        - risk_message: Explanation of risks found with suggestions
        - risk_tolerance: Current risk tolerance setting
        - would_execute: Whether this query would execute or be blocked
    """
    config = ctx.request_context.lifespan_context.config
    safe_timerange = config.get("spl_safe_timerange", "24h")
    risk_tolerance = config.get("spl_risk_tolerance", 75)
    
    risk_score, risk_message = validate_spl_query(query, safe_timerange)
    
    return {
        "risk_score": risk_score,
        "risk_message": risk_message,
        "risk_tolerance": risk_tolerance,
        "would_execute": risk_score <= risk_tolerance,
        "execution_note": f"Query would be {'executed' if risk_score <= risk_tolerance else 'BLOCKED - no search would be executed and no data would be returned'}"
    }


@mcp.tool()
async def search_oneshot(ctx: Context, query: str, earliest_time: str = "-24h", latest_time: str = "now", max_count: int = 100, output_format: str = "json", risk_tolerance: Optional[int] = None, sanitize_output: Optional[bool] = None) -> Dict[str, Any]:
    """
    Run a oneshot search query in Splunk and return results.
    Use MCP custom query agent to generate complex SPL queries ready to be executed here.
    
    Args:
        query: The Splunk search query (e.g., "index=main | head 10")
        earliest_time: Start time for search (default: -24h)
        latest_time: End time for search (default: now)
        max_count: Maximum number of results to return (default: 100, or SPL_MAX_EVENTS_COUNT from .env, 0 = unlimited)
        output_format: Format for results - json, markdown/md, csv, or summary (default: json)
        risk_tolerance: Override risk tolerance level (default: SPL_RISK_TOLERANCE from .env)
        sanitize_output: Override output sanitization (default: SPL_SANITIZE_OUTPUT from .env)
    
    Returns:
        Dictionary containing search results in the specified format
    """
    if not ctx.request_context.lifespan_context.splunk_client:
        return {"error": "Splunk client not initialized"}
    
    try:
        client = ctx.request_context.lifespan_context.splunk_client
        config = ctx.request_context.lifespan_context.config
        
        # Get risk tolerance and sanitization settings
        if risk_tolerance is None:
            risk_tolerance = config.get("spl_risk_tolerance", 75)
        if sanitize_output is None:
            sanitize_output = config.get("spl_sanitize_output", False)
        
        # Validate query if risk_tolerance < 100
        if risk_tolerance < 100:
            safe_timerange = config.get("spl_safe_timerange", "24h")
            risk_score, risk_message = validate_spl_query(query, safe_timerange)
            
            if risk_score > risk_tolerance:
                return {
                    "error": f"Query exceeds risk tolerance ({risk_score} > {risk_tolerance}). No search was executed and no data was returned.",
                    "risk_score": risk_score,
                    "risk_tolerance": risk_tolerance,
                    "risk_message": risk_message,
                    "search_executed": False,
                    "data_returned": None
                }
        
        # Use configured spl_max_events_count if max_count is default (100)
        if max_count == 100:
            max_count = config.get("spl_max_events_count", 100000)
        
        # ###! Hardcoding for demo datasets:
        if any(pattern in query for pattern in ["fraud_cms.", "fraud_web.", "index=web_traffic"]):
            earliest_time, latest_time = ("0", "now")        
            # Apply regex substitutions to remove time range from query
            query = re.sub(r"latest=[^\s]+", "*", query)
            query = re.sub(r"earliest=[^\s]+", "", query)

        # Execute search using client
        events = await client.search_oneshot(query, earliest_time, latest_time, max_count)
        
        # Sanitize output if requested
        if sanitize_output:
            from guardrails import sanitize_output as sanitize_fn
            events = sanitize_fn(events)
        
        # Format results based on output_format
        # Handle synonyms
        if output_format == "md":
            output_format = "markdown"
            
        if output_format == "json":
            return {
                "query": query,
                "event_count": len(events),
                "events": events,
                "search_params": {
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "max_count": max_count
                }
            }
        elif output_format == "markdown":
            return {
                "query": query,
                "event_count": len(events),
                "format": "markdown",
                "content": format_events_as_markdown(events, query),
                "search_params": {
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "max_count": max_count
                }
            }
        elif output_format == "csv":
            return {
                "query": query,
                "event_count": len(events),
                "format": "csv",
                "content": format_events_as_csv(events, query),
                "search_params": {
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "max_count": max_count
                }
            }
        elif output_format == "summary":
            return {
                "query": query,
                "event_count": len(events),
                "format": "summary",
                "content": format_events_as_summary(events, query, len(events)),
                "search_params": {
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "max_count": max_count
                }
            }
        else:
            return {"error": f"Invalid output_format: {output_format}. Must be one of: json, markdown (or md), csv, summary"}
        
    except SplunkAPIError as e:
        return {"error": str(e), "details": e.details}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@mcp.tool()
async def search_export(ctx: Context, query: str, earliest_time: str = "-24h", latest_time: str = "now", max_count: int = 100, output_format: str = "json", risk_tolerance: Optional[int] = None, sanitize_output: Optional[bool] = None) -> Dict[str, Any]:
    """
    Run an export search query in Splunk that streams results immediately.
    
    Args:
        query: The Splunk search query
        earliest_time: Start time for search (default: -24h)
        latest_time: End time for search (default: now)
        max_count: Maximum number of results to return (default: 100, or SPL_MAX_EVENTS_COUNT from .env, 0 = unlimited)
        output_format: Format for results - json, markdown/md, csv, or summary (default: json)
        risk_tolerance: Override risk tolerance level (default: SPL_RISK_TOLERANCE from .env)
        sanitize_output: Override output sanitization (default: SPL_SANITIZE_OUTPUT from .env)
    
    Returns:
        Dictionary containing search results in the specified format
    """
    if not ctx.request_context.lifespan_context.splunk_client:
        return {"error": "Splunk client not initialized"}
    
    try:
        client = ctx.request_context.lifespan_context.splunk_client
        config = ctx.request_context.lifespan_context.config
        
        # Get risk tolerance and sanitization settings
        if risk_tolerance is None:
            risk_tolerance = config.get("spl_risk_tolerance", 75)
        if sanitize_output is None:
            sanitize_output = config.get("spl_sanitize_output", False)
        
        # Validate query if risk_tolerance < 100
        if risk_tolerance < 100:
            safe_timerange = config.get("spl_safe_timerange", "24h")
            risk_score, risk_message = validate_spl_query(query, safe_timerange)
            
            if risk_score > risk_tolerance:
                return {
                    "error": f"Query exceeds risk tolerance ({risk_score} > {risk_tolerance}). No search was executed and no data was returned.",
                    "risk_score": risk_score,
                    "risk_tolerance": risk_tolerance,
                    "risk_message": risk_message,
                    "search_executed": False,
                    "data_returned": None
                }
        
        # Use configured spl_max_events_count if max_count is default (100)
        if max_count == 100:
            max_count = config.get("spl_max_events_count", 100000)
        
        # Execute export search using client
        events = await client.search_export(query, earliest_time, latest_time, max_count)
        
        # Sanitize output if requested
        if sanitize_output:
            from guardrails import sanitize_output as sanitize_fn
            events = sanitize_fn(events)
        
        # Format results based on output_format
        # Handle synonyms
        if output_format == "md":
            output_format = "markdown"
            
        if output_format == "json":
            return {
                "query": query,
                "event_count": len(events),
                "events": events,
                "is_preview": False
            }
        elif output_format == "markdown":
            return {
                "query": query,
                "event_count": len(events),
                "format": "markdown",
                "content": format_events_as_markdown(events, query),
                "is_preview": False
            }
        elif output_format == "csv":
            return {
                "query": query,
                "event_count": len(events),
                "format": "csv",
                "content": format_events_as_csv(events, query),
                "is_preview": False
            }
        elif output_format == "summary":
            return {
                "query": query,
                "event_count": len(events),
                "format": "summary",
                "content": format_events_as_summary(events, query, len(events)),
                "is_preview": False
            }
        else:
            return {"error": f"Invalid output_format: {output_format}. Must be one of: json, markdown (or md), csv, summary"}
        
    except SplunkAPIError as e:
        return {"error": str(e), "details": e.details}
    except Exception as e:
        return {"error": f"Export search failed: {str(e)}"}

@mcp.tool()
async def get_indexes(ctx: Context) -> Dict[str, Any]:
    """
    Get list of available Splunk indexes with detailed information.
    
    Returns:
        Dictionary containing list of indexes with their properties including:
        - name, datatype, event count, size, time range, and more
    """
    if not ctx.request_context.lifespan_context.splunk_client:
        return {"error": "Splunk client not initialized"}
    
    try:
        client = ctx.request_context.lifespan_context.splunk_client
        indexes = await client.get_indexes()
        
        return {"indexes": indexes, "count": len(indexes)}
        
    except SplunkAPIError as e:
        return {"error": str(e), "details": e.details}
    except Exception as e:
        return {"error": f"Failed to get indexes: {str(e)}"}


@mcp.tool()
async def get_config(ctx: Context) -> dict:
    """Get current server configuration."""
    config = ctx.request_context.lifespan_context.config.copy()
    # Remove sensitive information
    config.pop("splunk_password", None)
    config.pop("splunk_token", None)
    config["splunk_connected"] = ctx.request_context.lifespan_context.splunk_client is not None
    return config


@mcp.resource("splunk://indexes")
async def get_indexes_resource() -> str:
    """Provide index information as a resource with detailed metadata."""
    # Create a temporary client for resource access
    config = {
        "splunk_host": os.getenv("SPLUNK_HOST"),
        "splunk_port": int(os.getenv("SPLUNK_PORT", "8089")),
        "splunk_username": os.getenv("SPLUNK_USERNAME"),
        "splunk_password": os.getenv("SPLUNK_PASSWORD"),
        "splunk_token": os.getenv("SPLUNK_TOKEN"),
        "verify_ssl": os.getenv("VERIFY_SSL", "false").lower() == "true"
    }
    
    try:
        async with SplunkClient(config) as client:
            indexes = await client.get_indexes()
            
            content = "# Splunk Indexes\n\n"
            content += "| Index | Type | Events | Size (MB) | Max Size | Time Range | Status |\n"
            content += "|-------|------|--------|-----------|----------|------------|--------|\n"
            
            for idx in indexes:
                time_range = "N/A"
                if idx.get('minTime') and idx.get('maxTime'):
                    time_range = f"{idx['minTime']} to {idx['maxTime']}"
                    
                status = "✓ Enabled" if not idx.get('disabled', False) else "✗ Disabled"
                max_size = idx.get('maxDataSize', 'auto')
                
                content += f"| {idx['name']} | {idx.get('datatype', 'event')} | "
                content += f"{idx.get('totalEventCount', 0):,} | "
                content += f"{idx.get('currentDBSizeMB', 0):,.2f} | "
                content += f"{max_size} | {time_range} | {status} |\n"
                
            content += "\n## Index Details\n\n"
            
            for idx in indexes:
                if idx.get('totalEventCount', 0) > 0:  # Only show non-empty indexes
                    content += f"### {idx['name']}\n"
                    content += f"- **Total Events:** {idx.get('totalEventCount', 0):,}\n"
                    content += f"- **Current Size:** {idx.get('currentDBSizeMB', 0):,.2f} MB\n"
                    content += f"- **Max Size:** {idx.get('maxDataSize', 'auto')}\n"
                    if idx.get('frozenTimePeriodInSecs'):
                        frozen_days = int(idx['frozenTimePeriodInSecs']) / 86400
                        content += f"- **Retention:** {frozen_days:.0f} days\n"
                    content += "\n"
                
            return content
            
    except Exception as e:
        return f"Error retrieving indexes: {str(e)}"

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