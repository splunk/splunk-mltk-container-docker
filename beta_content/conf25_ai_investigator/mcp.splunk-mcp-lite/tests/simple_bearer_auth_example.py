"""
Simple Bearer Token Authentication Example for MCP Protocol

This example demonstrates how to implement bearer token authentication
with the MCP Python SDK using httpx.Auth interface.
"""

import httpx
from typing import Generator
from httpx import Request, Response


class BearerTokenAuth(httpx.Auth):
    """
    Simple Bearer Token authentication for httpx.
    
    This class implements the httpx.Auth interface to add
    Authorization headers with Bearer tokens to requests.
    """
    
    def __init__(self, token: str):
        """
        Initialize with a bearer token.
        
        Args:
            token: The bearer token to use for authentication
        """
        self.token = token
    
    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """
        Apply bearer token authentication to the request.
        
        This method is called by httpx to apply authentication.
        It adds the Authorization header with the bearer token.
        
        Args:
            request: The httpx Request object
            
        Yields:
            The modified request with authentication header
        """
        # Add the Authorization header with Bearer token
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class APIKeyAuth(httpx.Auth):
    """
    Simple API Key authentication for httpx.
    
    This class implements the httpx.Auth interface to add
    API key headers to requests.
    """
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        """
        Initialize with an API key.
        
        Args:
            api_key: The API key to use for authentication
            header_name: The header name for the API key (default: X-API-Key)
        """
        self.api_key = api_key
        self.header_name = header_name
    
    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        """
        Apply API key authentication to the request.
        
        Args:
            request: The httpx Request object
            
        Yields:
            The modified request with API key header
        """
        # Add the API key header
        request.headers[self.header_name] = self.api_key
        yield request


# Example usage with MCP client transports
async def example_usage():
    """Example of using authentication with MCP transports."""
    
    # Import MCP client modules
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    from datetime import timedelta
    
    # Example 1: Bearer Token Authentication
    bearer_auth = BearerTokenAuth(token="your-bearer-token-here")
    
    # Use with SSE transport
    async with sse_client(
        url="http://localhost:8052/sse",
        auth=bearer_auth,
        timeout=60
    ) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # Use the authenticated session...
    
    # Example 2: API Key Authentication
    api_auth = APIKeyAuth(api_key="your-api-key-here", header_name="X-API-Key")
    
    # Use with StreamableHTTP transport
    async with streamablehttp_client(
        url="http://localhost:8052/mcp",
        auth=api_auth,
        timeout=timedelta(seconds=60)
    ) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # Use the authenticated session...
            
    # Example 3: Custom headers approach (without auth parameter)
    # You can also pass authentication headers directly
    headers = {
        "Authorization": "Bearer your-token-here",
        # or
        "X-API-Key": "your-api-key-here"
    }
    
    async with streamablehttp_client(
        url="http://localhost:8052/mcp",
        headers=headers,
        timeout=timedelta(seconds=60)
    ) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # Use the authenticated session...


if __name__ == "__main__":
    import asyncio
    # asyncio.run(example_usage())
    print("See the example_usage() function for authentication examples")