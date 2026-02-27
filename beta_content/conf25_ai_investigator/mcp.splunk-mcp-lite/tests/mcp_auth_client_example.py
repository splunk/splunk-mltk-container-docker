#!/usr/bin/env python3
"""
MCP Client with Authentication Examples

This example demonstrates various authentication methods for MCP clients:
1. Bearer Token Authentication
2. API Key Authentication  
3. Basic Authentication
4. Custom Header Authentication
"""

import asyncio
import os
from typing import Optional
import httpx
from datetime import timedelta
from httpx import Request, Response
from typing import Generator

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client


class BearerTokenAuth(httpx.Auth):
    """Bearer Token authentication handler."""
    
    def __init__(self, token: str):
        self.token = token
    
    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class APIKeyAuth(httpx.Auth):
    """API Key authentication handler."""
    
    def __init__(self, api_key: str, header_name: str = "X-API-Key"):
        self.api_key = api_key
        self.header_name = header_name
    
    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        request.headers[self.header_name] = self.api_key
        yield request


class BasicAuth(httpx.Auth):
    """Basic Authentication handler."""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
    
    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        # httpx has built-in BasicAuth, but this shows the pattern
        import base64
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        request.headers["Authorization"] = f"Basic {encoded}"
        yield request


class AuthenticatedMCPClient:
    """MCP Client with authentication support."""
    
    def __init__(
        self, 
        server_url: str,
        auth: Optional[httpx.Auth] = None,
        headers: Optional[dict[str, str]] = None,
        transport_type: str = "streamable_http"
    ):
        self.server_url = server_url
        self.auth = auth
        self.headers = headers or {}
        self.transport_type = transport_type
        self.session: Optional[ClientSession] = None
    
    async def connect(self):
        """Connect to the MCP server with authentication."""
        print(f"🔗 Connecting to {self.server_url} with {self.transport_type} transport...")
        
        try:
            if self.transport_type == "sse":
                async with sse_client(
                    url=self.server_url,
                    auth=self.auth,
                    headers=self.headers,
                    timeout=60
                ) as (read_stream, write_stream):
                    await self._run_session(read_stream, write_stream, None)
            else:
                async with streamablehttp_client(
                    url=self.server_url,
                    auth=self.auth,
                    headers=self.headers,
                    timeout=timedelta(seconds=60)
                ) as (read_stream, write_stream, get_session_id):
                    await self._run_session(read_stream, write_stream, get_session_id)
                    
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
    
    async def _run_session(self, read_stream, write_stream, get_session_id):
        """Run the MCP session."""
        async with ClientSession(read_stream, write_stream) as session:
            self.session = session
            
            print("⚡ Initializing session...")
            result = await session.initialize()
            print(f"✅ Connected! Protocol version: {result.protocolVersion}")
            
            if get_session_id:
                session_id = get_session_id()
                if session_id:
                    print(f"📋 Session ID: {session_id}")
            
            # List available tools
            await self.list_tools()
            
            # Example: Call a tool if available
            # await self.call_tool("example_tool", {"param": "value"})
    
    async def list_tools(self):
        """List available tools from the server."""
        if not self.session:
            print("❌ Not connected")
            return
        
        result = await self.session.list_tools()
        if result.tools:
            print("\n📦 Available tools:")
            for tool in result.tools:
                print(f"  - {tool.name}: {tool.description}")
        else:
            print("📦 No tools available")
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the server."""
        if not self.session:
            print("❌ Not connected")
            return
        
        print(f"\n🔧 Calling tool '{tool_name}' with args: {arguments}")
        result = await self.session.call_tool(tool_name, arguments)
        
        if hasattr(result, 'content'):
            for content in result.content:
                if content.type == "text":
                    print(f"📤 Result: {content.text}")
                else:
                    print(f"📤 Result: {content}")


async def main():
    """Demonstrate different authentication methods."""
    
    # Get configuration from environment or use defaults
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8052")
    transport = os.getenv("MCP_TRANSPORT", "streamable_http")
    
    # Determine the endpoint based on transport type
    if transport == "sse":
        endpoint = f"{server_url}/sse"
    else:
        endpoint = f"{server_url}/mcp"
    
    print("🚀 MCP Authentication Examples")
    print(f"Server: {endpoint}")
    print(f"Transport: {transport}\n")
    
    # Example 1: Bearer Token Authentication
    if os.getenv("BEARER_TOKEN"):
        print("=== Example 1: Bearer Token Authentication ===")
        bearer_auth = BearerTokenAuth(token=os.getenv("BEARER_TOKEN"))
        client = AuthenticatedMCPClient(
            server_url=endpoint,
            auth=bearer_auth,
            transport_type=transport
        )
        try:
            await client.connect()
        except Exception as e:
            print(f"Bearer auth failed: {e}\n")
    
    # Example 2: API Key Authentication
    if os.getenv("API_KEY"):
        print("\n=== Example 2: API Key Authentication ===")
        api_auth = APIKeyAuth(
            api_key=os.getenv("API_KEY"),
            header_name=os.getenv("API_KEY_HEADER", "X-API-Key")
        )
        client = AuthenticatedMCPClient(
            server_url=endpoint,
            auth=api_auth,
            transport_type=transport
        )
        try:
            await client.connect()
        except Exception as e:
            print(f"API key auth failed: {e}\n")
    
    # Example 3: Basic Authentication
    if os.getenv("BASIC_USERNAME") and os.getenv("BASIC_PASSWORD"):
        print("\n=== Example 3: Basic Authentication ===")
        basic_auth = BasicAuth(
            username=os.getenv("BASIC_USERNAME"),
            password=os.getenv("BASIC_PASSWORD")
        )
        client = AuthenticatedMCPClient(
            server_url=endpoint,
            auth=basic_auth,
            transport_type=transport
        )
        try:
            await client.connect()
        except Exception as e:
            print(f"Basic auth failed: {e}\n")
    
    # Example 4: Custom Headers (no auth parameter)
    if os.getenv("CUSTOM_AUTH_HEADER"):
        print("\n=== Example 4: Custom Headers Authentication ===")
        headers = {
            os.getenv("CUSTOM_HEADER_NAME", "Authorization"): os.getenv("CUSTOM_AUTH_HEADER")
        }
        client = AuthenticatedMCPClient(
            server_url=endpoint,
            headers=headers,
            transport_type=transport
        )
        try:
            await client.connect()
        except Exception as e:
            print(f"Custom header auth failed: {e}\n")
    
    # Example 5: No Authentication (default)
    print("\n=== Example 5: No Authentication ===")
    client = AuthenticatedMCPClient(
        server_url=endpoint,
        transport_type=transport
    )
    try:
        await client.connect()
    except Exception as e:
        print(f"No auth connection failed: {e}\n")


if __name__ == "__main__":
    # Example environment variables:
    # export MCP_SERVER_URL=http://localhost:8052
    # export MCP_TRANSPORT=streamable_http
    # export BEARER_TOKEN=your-bearer-token
    # export API_KEY=your-api-key
    # export API_KEY_HEADER=X-API-Key
    # export BASIC_USERNAME=user
    # export BASIC_PASSWORD=pass
    # export CUSTOM_AUTH_HEADER="Bearer custom-token"
    # export CUSTOM_HEADER_NAME=Authorization
    
    asyncio.run(main())