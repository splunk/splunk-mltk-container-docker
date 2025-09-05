#!/usr/bin/env python3
"""Example SSE client for Splunk MCP server."""

import asyncio
import os
from pathlib import Path
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

async def main():
    # Server URL from .env configuration
    port = os.getenv("MCP_SERVER_PORT", "8050")
    server_url = f"http://localhost:{port}/sse"
    
    # Create SSE transport
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("\nAvailable tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Example 1: Get server configuration
            print("\n1. Getting server configuration...")
            result = await session.call_tool("get_config", arguments={})
            print(f"Config: {result.content[0].text}")
            
            # Example 2: List Splunk indexes
            print("\n2. Listing Splunk indexes...")
            result = await session.call_tool("get_indexes", arguments={})
            print(f"Indexes: {result.content[0].text}")
            
            # Example 3: Run a simple search
            print("\n3. Running a simple search...")
            result = await session.call_tool(
                "search_oneshot",
                arguments={
                    "query": "index=_internal | head 5",
                    "earliest_time": "-1h",
                    "max_count": 5
                }
            )
            print(f"Search results: {result.content[0].text}")
            
            # Example 4: List saved searches
            print("\n4. Listing saved searches...")
            result = await session.call_tool("get_saved_searches", arguments={})
            print(f"Saved searches: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())