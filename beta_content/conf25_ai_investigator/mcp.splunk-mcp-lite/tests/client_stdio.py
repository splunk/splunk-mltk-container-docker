#!/usr/bin/env python3
"""Example stdio client for Splunk MCP server."""

import asyncio
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Add parent directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

async def main():
    # Get the path to the server module
    server_path = Path(__file__).parent.parent / "server.py"
    
    # Create server parameters for stdio transport
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
        env={
            "TRANSPORT": "stdio",
            "SPLUNK_HOST": os.getenv("SPLUNK_HOST", "localhost"),
            "SPLUNK_PORT": os.getenv("SPLUNK_PORT", "8089"),
            "SPLUNK_USERNAME": os.getenv("SPLUNK_USERNAME", "admin"),
            "SPLUNK_PASSWORD": os.getenv("SPLUNK_PASSWORD", "changeme"),
            "VERIFY_SSL": os.getenv("VERIFY_SSL", "false")
        }
    )
    
    # Create stdio transport and session
    async with stdio_client(server_params) as (read, write):
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
            
            # Example 3: Run a simple search with export
            print("\n3. Running an export search...")
            result = await session.call_tool(
                "search_export",
                arguments={
                    "query": "index=_internal | head 3",
                    "earliest_time": "-15m",
                    "max_count": 3
                }
            )
            print(f"Export results: {result.content[0].text}")
            
            # Example 4: Get statistics by source
            print("\n4. Getting statistics by source...")
            result = await session.call_tool(
                "search_oneshot",
                arguments={
                    "query": "index=_internal earliest=-1h | stats count by source | head 5",
                    "earliest_time": "-1h",
                    "max_count": 5
                }
            )
            print(f"Stats: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(main())