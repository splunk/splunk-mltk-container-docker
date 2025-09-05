#!/usr/bin/env python3
"""Test connection to the Custom Query Agent MCP server."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_connection():
    """Test basic connection and get_config tool."""
    print("Testing connection to Custom Query Agent MCP server...")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(Path(__file__).parent.parent / "server.py")],
        env={"MCP_SERVER_TRANSPORT": "stdio"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            print("✓ Connected to server")
            
            # Initialize the session
            await session.initialize()
            print("✓ Session initialized")
            
            # List available tools
            try:
                tools_response = await session.list_tools()
                tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                print(f"\nAvailable tools: {len(tools)}")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
            except Exception as e:
                print(f"\n⚠ Could not list tools: {e}")
                # Continue with the test anyway
            
            # Test get_config tool
            print("\nTesting get_config tool...")
            try:
                result = await session.call_tool("get_config", arguments={})
                # Extract the actual content from the result
                if hasattr(result, 'content') and result.content:
                    config_data = json.loads(result.content[0].text)
                else:
                    config_data = result
                
                print("\nServer Configuration:")
                print(json.dumps(config_data, indent=2))
            
                # Check connection to SPL Vectors Agent
                if config_data.get("agent_connected"):
                    print("\n✓ SPL Vectors Agent is connected")
                else:
                    print("\n✗ SPL Vectors Agent is NOT connected")
                    print(f"  Make sure it's running at: {config_data.get('spl_vectors_agent')}")
            except Exception as e:
                print(f"\n✗ Error calling get_config: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_connection())
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)