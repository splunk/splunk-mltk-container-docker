#!/usr/bin/env python3
"""Test the include_context parameter functionality."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_include_context():
    """Test the include_context parameter."""
    print("Testing include_context parameter functionality...\n")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(Path(__file__).parent.parent / "server.py")],
        env={"TRANSPORT": "stdio"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to server\n")
            
            test_query = "Show me all errors in the last hour"
            
            # Test 1: include_context=True
            print("Test 1: include_context=True")
            print("-" * 40)
            try:
                result = await session.call_tool("generate_spl_query", {
                    "query": test_query,
                    "include_context": True
                })
                
                # Extract content from CallToolResult
                content = result.content[0].text if result.content else {}
                if isinstance(content, str):
                    content = json.loads(content)
                
                print(f"SPL Query: {content.get('spl_query', 'N/A')}")
                print(f"Has system_prompt: {'system_prompt' in content}")
                print(f"Has llm_query: {'llm_query' in content}")
                
                if 'system_prompt' in content and 'llm_query' in content:
                    print("✓ Context fields included as expected")
                else:
                    print("✗ ERROR: Context fields missing")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
            
            print("\n")
            
            # Test 2: include_context=False
            print("Test 2: include_context=False")
            print("-" * 40)
            try:
                result = await session.call_tool("generate_spl_query", {
                    "query": test_query,
                    "include_context": False
                })
                
                # Extract content from CallToolResult
                content = result.content[0].text if result.content else {}
                if isinstance(content, str):
                    content = json.loads(content)
                
                print(f"SPL Query: {content.get('spl_query', 'N/A')}")
                print(f"Has system_prompt: {'system_prompt' in content}")
                print(f"Has llm_query: {'llm_query' in content}")
                
                if 'system_prompt' not in content and 'llm_query' not in content:
                    print("✓ Context fields excluded as expected")
                else:
                    print("✗ ERROR: Context fields should not be present")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")
            
            print("\n")
            
            # Test 3: include_context not specified (default should be False)
            print("Test 3: include_context not specified (default=False)")
            print("-" * 40)
            try:
                result = await session.call_tool("generate_spl_query", {
                    "query": test_query
                })
                
                # Extract content from CallToolResult
                content = result.content[0].text if result.content else {}
                if isinstance(content, str):
                    content = json.loads(content)
                
                print(f"SPL Query: {content.get('spl_query', 'N/A')}")
                print(f"Has system_prompt: {'system_prompt' in content}")
                print(f"Has llm_query: {'llm_query' in content}")
                
                if 'system_prompt' not in content and 'llm_query' not in content:
                    print("✓ Context fields excluded by default as expected")
                else:
                    print("✗ ERROR: Context fields should not be present by default")
                    
            except Exception as e:
                print(f"✗ Exception: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_include_context())
        print("\n✓ All tests completed")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)