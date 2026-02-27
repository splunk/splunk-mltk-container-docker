#!/usr/bin/env python3
"""Test query generation functionality of the Custom Query Agent MCP server."""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Test queries
TEST_QUERIES = [
    "Show me all errors in the last hour",
    "Find failed login attempts from suspicious IP addresses",
    "What are the top 10 most active users today?",
    "Show me all 404 errors grouped by URL",
    "Find unusually high download volumes in the past week"
]

async def test_query_generation():
    """Test the generate_spl_query tool with various queries."""
    print("Testing Custom Query Agent query generation...")
    
    server_params = StdioServerParameters(
        command="python",
        args=[str(Path(__file__).parent.parent / "server.py")],
        env={"TRANSPORT": "stdio"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to server\n")
            
            # Test each query
            for i, query in enumerate(TEST_QUERIES, 1):
                print(f"\nTest {i}/{len(TEST_QUERIES)}: {query}")
                print("-" * 60)
                
                try:
                    # Generate SPL query
                    result = await session.call_tool("generate_spl_query", {
                        "query": query,
                        "include_context": False
                    })
                    
                    # Extract content from CallToolResult
                    content = result.content[0].text if result.content else {}
                    if isinstance(content, str):
                        content = json.loads(content)
                    
                    if "error" in content:
                        print(f"✗ Error: {content['error']}")
                        if "details" in content:
                            print(f"  Details: {content['details']}")
                    else:
                        print(f"✓ Generated SPL: {content.get('spl_query', 'N/A')}")
                        print(f"  Visualization: {content.get('viz_type', 'N/A')}")
                        print(f"  Tool used: {content.get('tool_used', 'N/A')}")
                        print(f"  Time range: {content.get('earliest', 'N/A')} to {content.get('latest', 'N/A')}")
                        
                except Exception as e:
                    print(f"✗ Exception: {e}")
            
            # Test with context included
            print("\n\nTesting with context included...")
            print("=" * 60)
            
            test_query = "Show me security alerts from the last 24 hours"
            result = await session.call_tool("generate_spl_query", {
                "query": test_query,
                "include_context": True,
                "q_samples": 5
            })
            
            # Extract content from CallToolResult
            content = result.content[0].text if result.content else {}
            if isinstance(content, str):
                content = json.loads(content)
            
            if "error" not in content:
                print(f"Query: {test_query}")
                print(f"Generated SPL: {content.get('spl_query', 'N/A')}")
                if "system_prompt" in content:
                    print(f"\n✓ System Prompt included (length: {len(content['system_prompt'])} chars)")
                    print(f"  Preview: {content['system_prompt'][:200]}...")
                if "llm_query" in content:
                    print(f"\n✓ LLM Query included (length: {len(content['llm_query'])} chars)")
                    print(f"  Preview: {content['llm_query'][:200]}...")
            
            # Test with context excluded (should not have context fields)
            print("\n\nTesting with context excluded...")
            print("=" * 60)
            
            result = await session.call_tool("generate_spl_query", {
                "query": test_query,
                "include_context": False
            })
            
            # Extract content from CallToolResult
            content = result.content[0].text if result.content else {}
            if isinstance(content, str):
                content = json.loads(content)
            
            if "error" not in content:
                print(f"Query: {test_query}")
                print(f"Generated SPL: {content.get('spl_query', 'N/A')}")
                if "system_prompt" in content:
                    print(f"\n✗ ERROR: System prompt should not be included when include_context=False")
                else:
                    print(f"\n✓ System prompt correctly excluded")
                if "llm_query" in content:
                    print(f"✗ ERROR: LLM query should not be included when include_context=False")
                else:
                    print(f"✓ LLM query correctly excluded")

if __name__ == "__main__":
    try:
        asyncio.run(test_query_generation())
        print("\n\n✓ All tests completed")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)