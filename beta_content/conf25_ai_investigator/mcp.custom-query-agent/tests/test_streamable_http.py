#!/usr/bin/env python3
"""Streamable HTTP transport test for Custom Query Agent MCP server."""

import asyncio
import json
import sys
import os
from pathlib import Path
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv("../.env")

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"{GREEN}✓{RESET} {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"{RED}✗{RESET} {test_name}: {error}")
    
    def print_summary(self):
        total = self.passed + self.failed
        print(f"\n{BOLD}Test Summary:{RESET}")
        print(f"  Total tests: {total}")
        print(f"  {GREEN}Passed: {self.passed}{RESET}")
        print(f"  {RED}Failed: {self.failed}{RESET}")
        
        if self.errors:
            print(f"\n{BOLD}Failed Tests:{RESET}")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        
        return self.failed == 0

async def test_streamable_http_transport():
    # Get configuration
    use_remote = os.getenv("MCP_TEST_URL") is not None
    
    if use_remote:
        # Use external URL for remote testing
        server_url = os.getenv("MCP_TEST_URL")
    else:
        # Use local port for local testing
        port = os.getenv("MCP_SERVER_STREAMABLE_HTTP_PORT", "7007")
        server_url = f"http://localhost:{port}/mcp/"
    
    results = TestResults()
    
    print(f"{BOLD}{CYAN}Running Streamable HTTP Transport Tests{RESET}")
    print(f"Server URL: {server_url}")
    print("=" * 50)
    
    try:
        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test 1: Server Configuration
                print(f"\n{BOLD}Testing Server Configuration{RESET}")
                try:
                    result = await session.call_tool("get_config", arguments={})
                    config = json.loads(result.content[0].text)
                    
                    # Check required fields
                    required_fields = ['name', 'description', 'transport', 'spl_vectors_agent', 'agent_connected']
                    for field in required_fields:
                        if field in config:
                            results.add_pass(f"get_config - Has {field}")
                        else:
                            results.add_fail(f"get_config - Missing {field}", "Field not found")
                    
                    # Check SPL Agent connection
                    if config.get('agent_connected'):
                        results.add_pass("get_config - SPL Vectors Agent connected")
                    else:
                        results.add_fail("get_config", "SPL Vectors Agent not connected")
                        
                except Exception as e:
                    results.add_fail("get_config", str(e))
                
                # Test 2: List Tools
                print(f"\n{BOLD}Testing Tool Discovery{RESET}")
                try:
                    tools = await session.list_tools()
                    expected_tools = ['generate_spl_query', 'get_config']
                    found_tools = [tool.name for tool in tools.tools]
                    
                    for expected in expected_tools:
                        if expected in found_tools:
                            results.add_pass(f"Tool discovery - Found {expected}")
                        else:
                            results.add_fail(f"Tool discovery - {expected}", "Tool not found")
                except Exception as e:
                    results.add_fail("Tool discovery", str(e))
                
                # Note: SPL query generation tests should only be run manually
                print(f"\n{BOLD}Note:{RESET} SPL query generation tests are available but must be run manually")
                print("  Use option 5 (Test Query Generation) in the menu for manual testing")
                
    except Exception as e:
        print(f"\n{RED}Failed to connect to Streamable HTTP server:{RESET} {e}")
        results.add_fail("Streamable HTTP Connection", str(e))
        return False
    
    # Print summary
    return results.print_summary()

if __name__ == "__main__":
    try:
        success = asyncio.run(test_streamable_http_transport())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Test failed with error:{RESET} {e}")
        sys.exit(1)