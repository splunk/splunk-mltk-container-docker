#!/usr/bin/env python3
"""Comprehensive stdio transport tests for Splunk MCP server."""

import asyncio
import json
import os
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
print(f"Loading .env from: {env_path}")
print(f".env exists: {env_path.exists()}")
load_dotenv(env_path)

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
            print(f"\n{RED}Failed Tests:{RESET}")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        
        return self.failed == 0

async def test_stdio_transport():
    results = TestResults()
    
    print(f"{BOLD}{CYAN}Running STDIO Transport Tests{RESET}")
    print("=" * 50)
    
    # Get the path to the server module
    server_path = Path(__file__).parent.parent / "server.py"
    
    # Create server parameters for stdio transport
    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
        env={
            "MCP_SERVER_TRANSPORT": "stdio",
            "SPLUNK_HOST": os.getenv("SPLUNK_HOST", "localhost"),
            "SPLUNK_PORT": os.getenv("SPLUNK_PORT", "8089"),
            "SPLUNK_USERNAME": os.getenv("SPLUNK_USERNAME", "admin"),
            "SPLUNK_PASSWORD": os.getenv("SPLUNK_PASSWORD", "*********"),
            "SPLUNK_TOKEN": os.getenv("SPLUNK_TOKEN", ""),
            "VERIFY_SSL": os.getenv("VERIFY_SSL", "false"),
            "MAX_EVENTS_COUNT": os.getenv("MAX_EVENTS_COUNT", "100000"),
            "SERVER_NAME": os.getenv("SERVER_NAME", "Splunk MCP"),
            "SERVER_DESCRIPTION": os.getenv("SERVER_DESCRIPTION", "MCP server for retrieving data from Splunk"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "WARNING")
        }
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Test 1: Server Configuration
                print(f"\n{BOLD}Testing Server Configuration{RESET}")
                try:
                    result = await session.call_tool("get_config", arguments={})
                    config = json.loads(result.content[0].text)
                    if config.get('splunk_connected'):
                        results.add_pass("get_config - Server connected to Splunk")
                    else:
                        results.add_fail("get_config", "Server not connected to Splunk")
                except Exception as e:
                    results.add_fail("get_config", str(e))
                
                # Test 2: List Tools
                print(f"\n{BOLD}Testing Tool Discovery{RESET}")
                try:
                    tools = await session.list_tools()
                    expected_tools = ['validate_spl', 'search_oneshot', 'search_export', 'get_indexes', 
                                    'get_config']
                    found_tools = [tool.name for tool in tools.tools]
                    
                    for expected in expected_tools:
                        if expected in found_tools:
                            results.add_pass(f"Tool discovery - Found {expected}")
                        else:
                            results.add_fail(f"Tool discovery - {expected}", "Tool not found")
                except Exception as e:
                    results.add_fail("Tool discovery", str(e))
                
                # Test 3: Validate SPL
                print(f"\n{BOLD}Testing validate_spl{RESET}")
                test_queries = [
                    ("index=_internal | head 10", "Low risk query"),
                    ("index=* | delete", "High risk query"),
                    ("index=main", "Missing time range query")
                ]
                
                for query, description in test_queries:
                    try:
                        result = await session.call_tool("validate_spl", arguments={"query": query})
                        validation = json.loads(result.content[0].text)
                        if 'risk_score' in validation and 'risk_message' in validation:
                            results.add_pass(f"validate_spl - {description} (score: {validation['risk_score']})")
                        else:
                            results.add_fail(f"validate_spl - {description}", "Invalid response format")
                    except Exception as e:
                        results.add_fail(f"validate_spl - {description}", str(e))
                
                # Test 4: Get Indexes
                print(f"\n{BOLD}Testing get_indexes{RESET}")
                try:
                    result = await session.call_tool("get_indexes", arguments={})
                    indexes = json.loads(result.content[0].text)
                    if 'indexes' in indexes and indexes['count'] > 0:
                        results.add_pass(f"get_indexes - Found {indexes['count']} indexes")
                    else:
                        results.add_fail("get_indexes", "No indexes found")
                except Exception as e:
                    results.add_fail("get_indexes", str(e))
                
                # Test 5: Search with different output formats
                print(f"\n{BOLD}Testing search_oneshot with different formats{RESET}")
                formats = ['json', 'markdown', 'csv', 'summary']
                query = "index=_internal | head 3"
                
                for fmt in formats:
                    try:
                        result = await session.call_tool(
                            "search_oneshot",
                            arguments={
                                "query": query,
                                "earliest_time": "-1h",
                                "max_count": 3,
                                "output_format": fmt
                            }
                        )
                        search_result = json.loads(result.content[0].text)
                        if 'error' not in search_result:
                            results.add_pass(f"search_oneshot - {fmt} format")
                        else:
                            results.add_fail(f"search_oneshot - {fmt} format", search_result['error'])
                    except Exception as e:
                        results.add_fail(f"search_oneshot - {fmt} format", str(e))
                
                # Test 6: Export Search
                print(f"\n{BOLD}Testing search_export{RESET}")
                try:
                    result = await session.call_tool(
                        "search_export",
                        arguments={
                            "query": "index=_internal | stats count by sourcetype | head 5",
                            "earliest_time": "-15m",
                            "output_format": "json"
                        }
                    )
                    export_result = json.loads(result.content[0].text)
                    if 'error' not in export_result:
                        results.add_pass(f"search_export - Found {export_result['event_count']} results")
                    else:
                        results.add_fail("search_export", export_result['error'])
                except Exception as e:
                    results.add_fail("search_export", str(e))
                
                # Test 7: Resources
                print(f"\n{BOLD}Testing Resources{RESET}")
                resources = ["splunk://indexes"]
                
                for resource_uri in resources:
                    try:
                        resource = await session.read_resource(resource_uri)
                        if resource.contents and len(resource.contents[0].text) > 0:
                            results.add_pass(f"Resource - {resource_uri}")
                        else:
                            results.add_fail(f"Resource - {resource_uri}", "Empty content")
                    except Exception as e:
                        results.add_fail(f"Resource - {resource_uri}", str(e))
                
                # Test 8: Error Handling
                print(f"\n{BOLD}Testing Error Handling{RESET}")
                
                # Test invalid query
                try:
                    result = await session.call_tool(
                        "search_oneshot",
                        arguments={
                            "query": "invalid syntax [[[ bad query",
                            "earliest_time": "-1h"
                        }
                    )
                    search_result = json.loads(result.content[0].text)
                    if 'error' in search_result:
                        results.add_pass("Error handling - Invalid query detected")
                    else:
                        results.add_fail("Error handling - Invalid query", "Should have returned error")
                except Exception as e:
                    results.add_pass("Error handling - Invalid query raised exception")
                
                # Test non-existent resource
                try:
                    await session.read_resource("splunk://non-existent")
                    results.add_fail("Error handling - Non-existent resource", "Should have raised error")
                except Exception as e:
                    results.add_pass("Error handling - Non-existent resource properly rejected")
                
                # Test 9: Time range formats
                print(f"\n{BOLD}Testing Time Range Formats{RESET}")
                time_ranges = [
                    ("-24h", "now"),
                    ("-7d", "now"),
                    ("-1h@h", "now"),
                    ("0", "now")  # All time
                ]
                
                for earliest, latest in time_ranges:
                    try:
                        result = await session.call_tool(
                            "search_oneshot",
                            arguments={
                                "query": "index=_internal | head 1",
                                "earliest_time": earliest,
                                "latest_time": latest,
                                "max_count": 1
                            }
                        )
                        search_result = json.loads(result.content[0].text)
                        if 'error' not in search_result:
                            results.add_pass(f"Time range - {earliest} to {latest}")
                        else:
                            results.add_fail(f"Time range - {earliest} to {latest}", search_result['error'])
                    except Exception as e:
                        results.add_fail(f"Time range - {earliest} to {latest}", str(e))
    
    except Exception as e:
        print(f"\n{RED}Fatal error with stdio transport:{RESET} {str(e)}")
        print("Check that server.py can be run with stdio transport")
        return False
    
    # Print summary
    success = results.print_summary()
    
    if success:
        print(f"\n{GREEN}{BOLD}All STDIO transport tests passed!{RESET}")
    else:
        print(f"\n{RED}{BOLD}Some tests failed. Please check the errors above.{RESET}")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(test_stdio_transport())
    sys.exit(0 if success else 1)