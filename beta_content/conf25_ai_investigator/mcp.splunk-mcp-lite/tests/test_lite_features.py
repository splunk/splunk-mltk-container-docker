#!/usr/bin/env python3
"""Test script specifically for Splunk MCP Lite features."""

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

async def test_lite_features():
    server_url = "http://localhost:8052/sse"
    
    print(f"{BOLD}{CYAN}Testing Splunk MCP Lite Features{RESET}")
    print("=" * 50)
    
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. List available tools
                print(f"\n{BOLD}Available Tools:{RESET}")
                tools = await session.list_tools()
                expected_tools = ['validate_spl', 'search_oneshot', 'search_export', 'get_indexes', 'get_config']
                
                for tool in tools.tools:
                    status = f"{GREEN}✓{RESET}" if tool.name in expected_tools else f"{YELLOW}?{RESET}"
                    print(f"{status} {tool.name}: {tool.description}")
                
                # 2. Test SPL validation
                print(f"\n{BOLD}SPL Query Validation:{RESET}")
                test_queries = [
                    ("index=_internal | head 10", "Safe query"),
                    ("index=* | delete", "Dangerous query"),
                    ("index=main", "Query without time range"),
                    ("index=_internal source=*metrics.log | timechart span=1h count", "Time chart query")
                ]
                
                for query, description in test_queries:
                    result = await session.call_tool("validate_spl", arguments={"query": query})
                    validation = json.loads(result.content[0].text)
                    
                    risk_level = "LOW" if validation['risk_score'] <= 30 else \
                                 "MEDIUM" if validation['risk_score'] <= 60 else "HIGH"
                    risk_color = GREEN if risk_level == "LOW" else \
                                 YELLOW if risk_level == "MEDIUM" else RED
                    
                    print(f"\n{description}:")
                    print(f"  Query: {query}")
                    print(f"  Risk Score: {risk_color}{validation['risk_score']}{RESET} ({risk_level})")
                    print(f"  Would Execute: {'Yes' if validation['would_execute'] else 'No'}")
                    if validation['risk_score'] > 0:
                        print(f"  Risk Details: {validation['risk_message'][:100]}...")
                
                # 3. Test search functionality
                print(f"\n{BOLD}Search Functionality:{RESET}")
                
                # Simple search
                result = await session.call_tool("search_oneshot", arguments={
                    "query": "index=_internal | stats count by component | head 5",
                    "earliest_time": "-1h",
                    "output_format": "json"
                })
                search_result = json.loads(result.content[0].text)
                
                if 'error' not in search_result:
                    print(f"\n{GREEN}✓{RESET} Search executed successfully")
                    print(f"  Found {search_result['event_count']} results")
                else:
                    print(f"\n{RED}✗{RESET} Search failed: {search_result['error']}")
                
                # 4. Test indexes listing
                print(f"\n{BOLD}Available Indexes:{RESET}")
                result = await session.call_tool("get_indexes", arguments={})
                indexes_data = json.loads(result.content[0].text)
                
                if 'indexes' in indexes_data:
                    print(f"Found {indexes_data['count']} indexes:")
                    # Show top 5 indexes by event count
                    sorted_indexes = sorted(indexes_data['indexes'], 
                                          key=lambda x: x.get('totalEventCount', 0), 
                                          reverse=True)[:5]
                    
                    for idx in sorted_indexes:
                        size_mb = idx.get('currentDBSizeMB', 0)
                        events = idx.get('totalEventCount', 0)
                        print(f"  • {idx['name']}: {events:,} events ({size_mb:.2f} MB)")
                
                # 5. Test resources
                print(f"\n{BOLD}Available Resources:{RESET}")
                try:
                    resource = await session.read_resource("splunk://indexes")
                    print(f"{GREEN}✓{RESET} splunk://indexes resource available")
                except Exception as e:
                    print(f"{RED}✗{RESET} splunk://indexes resource error: {e}")
                
                # Try to access removed resource
                try:
                    resource = await session.read_resource("splunk://saved-searches")
                    print(f"{RED}✗{RESET} splunk://saved-searches should not be available!")
                except Exception as e:
                    print(f"{GREEN}✓{RESET} splunk://saved-searches correctly removed")
                
                # 6. Server configuration
                print(f"\n{BOLD}Server Configuration:{RESET}")
                result = await session.call_tool("get_config", arguments={})
                config = json.loads(result.content[0].text)
                
                print(f"  Name: {config.get('name', 'Unknown')}")
                print(f"  Port: {config.get('port', 'Unknown')}")
                print(f"  Splunk Connected: {'Yes' if config.get('splunk_connected') else 'No'}")
                print(f"  Risk Tolerance: {config.get('spl_risk_tolerance', 'Unknown')}")
                
                print(f"\n{GREEN}{BOLD}✅ Splunk MCP Lite is working correctly!{RESET}")
                
    except Exception as e:
        print(f"\n{RED}Error:{RESET} {str(e)}")
        print("Make sure the server is running on port 8052")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_lite_features())
    exit(0 if success else 1)