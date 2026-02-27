#!/usr/bin/env python3
"""List all capabilities of the Splunk MCP server."""

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()

# ANSI color codes
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

async def list_capabilities():
    server_url = "http://localhost:8052/sse"
    
    print(f"{BOLD}{CYAN}Splunk MCP Server Capabilities{RESET}")
    print("=" * 60)
    
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # 1. Server Information
                print(f"\n{BOLD}📋 Server Information{RESET}")
                print("-" * 40)
                try:
                    result = await session.call_tool("get_config", arguments={})
                    config = json.loads(result.content[0].text)
                    
                    print(f"Name: {config.get('name', 'Unknown')}")
                    print(f"Description: {config.get('description', 'Unknown')}")
                    print(f"Host: {config.get('host', 'Unknown')}:{config.get('port', 'Unknown')}")
                    print(f"Transport: {config.get('transport', 'Unknown')}")
                    print(f"Splunk Host: {config.get('splunk_host', 'Unknown')}:{config.get('splunk_port', 'Unknown')}")
                    print(f"Splunk Connected: {GREEN if config.get('splunk_connected') else YELLOW}{'Yes' if config.get('splunk_connected') else 'No'}{RESET}")
                    print(f"Max Events Count: {config.get('max_events_count', 'Unknown')}")
                    print(f"SSL Verification: {'Enabled' if config.get('verify_ssl') else 'Disabled'}")
                except Exception as e:
                    print(f"Error getting server info: {e}")
                
                # 2. Available Tools
                print(f"\n{BOLD}🔧 Available Tools{RESET}")
                print("-" * 40)
                try:
                    tools = await session.list_tools()
                    for tool in tools.tools:
                        print(f"\n{GREEN}▸ {tool.name}{RESET}")
                        print(f"  {tool.description}")
                        
                        if tool.inputSchema:
                            # Parse the schema to show parameters
                            schema = tool.inputSchema
                            if 'properties' in schema:
                                print(f"  {YELLOW}Parameters:{RESET}")
                                for param, details in schema['properties'].items():
                                    required = param in schema.get('required', [])
                                    param_type = details.get('type', 'unknown')
                                    description = details.get('description', '')
                                    default = details.get('default', None)
                                    
                                    req_marker = "*" if required else ""
                                    print(f"    - {param}{req_marker} ({param_type}): {description}")
                                    if default is not None:
                                        print(f"      Default: {default}")
                except Exception as e:
                    print(f"Error listing tools: {e}")
                
                # 3. Available Resources
                print(f"\n{BOLD}📚 Available Resources{RESET}")
                print("-" * 40)
                
                # We know the resources from the server implementation
                resources = [
                    ("splunk://indexes", "Detailed information about all Splunk indexes")
                ]
                
                for uri, description in resources:
                    print(f"\n{GREEN}▸ {uri}{RESET}")
                    print(f"  {description}")
                    
                    # Try to get a preview
                    try:
                        resource = await session.read_resource(uri)
                        content = resource.contents[0].text
                        lines = content.split('\n')
                        print(f"  {YELLOW}Preview:{RESET}")
                        for line in lines[:5]:  # Show first 5 lines
                            if line.strip():
                                print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")
                        if len(lines) > 5:
                            print(f"    ... ({len(lines)} total lines)")
                    except Exception as e:
                        print(f"  Error reading resource: {e}")
                
                print(f"\n{BOLD}✅ Capability listing complete!{RESET}")
                print(f"\n{YELLOW}💡 Tip: Run 'Code Samples' from the menu for detailed Python and curl examples{RESET}")
                
    except Exception as e:
        print(f"\n{YELLOW}Error connecting to server:{RESET} {str(e)}")
        print("Make sure the server is running on port 8052 with SSE transport")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(list_capabilities())