#!/usr/bin/env python3
"""Interactive SPL validation test."""

import asyncio
import json
import os
import readline  # Enable line editing with arrow keys
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure readline for better interactive experience
readline.parse_and_bind('tab: complete')
readline.parse_and_bind('set editing-mode emacs')

# Set up history file
import atexit
history_file = os.path.expanduser('~/.splunk_validate_history')
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, history_file)

# Colors for output (minimal, matching Interactive SPL Search style)
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
RESET = '\033[0m'


async def test_validate_spl():
    """Interactive SPL validation test."""
    server_url = "http://localhost:8052/sse"
    
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize session
            await session.initialize()
            
            print("🛡️  SPL Query Validation Tool")
            print("=" * 50)
            print("Enter SPL queries to validate. Type /q or /x to exit.")
            print("")
            print("Features:")
            print("  - Risk assessment for SPL queries")
            print("  - Use arrow keys to navigate and edit queries")
            print("  - Use up/down arrows to access command history")
            print("  - History is saved between sessions")
            print("")
            print("Example risky queries:")
            print("  index=* | delete")
            print("  index=main | collect index=summary override=true")
            print("  | script python dangerous.py")
            print("=" * 50)
            
            try:
                while True:
                    # Read user input with support for ESC key
                    try:
                        # Get user input with readline support
                        query = input(f"\nSPL> ").strip()
                    except EOFError:
                        # Handle Ctrl+D
                        break
                    except KeyboardInterrupt:
                        # Handle Ctrl+C
                        break
                    
                    # Check for exit commands
                    if query.lower() in ['exit', '/q', '/x']:
                        break
                    
                    if not query.strip():
                        continue
                    
                    try:
                        # Call validate_spl tool
                        result = await session.call_tool(
                            "validate_spl",
                            arguments={"query": query}
                        )
                        
                        # Parse response (same as splunk_sse_search.py)
                        data = json.loads(result.content[0].text)
                        
                        # Display results
                        if "error" in data:
                            print(f"\n{RED}Error: {data['error']}{RESET}")
                            continue
                            
                        risk_score = data.get("risk_score", 0)
                        risk_message = data.get("risk_message", "")
                        
                        # Display results with appropriate emoji and color
                        if risk_score == 0:
                            print(f"\n✅ Query is SAFE (Risk Score: {risk_score}/100)")
                        elif risk_score <= 30:
                            print(f"\n⚠️  LOW RISK (Risk Score: {risk_score}/100)")
                        elif risk_score <= 60:
                            print(f"\n⚠️  MEDIUM RISK (Risk Score: {risk_score}/100)")
                        else:
                            print(f"\n❌ HIGH RISK (Risk Score: {risk_score}/100)")
                        
                        # Print risk message with proper formatting
                        if risk_message:
                            print(f"\n{risk_message}")
                        
                    except Exception as e:
                        print(f"\n{RED}Error validating query: {e}{RESET}")
                        
            except KeyboardInterrupt:
                pass
            
            print(f"\nGoodbye!\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_validate_spl())
    except KeyboardInterrupt:
        print(f"\n{GREEN}Test interrupted.{RESET}")