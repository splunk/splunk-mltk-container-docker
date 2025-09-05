#!/usr/bin/env python3
"""Interactive Splunk search client using SSE transport."""

import asyncio
import json
import os
import readline  # Enable line editing with arrow keys
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()

# Configure readline for better interactive experience
readline.parse_and_bind('tab: complete')
readline.parse_and_bind('set editing-mode emacs')  # or 'vi' if you prefer

# Optional: Set up history file
import atexit
history_file = os.path.expanduser('~/.splunk_search_history')
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, history_file)

async def interactive_search():
    server_url = "http://localhost:8052/sse"
    
    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("🔍 Splunk Interactive Search Client")
            print("=" * 50)
            print("Enter SPL queries to execute. Type /q or /x to exit.")
            print("Default: All time, JSON format")
            print("")
            print("Commands:")
            print("  /format <type>  - Set output format (json, markdown/md, csv, summary)")
            print("  /f <type>       - Shorthand for /format")
            print("  /q or /x       - Exit")
            print("")
            print("Features:")
            print("  - Use arrow keys to navigate and edit queries")
            print("  - Use up/down arrows to access command history")
            print("  - History is saved between sessions")
            print("")
            print("Example: index=_internal | head 10")
            print("=" * 50)
            
            # Default output format
            output_format = "json"
            
            while True:
                try:
                    # Get user input
                    query = input(f"\nSPL [{output_format}]> ").strip()
                    
                    # Check for exit commands
                    if query.lower() in ['/q', '/x', 'exit', 'quit']:
                        print("👋 Goodbye!")
                        break
                    
                    # Check for format command (also accept /f as shorthand)
                    if query.lower().startswith('/format') or query.lower() == '/f' or query.lower().startswith('/f '):
                        parts = query.split(None, 1)
                        if len(parts) == 2:
                            new_format = parts[1].lower()
                            # Handle md synonym
                            if new_format == 'md':
                                new_format = 'markdown'
                            
                            if new_format in ['json', 'markdown', 'csv', 'summary']:
                                output_format = new_format
                                print(f"✅ Output format changed to: {output_format}")
                            else:
                                print(f"❌ Invalid format. Choose from: json, markdown (or md), csv, summary")
                        else:
                            print(f"Current format: {output_format}")
                            print("Usage: /format <json|markdown|md|csv|summary>")
                            print("       /f <json|markdown|md|csv|summary>")
                        continue
                    
                    # Skip empty queries
                    if not query:
                        continue
                    
                    # Set defaults - ALL time
                    earliest = "0"  # All time
                    latest = "now"
                    
                    # Display the REST API request that will be sent
                    splunk_host = os.getenv("SPLUNK_HOST", "localhost")
                    splunk_port = os.getenv("SPLUNK_PORT", "8089")
                    # Determine actual search parameter
                    if query.strip().startswith("|"):
                        search_param = query
                    else:
                        search_param = f"search {query}"
                    
                    print(f"\n📡 REST API Request:")
                    print(f"   URL: https://{splunk_host}:{splunk_port}/services/search/jobs/oneshot")
                    print(f"   Method: POST")
                    print(f"   Body Parameters:")
                    print(f"     - search: {search_param}")
                    print(f"     - earliest_time: {earliest}")
                    print(f"     - latest_time: {latest}")
                    print(f"     - output_mode: json")
                    
                    # Execute search
                    print(f"\n⏳ Executing search with {output_format} format...")
                    result = await session.call_tool(
                        "search_oneshot",
                        arguments={
                            "query": query,
                            "earliest_time": earliest,
                            "latest_time": latest,
                            "output_format": output_format
                        }
                    )
                    
                    # Parse and display results
                    search_result = json.loads(result.content[0].text)
                    
                    if "error" in search_result:
                        print(f"❌ Error: {search_result['error']}")
                        # Display risk message if available
                        if "risk_message" in search_result:
                            print(f"\n{search_result['risk_message']}")
                    else:
                        print(f"\n✅ Found {search_result['event_count']} events")
                        
                        if output_format == "json":
                            # Original JSON display
                            if search_result['event_count'] > 0:
                                print("\nResults:")
                                print("-" * 80)
                                
                                for i, event in enumerate(search_result['events'], 1):
                                    print(f"\n--- Event {i} ---")
                                    
                                    # Display _raw if available
                                    if '_raw' in event:
                                        print(f"Raw: {event['_raw']}")
                                    
                                    # Display all fields (including underscore fields)
                                    for key, value in event.items():
                                        if key != '_raw':  # Skip _raw since we already displayed it
                                            print(f"{key}: {value}")
                                    
                                    if i < len(search_result['events']):
                                        print()  # Add spacing between events
                                
                                print("-" * 80)
                            else:
                                print("No events found for this search.")
                        else:
                            # Display formatted content
                            if 'content' in search_result:
                                print("\nFormatted Results:")
                                print("-" * 80)
                                print(search_result['content'])
                                print("-" * 80)
                            else:
                                print("No formatted content available.")
                
                except KeyboardInterrupt:
                    print("\n\n👋 Search interrupted. Type /q to exit.")
                except Exception as e:
                    print(f"❌ Error executing search: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(interactive_search())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")