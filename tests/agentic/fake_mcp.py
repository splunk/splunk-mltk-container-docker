"""Fake Splunk MCP server (streamable HTTP) for tools/agentic-security-test.sh: one tool, logs whether each
request carries the configured bearer token (never prints the token). The query "broken" fails, "crash" ends the server."""
import os
import uvicorn
from mcp.server.fastmcp import FastMCP

EXPECTED = "Bearer " + os.environ["MI_MCP_TOKEN"]
mcp = FastMCP("fake-splunk", host="0.0.0.0", port=8001)
seen = {"auth_ok": None}


@mcp.tool()
def run_splunk_query(query: str) -> str:
    """Run a Splunk search (SPL) and return the results."""
    if query == "broken":  # as Splunk answers a bad SPL: an error result
        raise ValueError("Error in 'search' command: Unknown search command 'broken'.")
    if query == "crash":  # the server goes away during the call (a Splunk restart)
        os._exit(1)
    print(f"[fake-mcp] TOOL EXECUTED run_splunk_query query={query!r} auth_ok={seen['auth_ok']}", flush=True)
    return "index=main count=42"


inner = mcp.streamable_http_app()


async def app(scope, receive, send):
    if scope["type"] == "http":
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        seen["auth_ok"] = headers.get("authorization") == EXPECTED
        print(f"[fake-mcp] {scope['method']} {scope['path']} auth_ok={seen['auth_ok']}", flush=True)
    await inner(scope, receive, send)

uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
