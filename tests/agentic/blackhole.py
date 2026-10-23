"""Endpoints that accept TCP connections on 8000 and 8001 and never answer, for
tools/agentic-security-test.sh: an MCP server or LLM that hangs, independent of the host's routing."""
import socket
import threading
import time

held = []


def serve(port):
    server = socket.create_server(("0.0.0.0", port), backlog=128)
    while True:
        connection, _ = server.accept()
        held.append(connection)  # keep it open, send nothing


for port in (8000, 8001):
    threading.Thread(target=serve, args=(port,), daemon=True).start()
print("[blackhole] listening on 8000 and 8001", flush=True)
while True:
    time.sleep(3600)
