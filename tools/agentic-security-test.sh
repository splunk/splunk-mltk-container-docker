#!/usr/bin/env bash
# Security and robustness checks of the agentic API (app.main_agentic) against fake services.
#
# Usage: tools/agentic-security-test.sh <agentic image>
#
# Runs a fake OpenAI-compatible LLM, a fake Splunk MCP server and a blackhole (accepts TCP, never
# answers) from tests/agentic/ in the image itself, on a test network of their own, then the agentic
# image in PROD mode in these scenarios (tests/agentic/client.py runs the checks inside it):
#   default      token not required: status shows no client configuration, a wrong token is
#                refused, chat with token runs the MCP tool once (with the configured MCP token),
#                only an exact "yes" from the LLM option that asked approves it, a failing tool call
#                is answered, /logReview drops a pending approval, one turn at a time per session
#                within DSDL_AGENTIC_TURN_TIMEOUT, names cannot escape /srv/logs, no CORS, one
#                start-up probe per LLM, bounded /logReview and /chatbox bodies
#   required     DSDL_AGENTIC_REQUIRE_TOKEN=true: no token -> 401 on all chat routes
#   unreachable  MCP and LLM endpoints never answer: the DSDL API is not blocked
#   probehang    MCP connected, the LLM's start-up probe hangs: the API and the status answer
#   noconfig     no llm_config: the API starts and the chat answers clearly
#   slowprobe    the probe answers late: tool calling stays on; MAX_SESSIONS=2 evicts the oldest
#                session with its pending approval and graph checkpoint
#   reconnect    the MCP server is down at start-up, comes up, then restarts (as Splunk does):
#                the chat gets the tools back each time; an approval from before is not resumed;
#                a server that stops answering (paused) fails two pings and is reopened once it
#                answers; a tool call during which the server dies is answered
# SIGTERM must shut the API down cleanly and quickly: with an open MCP session (which it closes),
# during the MCP connect, during the LLM probe, and while the start-up waits for its endpoints.
# Containers, volumes and the network are removed afterwards. AGENTIC_TEST_DOCKER_ARGS adds
# docker run options to the API container, split on spaces and without quotes, e.g.
# AGENTIC_TEST_DOCKER_ARGS="-v $PWD/app:/dltk/app:ro" to test a working tree. DSDL_TEST_PLATFORM
# selects the platform of a multi-platform local image, as in tools/smoke-test.sh.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
image="${1:?agentic image reference}"
platform="${DSDL_TEST_PLATFORM:-$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")}"
net=dsdl-agentic-test-net
token=agentic-test-api-token
mcp_token=agentic-test-mcp-token
failed=0
read -r -a extra_args <<< "${AGENTIC_TEST_DOCKER_ARGS:-}"
containers=(dsdl-agentic-fakellm dsdl-agentic-fakemcp dsdl-agentic-blackhole dsdl-agentic-api)

cleanup() {
  docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
  docker volume rm dsdl-agentic-api-srv >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup
docker network create "$net" >/dev/null

# llm_config as DSDL builds it: openai and azure_openai configured ("true", as DSDL's setup page
# writes it; both are the fake LLM), gemini set to "false", the others 0; MCP entry 1 enabled
llm_config() { # <llm base url> <mcp url>
  python3 - "$1" "$2" "$mcp_token" <<'EOF'
import json, sys
llm, mcp, token = sys.argv[1:]
off = lambda **k: [dict({"is_configured": 0, "model": None}, **k)]
print(json.dumps({
    "llm": {"ollama": off(base_url=None),
            "azure_openai": [{"is_configured": "true", "model": "fake-model", "deployment_name": "fake", "api_key": "sk-test",
                              "api_version": "2024-06-01", "azure_endpoint": llm.rsplit("/v1", 1)[0]}],
            "openai": [{"is_configured": "true", "model": "fake-model", "base_url": llm, "api_key": "sk-test"}],
            "bedrock": off(region_name=None), "gemini": [{"is_configured": "false", "model": "x", "api_key": "x"}]},
    "embedding_model": {}, "vector_db": {}, "graph_db": {},
    "mcp": [{"enabled": "1", "name": "splunk", "url": mcp, "token": token}]
           + [{"enabled": None, "name": None, "url": None, "token": None} for _ in range(4)]}))
EOF
}

service() { # <name> <script> [docker args...]: run a tests/agentic script; copied in, not bind-mounted
  local name="$1" script="$2"; shift 2
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker create --platform "$platform" --name "$name" --network "$net" "$@" \
    --entrypoint /opt/dltk/venv/bin/python "$image" -I "/tmp/$(basename "$script")" >/dev/null
  docker cp "$script" "$name:/tmp/" >/dev/null
  docker start "$name" >/dev/null
}

listening() { # <name> <port>...: wait until the service accepts connections
  local name="$1" port; shift
  for port in "$@"; do
    for _ in $(seq 1 60); do
      docker exec "$name" /opt/dltk/venv/bin/python -c \
        "import socket; socket.create_connection(('127.0.0.1', $port), 1)" >/dev/null 2>&1 && continue 2
      sleep 0.5
    done
    echo "FAIL $name does not listen on $port"; docker logs "$name" 2>&1 | tail -20; exit 1
  done
}

fake_mcp() {
  service dsdl-agentic-fakemcp tests/agentic/fake_mcp.py -e MI_MCP_TOKEN="$mcp_token"
  listening dsdl-agentic-fakemcp 8001
}

fakes() { # [fake LLM docker args...]
  service dsdl-agentic-fakellm tests/agentic/fake_llm.py "$@"
  listening dsdl-agentic-fakellm 8000
  fake_mcp
}

start_api() { # <docker env args...>
  docker rm -f dsdl-agentic-api >/dev/null 2>&1 || true
  docker volume rm dsdl-agentic-api-srv >/dev/null 2>&1 || true
  docker run -d --platform "$platform" --name dsdl-agentic-api --network "$net" -v dsdl-agentic-api-srv:/srv \
    -e MODE_DEV_PROD=PROD -e api_token="$token" ${extra_args[@]+"${extra_args[@]}"} "$@" "$image" >/dev/null
}

client() { # <phase>: run the checks of one phase inside the API container
  docker exec -i -e AGENTIC_TEST_TOKEN="$token" dsdl-agentic-api /opt/dltk/venv/bin/python - "$1" \
    < tests/agentic/client.py || { failed=1; docker logs dsdl-agentic-api 2>&1 | tail -20; }
}

scenario() { # <name> <docker env args...>
  local name="$1"; shift
  echo; echo "== $name"
  start_api "$@"
  client "$name"
}

api_log_has() { # <extended regex>: wait up to 30 s for a line in the API log
  for _ in $(seq 1 60); do
    grep -qE "$1" <<< "$(docker logs dsdl-agentic-api 2>&1)" && return 0
    sleep 0.5
  done
  return 1
}

# SIGTERM to the API container: the lifespan must cancel the start-up and close the MCP session
# (anyio refuses to exit the MCP client's task group from another task). A teardown error is
# logged as "session ... ended"; a hang ends in SIGKILL (exit 137) after the stop timeout.
clean_shutdown() { # <description>
  local started seconds code api_log
  started="$(date +%s)"
  docker stop -t 30 dsdl-agentic-api >/dev/null
  seconds=$(( $(date +%s) - started ))
  code="$(docker inspect -f '{{.State.ExitCode}}' dsdl-agentic-api)"
  api_log="$(docker logs dsdl-agentic-api 2>&1)"
  if [ "$seconds" -lt 20 ] && [ "$code" != 137 ] && grep -q "Application shutdown complete" <<< "$api_log" \
     && ! grep -qE "shutdown failed|Traceback|Task was destroyed|Splunk MCP session .* ended" <<< "$api_log"; then
    echo "PASS clean shutdown $1 (${seconds} s, exit $code)"
  else
    echo "FAIL shutdown $1 (${seconds} s, exit $code): $(grep -E 'Error|shutdown|Task was|ended' <<< "$api_log" \
      | tail -3 | tr '\n' ' ')"; failed=1
  fi
}

tool_runs() { # number of MCP tool runs with the configured MCP token, from the fake MCP log
  grep -c "TOOL EXECUTED.*auth_ok=True" <<< "$(docker logs dsdl-agentic-fakemcp 2>&1)" || true
}

service dsdl-agentic-blackhole tests/agentic/blackhole.py
listening dsdl-agentic-blackhole 8000 8001
fakes
config="$(llm_config http://dsdl-agentic-fakellm:8000/v1 http://dsdl-agentic-fakemcp:8001/mcp)"
scenario default -e "llm_config=$config" -e DSDL_AGENTIC_MAX_BODY_MB=1 -e DSDL_AGENTIC_TURN_TIMEOUT=8
runs="$(tool_runs)"
if [ "$runs" = 1 ]; then echo "PASS the MCP tool ran once (approved once, refused three times, stale approval dropped) with the configured MCP token"
else echo "FAIL the MCP tool ran $runs times with the configured MCP token (expected 1)"; failed=1; fi
deletes_before="$(grep -c "DELETE /mcp" <<< "$(docker logs dsdl-agentic-fakemcp 2>&1)" || true)"
clean_shutdown "with an open MCP session"
deletes_after="$(grep -c "DELETE /mcp" <<< "$(docker logs dsdl-agentic-fakemcp 2>&1)" || true)"
if [ "$deletes_after" -gt "$deletes_before" ]; then echo "PASS the shutdown closed the MCP session (DELETE /mcp)"
else echo "FAIL the shutdown did not close the MCP session (no DELETE /mcp after the stop)"; failed=1; fi

fakes
scenario required -e "llm_config=$config" -e DSDL_AGENTIC_REQUIRE_TOKEN=true

hanging="$(llm_config http://dsdl-agentic-blackhole:8000/v1 http://dsdl-agentic-blackhole:8001/mcp)"
scenario unreachable -e "llm_config=$hanging"
clean_shutdown "while the start-up still waits for the endpoints"

echo; echo "== shutdown during the MCP connect"
start_api -e "llm_config=$hanging"
api_log_has "connecting Splunk MCP" || echo "WARNING: no start-up line in the API log"
sleep 3
clean_shutdown "during the MCP connect"

scenario probehang -e "llm_config=$(llm_config http://dsdl-agentic-blackhole:8000/v1 http://dsdl-agentic-fakemcp:8001/mcp)"
api_log="$(docker logs dsdl-agentic-api 2>&1)"
if grep -q "Success: MCP Connection" <<< "$api_log" && ! grep -q "start-up done" <<< "$api_log"; then
  echo "PASS the start-up waits in the LLM probe (MCP connected, start-up not done)"
else echo "FAIL the start-up is not in the LLM probe"; failed=1; fi
clean_shutdown "during the LLM probe"

scenario noconfig

fakes -e FAKE_LLM_FIRST_DELAY=60
scenario slowprobe -e "llm_config=$config" -e DSDL_AGENTIC_PROBE_TIMEOUT=3 -e MAX_SESSIONS=2

echo; echo "== reconnect"
docker rm -f dsdl-agentic-fakemcp >/dev/null  # the MCP server is down when the container starts
start_api -e "llm_config=$config" -e DSDL_AGENTIC_MCP_RETRY=3 -e DSDL_AGENTIC_TOOL_TIMEOUT=5
client reconnect-down
fake_mcp
client reconnect-up
runs_before="$(tool_runs)"  # docker logs spans the restart
docker restart -t 2 dsdl-agentic-fakemcp >/dev/null  # as a Splunk restart: the open session is gone
listening dsdl-agentic-fakemcp 8001
client reconnect-again
runs_after="$(tool_runs)"
if [ "$runs_after" = $((runs_before + 1)) ]; then echo "PASS the MCP tool ran once on the restarted MCP server (not for the stale approval)"
else echo "FAIL the MCP tool ran $((runs_after - runs_before)) times on the restarted MCP server (expected 1)"; failed=1; fi
docker pause dsdl-agentic-fakemcp >/dev/null  # the server stops answering, its connections stay open
client reconnect-paused
if grep -q "Splunk MCP ping failed (2)" <<< "$(docker logs dsdl-agentic-api 2>&1)"; then
  echo "PASS two failed pings came before the reconnect"
else echo "FAIL no second failed ping in the API log"; failed=1; fi
docker unpause dsdl-agentic-fakemcp >/dev/null
client reconnect-resumed
client reconnect-crash

echo
if [ "$failed" = 0 ]; then echo "AGENTIC SECURITY TEST PASSED: $image"; else echo "AGENTIC SECURITY TEST FAILED: $image"; fi
exit "$failed"
