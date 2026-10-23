"""Checks of the agentic API, run inside the API container by tools/agentic-security-test.sh.

Usage: python - <phase> < client.py   (AGENTIC_TEST_TOKEN holds the container's api_token)
Phases: default (fake LLM and MCP reachable), required (DSDL_AGENTIC_REQUIRE_TOKEN=true),
unreachable (MCP and LLM endpoints never answer), probehang (MCP up, the LLM probe hangs), noconfig
(no llm_config at all), slowprobe (the fake LLM answers the start-up probe late; MAX_SESSIONS=2),
reconnect-down / reconnect-up / reconnect-again (MCP down at start-up, then up, then restarted).
Prints PASS/FAIL lines; exits 1 on any FAIL.
"""
import glob
import json
import threading
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

BASE = "https://127.0.0.1:5000"
FAKE_LLM = "http://dsdl-agentic-fakellm:8000"
TOKEN = os.environ["AGENTIC_TEST_TOKEN"]
CTX = ssl._create_unverified_context()
failures = 0


def check(ok, text):
    global failures
    failures += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + text, flush=True)


def call(method, path, body=None, token=None, headers=None, timeout=120, chunked=False):
    data = json.dumps(body).encode() if body is not None else None
    if chunked and data is not None:  # an iterable body: urllib sends it chunked, without Content-Length
        data = iter([data[i:i + 65536] for i in range(0, len(data), 65536)])
    h = {"Content-Type": "application/json"}
    if token is not None:
        h["Authorization"] = token
    h.update(headers or {})
    request = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, context=CTX, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    except Exception as error:  # timeout, connection refused: report it as status 0, do not abort
        return 0, f"{type(error).__name__}: {error}", {}, time.monotonic() - started
    text = response.read().decode(errors="replace")
    return response.status, text, dict(response.headers), time.monotonic() - started


def wait_for_api():
    started = time.monotonic()
    for _ in range(600):
        try:
            urllib.request.urlopen(BASE + "/", context=CTX, timeout=3)
            break
        except urllib.error.HTTPError:
            break
        except Exception:
            time.sleep(0.25)
    return time.monotonic() - started


def status_when_started(token=None, limit=90):
    deadline = time.monotonic() + limit
    while True:
        code, text, _, _ = call("GET", "/mcp/status", token=token, timeout=20)
        data = json.loads(text) if code == 200 else {}
        if code != 200 or "start-up" not in str(data.get("detail")) or time.monotonic() > deadline:
            return code, data
        time.sleep(0.5)


def wait_for(predicate, limit):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.5)
    return False


def fake_llm(key="calls"):
    return json.load(urllib.request.urlopen(FAKE_LLM + "/calls", timeout=10))[key]


def chat(session, message, user="u", llm="openai"):
    body = {"userName": user, "sessionID": session, "message": message, "llmOption": llm}
    code, text, _, _ = call("POST", "/chatbox", body, token=TOKEN)
    return code, text


def startup_log_count(text):
    return sum(open(path).read().count(text) for path in glob.glob("/srv/logs/Chatbox_Container/*.log"))


def connected():
    code, text, _, _ = call("GET", "/mcp/status", timeout=10)
    data = json.loads(text) if code == 200 else {}
    return data.get("connected") is True and data.get("detail") == "1 tools"


def tool_round_trip(session, label):
    code, text = chat(session, "list the indexes")
    code2, text2 = chat(session, "yes")
    check(code == 200 and "run_splunk_query" in text and code2 == 200 and "Summary" in text2,
          f"{label}: the chat offers the MCP tool and runs it ({text[:40]!r} / {text2[:40]!r})")


routes = (("GET", "/mcp/status", None),
          ("POST", "/chatbox", {"userName": "u", "sessionID": "s", "message": "hi", "llmOption": "openai"}),
          ("POST", "/logReview", {"userName": "u", "sessionID": "s", "logs": []}))
phase = sys.argv[1]
elapsed = wait_for_api()
if not phase.startswith("reconnect-") or phase == "reconnect-down":
    check(elapsed < 20, f"API answers {elapsed:.1f} s after the container start (limit 20 s)")

if phase == "default":
    code, data = status_when_started()
    check(code == 200 and data.get("connected") is True and data.get("detail") == "1 tools",
          f"/mcp/status without token (token not required): {code} connected={data.get('connected')} "
          f"detail={data.get('detail')!r}")
    extra = set(data) - {"connected", "detail", "llm_list"}
    leaked = {k for entry in (data.get("llm_list") or {}).values() for k in entry} - {"Able_to_hold_tools", "available"}
    check(not extra and not leaked, f"/mcp/status exposes no client configuration (extra keys: {sorted(extra | leaked)})")
    check((data.get("llm_list") or {}).get("openai", {}).get("Able_to_hold_tools") is True,
          "the configured LLM (is_configured 'true') is reported as able to call tools")
    available = {name: entry.get("available") for name, entry in (data.get("llm_list") or {}).items()}
    check(available == {"ollama": False, "bedrock": False, "azure_openai": True, "openai": True, "gemini": False},
          f"only the configured LLMs are available; is_configured 'false' and 0 mean not configured ({available})")
    calls = fake_llm()
    check(calls == 2, f"start-up sent one tool-calling probe per configured LLM before any chat ({calls})")
    for method, path, body in routes:
        code, text, _, _ = call(method, path, body, token="wrong-token")
        check(code == 401, f"{method} {path} with a wrong token: {code}")
    turn = {"userName": "../../tmp/agentic-traversal", "sessionID": "../../../tmp/agentic-session",
            "message": "list the indexes", "llmOption": '"openai"'}
    code, text, _, _ = call("POST", "/chatbox", turn, token=TOKEN)
    check(code == 200 and "run_splunk_query" in text, f"chat turn 1 with token asks to approve the MCP tool: {code}")
    turn["message"] = "yes"
    code, text, _, _ = call("POST", "/chatbox", turn, token=TOKEN)
    check(code == 200 and "Summary" in text, f"chat turn 2 'yes' runs the tool and answers: {code}")
    escaped = glob.glob("/tmp/agentic-traversal*") + glob.glob("/tmp/agentic-session*")
    check(not escaped, f"user and session names cannot leave /srv/logs ({escaped})")
    check(any(name.startswith("_") for name in os.listdir("/srv/logs")),
          f"the unsafe user name was sanitised ({sorted(os.listdir('/srv/logs'))})")
    code, _, headers, _ = call("GET", "/mcp/status", token=TOKEN, headers={"Origin": "https://evil.example"})
    allow = {k: v for k, v in headers.items() if k.lower().startswith("access-control-allow")}
    check(code == 200 and not allow, f"no CORS headers for a foreign origin ({allow})")
    code, _, _, _ = call("POST", "/logReview", {"userName": "u", "sessionID": "s", "logs": [{"_raw": "x"}]}, token=TOKEN)
    check(code == 200, f"/logReview with token: {code}")
    # malformed events first: only the first MAX_LOG_ENTRIES events are read
    flood = [{"no_raw": 1}, "not a dict", None] + [{"_raw": ""}] * 50000
    code, _, _, seconds = call("POST", "/logReview", {"userName": "u", "sessionID": "flood", "logs": flood}, token=TOKEN)
    check(code == 200 and seconds < 5, f"/logReview with 50000 empty and malformed events stays bounded ({code}, {seconds:.1f} s)")
    flood_logs = "".join(open(path).read() for path in glob.glob("/srv/logs/u/flood_*.log"))
    check("Logs: 5000 events" in flood_logs, "/logReview reads at most 5000 events (MAX_LOG_ENTRIES)")
    big = {"userName": "u", "sessionID": "big", "logs": [{"_raw": "x" * 1000}] * 2000}  # 2 MB, DSDL_AGENTIC_MAX_BODY_MB=1
    for chunked in (False, True):
        code, text, _, _ = call("POST", "/logReview", big, token=TOKEN, chunked=chunked)
        check(code == 200 and "too large" in text,
              f"/logReview over the body limit ({'chunked' if chunked else 'Content-Length'}) answers clearly: {text[:60]!r}")
    code, text, _, _ = call("POST", "/chatbox", {"userName": "u", "sessionID": "big", "message": "x" * 300_000,
                                                 "llmOption": "openai"}, token=TOKEN, chunked=True)
    check(code == 200 and "too long" in text, f"/chatbox over its limit (chunked) answers clearly: {text[:60]!r}")
    for session, reply in (("approval-1", "what about yesterday?"), ("approval-2", "yes, but only for index=main"),
                           ("approval-3", "y")):
        code, text = chat(session, "list the indexes")
        code2, text2 = chat(session, reply)
        check(code == 200 and "Do you approve" in text and "will not run the tools" in text2,
              f"the reply {reply!r} does not approve the tool call: {text2[:60]!r}")
    # a tool call that fails (as a bad SPL does) is explained, and the next turn of the session works
    code, text = chat("broken-1", "run the broken search")
    code2, text2 = chat("broken-1", "yes")
    code3, text3 = chat("broken-1", "and now?")
    check(code == 200 and "Do you approve" in text and code2 == 200 and "Summary" in text2 and code3 == 200
          and "Error" not in text3 and "failed" not in text3,
          f"a failing MCP tool call is answered and the session goes on ({text2[:40]!r} / {text3[:40]!r})")
    # an approval is answered only through the LLM option that asked for it; a dropped one leaves
    # no checkpoint behind (the fake LLM counts the messages it gets)
    chat("switch", "list the indexes", llm="azure_openai")     # azure asks to run its call
    code, text = chat("switch", "yes")                         # openai: a new turn, not azure's call
    messages = fake_llm("last_messages")
    check(code == 200 and "Do you approve" in text and messages == 1,
          f"a 'yes' after switching the LLM option does not run the other option's pending call ({messages} messages)")
    code, text = chat("switch", "and now?", llm="azure_openai")  # drops openai's question, azure starts afresh
    messages = fake_llm("last_messages")
    check(code == 200 and messages == 1, f"a dropped approval leaves no checkpoint behind ({messages} messages)")
    # a pending tool approval does not take the "yes" meant for the log summary
    chat("logs-1", "list the indexes")
    call("POST", "/logReview", {"userName": "u", "sessionID": "logs-1", "logs": [{"_raw": "error 42"}]}, token=TOKEN)
    code, text = chat("logs-1", "yes")
    messages = fake_llm("last_messages")  # the logs, the summary question and "yes"
    check(code == 200 and "Do you approve" in text and messages == 3,
          f"/logReview drops a pending tool approval and keeps the logs ({messages} messages)")
    # a turn that outlasts DSDL_AGENTIC_TURN_TIMEOUT ends with a short answer; meanwhile the session is busy
    slow = {}
    worker = threading.Thread(target=lambda: slow.update(zip(("code", "text"), chat("slow-turn", "answer slowly"))))
    worker.start()
    time.sleep(1.5)
    code, text = chat("slow-turn", "another question")
    code2, text2, _, _ = call("POST", "/logReview", {"userName": "u", "sessionID": "slow-turn", "logs": [{"_raw": "x"}]},
                              token=TOKEN)
    check("Still answering" in text and "Still answering" in text2,
          f"a session answers one message at a time ({text[:40]!r} / {text2[:40]!r})")
    worker.join(60)
    check(slow.get("code") == 200 and "took longer" in str(slow.get("text")),
          f"a turn over its time limit ends with a short answer ({str(slow.get('text'))[:60]!r})")
    code, _, _, _ = call("GET", "/summary", token=TOKEN)
    check(code == 200, f"the DSDL API (/summary) answers in the agentic image: {code}")

elif phase == "required":
    for method, path, body in routes:
        code, text, _, _ = call(method, path, body)
        check(code == 401, f"{method} {path} without token (token required): {code}")
    code, data = status_when_started(token=TOKEN)
    check(code == 200 and data.get("connected") is True, f"/mcp/status with token: {code} {data.get('detail')!r}")

elif phase == "unreachable":
    # the start-up waits for endpoints that never answer; the DSDL API must not wait with it
    code, _, _, seconds = call("GET", "/summary", token=TOKEN, timeout=30)
    check(code == 200 and seconds < 3, f"/summary answers during the agentic start-up ({code}, {seconds:.1f} s)")
    code, text, _, seconds = call("GET", "/mcp/status", timeout=30)
    data = json.loads(text) if code == 200 else {}
    check(code == 200 and seconds < 2 and data.get("connected") is False and "start-up" in str(data.get("detail")),
          f"/mcp/status answers at once while MCP does not answer ({code}, {seconds:.1f} s, {data.get('detail')!r})")
    code, text, _, seconds = call("POST", "/chatbox", routes[1][2], token=TOKEN, timeout=90)
    check(code == 200 and 15 < seconds < 30 and "still starting" in text,
          f"/chatbox during the start-up answers after about 20 s, within the DSDL proxy's 60 s "
          f"({code}, {seconds:.1f} s): {text[:80]}")
    code, _, _, seconds = call("GET", "/summary", token=TOKEN, timeout=30)
    check(code == 200 and seconds < 3, f"/summary still answers ({code}, {seconds:.1f} s)")

elif phase == "probehang":
    # MCP connects at once, then the start-up probe waits for an LLM that never answers
    time.sleep(3)
    code, _, _, seconds = call("GET", "/summary", token=TOKEN, timeout=30)
    check(code == 200 and seconds < 3, f"/summary answers during the LLM probe ({code}, {seconds:.1f} s)")
    code, text, _, seconds = call("GET", "/mcp/status", timeout=30)
    data = json.loads(text) if code == 200 else {}
    check(code == 200 and seconds < 2 and "start-up in progress" in str(data.get("detail")),
          f"/mcp/status answers at once during the LLM probe ({code}, {seconds:.1f} s, {data.get('detail')!r})")

elif phase == "noconfig":
    code, data = status_when_started()
    check(code == 200 and data.get("connected") is False, f"/mcp/status without llm_config: {code} {data}")
    code, text, _, _ = call("POST", "/chatbox", routes[1][2], token=TOKEN)
    check(code == 200 and "not available" in text, f"chat without any LLM answers clearly: {code} {text[:100]}")

elif phase == "slowprobe":
    # the probe answers after 60 s, the probe timeout is 3 s: the start-up ends at once and the tools stay
    started = time.monotonic()
    code, data = status_when_started()
    seconds = time.monotonic() - started
    kept = startup_log_count("no answer to the tool-calling check within 3 s; keeping the tools")
    check(code == 200 and seconds < 20 and kept == 1
          and all(entry.get("Able_to_hold_tools") for name, entry in (data.get("llm_list") or {}).items()
                  if name in ("openai", "azure_openai")),
          f"a probe without answer ends after its timeout and keeps tool calling ({seconds:.1f} s, {kept} timed out)")
    code, text = chat("slow-1", "list the indexes")
    check(code == 200 and "run_splunk_query" in text, f"chat still offers the MCP tool: {code} {text[:60]!r}")
    # MAX_SESSIONS=2: a third session evicts the oldest, with its pending approval and graph checkpoint
    chat("slow-2", "list the indexes")
    chat("slow-3", "list the indexes")
    code, text = chat("slow-1", "yes")
    messages = fake_llm("last_messages")
    check(code == 200 and "Do you approve" in text and messages == 1,
          f"an evicted session starts over without its checkpoint ({messages} messages sent to the LLM)")

elif phase == "reconnect-down":
    code, data = status_when_started()
    check(code == 200 and data.get("connected") is False and data.get("detail") == "0 tools",
          f"MCP down at start-up: the start-up finishes without tools ({data.get('detail')!r})")
    code, text = chat("r-0", "list the indexes")
    check(code == 200 and "run_splunk_query" not in text, f"the chat answers without tools: {text[:60]!r}")

elif phase == "reconnect-up":
    check(wait_for(connected, 30), "the MCP server came up: the supervisor connects within 30 s")
    tool_round_trip("r-1", "after the first connect")
    code, text = chat("r-pending", "list the indexes")  # left pending across the next reconnect
    check(code == 200 and "Do you approve" in text, "a tool approval is pending before the MCP restart")

elif phase == "reconnect-paused":
    # the server stops answering (paused): two failed pings, then /mcp/status reports it
    check(wait_for(lambda: not connected(), 45), "an MCP server that stops answering is reported disconnected")

elif phase == "reconnect-resumed":
    check(wait_for(connected, 60), "the MCP server answers again: the supervisor reconnects")
    tool_round_trip("r-3", "after the pause")

elif phase == "reconnect-crash":
    # the MCP server dies during the tool call: the turn is answered, not a server error
    chat("crash-1", "crash the server")
    code, text = chat("crash-1", "yes")
    check(code == 200 and "Summary" in text and "chat failed" not in text,
          f"a tool call during which the MCP server goes away is answered ({code}, {text[:50]!r})")

elif phase == "reconnect-again":
    # the restarted MCP server does not know the old session: the ping fails and the supervisor reconnects
    check(wait_for(lambda: startup_log_count("Splunk MCP reconnected") >= 2, 40) and connected(),
          f"the MCP server restarted: the supervisor reconnects ({startup_log_count('Splunk MCP reconnected')} reconnects)")
    tool_round_trip("r-2", "after the MCP restart")
    code, text = chat("r-pending", "yes")  # the approval belonged to the graphs before the reconnect
    check(code == 200 and "Do you approve" in text,
          f"a 'yes' to an approval from before the reconnect starts a new turn, not the old tool call ({text[:40]!r})")

sys.exit(1 if failures else 0)
