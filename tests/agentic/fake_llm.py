"""Fake OpenAI-compatible chat completions endpoint: asks for a tool call when tools are
offered and the last non-system message is from the user (with the query "broken" or "crash"
when that message says so), else answers in text. Counts calls
and reports the number of non-system messages of the last request (GET /calls).
FAKE_LLM_FIRST_DELAY delays the first answer (the start-up probe) by that many seconds."""
import asyncio
import json
import os
import time
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
calls = {"n": 0, "last_messages": 0}
FIRST_DELAY = float(os.environ.get("FAKE_LLM_FIRST_DELAY", "0"))


@app.get("/calls")
async def call_count():
    return {"calls": calls["n"], "last_messages": calls["last_messages"]}


@app.post("/v1/chat/completions")
@app.post("/openai/deployments/{deployment}/chat/completions")  # the same fake as an Azure OpenAI deployment
async def completions(req: Request):
    body = await req.json()
    calls["n"] += 1
    msgs = [m for m in body.get("messages", []) if m.get("role") not in ("system", "developer")]
    calls["last_messages"] = len(msgs)  # before the delay: a late probe must not overwrite a later call's count
    if calls["n"] == 1 and FIRST_DELAY:
        await asyncio.sleep(FIRST_DELAY)
    last = msgs[-1] if msgs else {}
    if "slowly" in str(last.get("content")):  # an answer that outlasts the chat's turn timeout
        await asyncio.sleep(30)
    tools = [t["function"]["name"] for t in body.get("tools", []) or []]
    print(f"[fake-llm] call#{calls['n']} last_role={last.get('role')} tools={tools} "
          f"last={str(last.get('content'))[:50]!r}", flush=True)
    if tools and last.get("role") == "user":
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{calls['n']}", "type": "function",
            "function": {"name": tools[0], "arguments": json.dumps(
                {"query": next((word for word in ("broken", "crash") if word in str(last.get("content"))),
                               "search index=* | stats count by index")})}}]}
        finish = "tool_calls"
    else:
        msg = {"role": "assistant", "content": "Summary: the search returned 42 events."}
        finish = "stop"
    return {"id": "mi", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
