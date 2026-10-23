# Splunk App for Data Science and Deep Learning 5.2.5
# Creator: Philipp Drieger, Principal AI Architect
# Authors: Huaibo Zhao
# 2018-2026
# -------------------------------------------------------------------------------
"""Agentic AI API of the agentic-ai image (DSDL_APP_MODULE=app.main_agentic:app).

It serves the DSDL API of app.main (fit, apply, summary, ...) unchanged and adds the chat
routes /chatbox, /logReview and /mcp/status:

* The Splunk MCP connection, the LLM clients and the LangGraph graphs are set up in the
  background once the API runs, so a slow or unreachable MCP or LLM endpoint no longer delays
  /fit and /apply. /mcp/status reports "start-up in progress" until then, and a chat waits up to
  CHAT_WAIT_SECONDS for it. A start-up that raises is retried after RETRY_SECONDS.
* The Splunk MCP connection is supervised: every DSDL_AGENTIC_MCP_RETRY seconds a connection that
  failed (refused, 401, no answer within DSDL_AGENTIC_MCP_TIMEOUT seconds) is retried and one that
  failed two pings in a row (a Splunk restart) is reopened; the chat then gets new graphs with the
  new tools (the tool-calling check runs again, a pending tool approval is dropped, the
  conversations go on). A Splunk MCP request (a tool call) that takes longer than
  DSDL_AGENTIC_TOOL_TIMEOUT seconds fails, and the LLM explains the failure; a whole chat turn
  ends after DSDL_AGENTIC_TURN_TIMEOUT seconds, inside the DSDL proxy's 60 s. A session runs one
  turn at a time: a message sent meanwhile gets "still answering".
* The chat routes check the DSDL api_token like /fit and /apply when DSDL_AGENTIC_REQUIRE_TOKEN
  is set (any value but "", 0, false, no, off). Until the DSDL app sends the token on its chat
  requests the default is off; a request that sends a wrong token is refused either way.
* No CORS: the DSDL UI reaches these routes only through the Splunk REST proxy, server side.
* /mcp/status returns only what the DSDL UI reads, never the client configuration.
* At start-up, one short tool-calling request per configured LLM checks that the provider accepts
  tool calls (possibly a paid request); one that fails gets the chat without tools, one that takes
  longer than DSDL_AGENTIC_PROBE_TIMEOUT seconds keeps them. DSDL_AGENTIC_PROBE_TOOLS=false skips
  the requests; bind_tools then decides.
* Bounded memory: request bodies are capped (DSDL_AGENTIC_MAX_BODY_MB for /logReview,
  CHAT_MAX_BYTES for /chatbox, also for chunked uploads), /logReview reads at most MAX_LOG_ENTRIES
  events, a session keeps its last MAX_MSGS messages and a LangGraph checkpoint only while a tool
  approval is pending (bound to the graph that asked), and sessions are evicted after
  TTL_EVICT_SECONDS without use, the oldest beyond MAX_SESSIONS. The session logs record each
  message and answer, not the whole conversation every turn.
  Chat state lives in the worker process, so the entrypoint runs this API with one worker.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, Header, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.libraries.agent_chatbot_langgraph import build_chatbot_graph
from app.libraries.llm_mcp_factory import SplunkMCPManager
from app.libraries.logging_function import get_logger
from app.main import app, validate_token  # the DSDL API (fit, apply, summary, ...)
from app.model.llm_utils_chat import is_configured


def env_int(name, default):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_value(name, default=""):
    return os.environ.get(name, default).strip().strip("\"'").strip().lower()


def env_flag(name, default):
    return env_value(name, default) in ("1", "true", "yes", "on")


MAX_MSGS = env_int("MAX_MSGS", 20)
TTL_EVICT_SECONDS = env_int("TTL_EVICT_SECONDS", 3600)
MAX_LOG_TOKEN_SIZE = env_int("MAX_LOG_TOKEN_SIZE", 3000)
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "bedrock").strip('"')
# fails closed: any value but an explicit "off" requires the token
REQUIRE_TOKEN = env_value("DSDL_AGENTIC_REQUIRE_TOKEN") not in ("", "0", "false", "no", "off")
PROBE_TOOLS = env_flag("DSDL_AGENTIC_PROBE_TOOLS", "true")
PROBE_TIMEOUT = env_int("DSDL_AGENTIC_PROBE_TIMEOUT", 30)
MCP_TIMEOUT = env_int("DSDL_AGENTIC_MCP_TIMEOUT", 60)
MCP_RETRY_SECONDS = env_int("DSDL_AGENTIC_MCP_RETRY", 60)
# an MCP request (a tool call) may take this long; at least the 10 s of a ping, which it also bounds
TOOL_TIMEOUT = max(10, env_int("DSDL_AGENTIC_TOOL_TIMEOUT", 40))
MAX_SESSIONS = env_int("MAX_SESSIONS", 500)
MAX_LOG_ENTRIES = env_int("MAX_LOG_ENTRIES", 5000)  # the DSDL chat viz sends at most 5000 results
MAX_BODY_BYTES = env_int("DSDL_AGENTIC_MAX_BODY_MB", 50) * 1_000_000  # /logReview: results with all their fields
CHAT_MAX_BYTES = 100_000  # a typed message; it stays in the history that every turn sends to the LLM
RETRY_SECONDS = 30
CHAT_WAIT_SECONDS = 20  # the DSDL proxy gives /chatbox 60 s in all, the LLM call included
CHAT_TURN_SECONDS = env_int("DSDL_AGENTIC_TURN_TIMEOUT", 55)  # the whole turn, start-up wait included

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "").strip('"')
system_prompt_chat = SystemMessage(content=SYSTEM_PROMPT or
                                   "You are a friendly chatbot that is well-verse in Splunk and logs. "
                                   "You are here to help people ")

try:
    mcp_list = [item for item in json.loads(os.environ["llm_config"])["mcp"]
                if is_configured({"is_configured": item.get("enabled")})]
    SPLUNK_MCP_URL = mcp_list[0]["url"]
    SPLUNK_MCP_TOKEN = mcp_list[0]["token"]
except (KeyError, IndexError, TypeError, ValueError):
    SPLUNK_MCP_URL = None
    SPLUNK_MCP_TOKEN = None

## Base Directory for Logging
base_dir = os.getcwd()

## Background start-up of the agentic part (MCP connection, LLM clients, graphs)
startup = {"task": None, "failed_at": 0.0, "supervisor": None}


def startup_logger():
    return get_logger(userName="Chatbox_Container", userSession=time.strftime("%Y-%m-%d"), base_dir=base_dir)


async def start_agentic():
    log = startup_logger()
    log.info(f"\n========================== AGENTIC START-UP: {time.strftime('%Y-%m-%d %H:%M:%S')} =========================\n")
    print("Agentic start-up: connecting Splunk MCP and LLM clients in the background", flush=True)
    mcp = SplunkMCPManager(url=SPLUNK_MCP_URL, token=SPLUNK_MCP_TOKEN, request_timeout=TOOL_TIMEOUT)
    try:
        tools = await mcp.connect(log=log, timeout=MCP_TIMEOUT)
        if mcp.is_connected:
            print(f"Success: MCP Connection! ({mcp.num_of_tools} tools)")
            log.info("Success: MCP Connection!")
        elif not SPLUNK_MCP_URL:
            print("No Splunk MCP server configured")
            log.info("No Splunk MCP server configured")
        else:
            print(f"WARNING: MCP Connection failed: no Splunk MCP tools available; retrying every {MCP_RETRY_SECONDS} s")
            log.warning("MCP Connection failed: no Splunk MCP tools available")
        llm_clients, graphs = await build_chat(mcp, tools, log)
    except BaseException:
        await mcp.close()  # a retried start-up opens a new MCP session
        raise
    publish_chat(mcp, llm_clients, graphs)
    summary = {name: v["Able_to_hold_tools"] for name, v in llm_clients.items() if v["llm_client"] is not None}
    print(f"Agentic start-up done: chat with {sorted(graphs) or 'no LLM'}; tool calling: {summary}", flush=True)
    log.info(f"Agentic start-up done: graphs {sorted(graphs)}, tool calling {summary}")
    if SPLUNK_MCP_URL and startup["supervisor"] is None:
        startup["supervisor"] = asyncio.get_running_loop().create_task(supervise_mcp())


async def build_chat(mcp, tools, log):
    """The LLM clients and one chat graph per configured LLM, with the tools of this connection."""
    llm_clients = await mcp.get_llm(tools=tools, llm=None, log=log, probe=PROBE_TOOLS, probe_timeout=PROBE_TIMEOUT)
    graphs = {}
    for llm_name, value in llm_clients.items():
        if value["llm_client"] is None:
            continue
        try:
            graphs[llm_name] = build_chatbot_graph(llm=value["llm_client"], llm_with_tools=value["llm_with_tools"],
                                                   mcp_tools=tools, log=log)
        except Exception as e:
            log.info(f"Graph {llm_name} failed to build. Error: {e}")
            print(f"Graph {llm_name} failed to build: {type(e).__name__}: {e}")
    return llm_clients, graphs


def publish_chat(mcp, llm_clients, graphs):
    """Serve the chat from this connection and these graphs. A session keeps its conversation (it is
    passed in on every turn); a pending tool approval belonged to the old graphs and is dropped."""
    app.state.splunk_mcp = mcp
    app.state.llm_clients = llm_clients
    app.state.chatbot_graph_with_splunk_mcp_dict = graphs
    for session in user_session_dict.values():
        session["tool_call_flag"] = False
        session["pending_graph"] = None


async def supervise_mcp():
    """Every MCP_RETRY_SECONDS: retry a Splunk MCP connection that failed, reopen one that no longer
    answers a ping, and rebuild the chat with the new tools."""
    log = startup_logger()
    failed_pings = 0
    while True:
        await asyncio.sleep(MCP_RETRY_SECONDS)
        current = app.state.splunk_mcp
        if current is not None and current.is_connected:
            if await current.ping():
                failed_pings = 0
                continue
            failed_pings += 1
            print(f"Splunk MCP ping failed ({failed_pings}): {current.last_error}", flush=True)
            if failed_pings < 2 and not current.terminated:  # one busy moment of Splunk is not a dead session
                continue
            current.is_connected = False  # /mcp/status reports it until a reconnect succeeds
        failed_pings = 0
        print("Splunk MCP is not connected: reconnecting", flush=True)
        mcp = SplunkMCPManager(url=SPLUNK_MCP_URL, token=SPLUNK_MCP_TOKEN, request_timeout=TOOL_TIMEOUT)
        try:
            tools = await mcp.connect(log=log, timeout=MCP_TIMEOUT)
            if not mcp.is_connected:
                await mcp.close()
                continue
            llm_clients, graphs = await build_chat(mcp, tools, log)
        except asyncio.CancelledError:
            await mcp.close()
            raise
        except Exception as e:
            await mcp.close()
            print(f"WARNING: Splunk MCP reconnect failed: {type(e).__name__}: {e}", flush=True)
            continue
        publish_chat(mcp, llm_clients, graphs)
        if current is not None:
            await current.close()
        summary = {name: v["Able_to_hold_tools"] for name, v in llm_clients.items() if v["llm_client"] is not None}
        print(f"Splunk MCP reconnected ({mcp.num_of_tools} tools); tool calling: {summary}", flush=True)
        log.info(f"Splunk MCP reconnected ({mcp.num_of_tools} tools); tool calling: {summary}")


def startup_finished(task):
    if not task.cancelled() and task.exception() is not None:
        startup["failed_at"] = time.monotonic()
        print(f"WARNING: agentic start-up failed, retrying after {RETRY_SECONDS} s: "
              f"{type(task.exception()).__name__}: {task.exception()}", flush=True)


def ensure_started():
    """The start-up task; a failed one is replaced once RETRY_SECONDS have passed."""
    task = startup["task"]
    failed = task is not None and task.done() and (task.cancelled() or task.exception() is not None)
    if task is None or (failed and time.monotonic() - startup["failed_at"] >= RETRY_SECONDS):
        task = asyncio.get_running_loop().create_task(start_agentic())
        task.add_done_callback(startup_finished)
        startup["task"] = task
    return task


async def started(timeout):
    """True when the agentic start-up finished successfully, waiting up to timeout seconds
    (0: no wait). The wait never cancels the start-up."""
    task = ensure_started()
    if not task.done() and timeout:
        await asyncio.wait({task}, timeout=timeout)
    return task.done() and not task.cancelled() and task.exception() is None


@asynccontextmanager
async def lifespan(_app):
    app.state.splunk_mcp = None
    app.state.llm_clients = {}
    app.state.chatbot_graph_with_splunk_mcp_dict = {}
    if not REQUIRE_TOKEN:
        print("WARNING: the chat routes accept requests without the api_token "
              "(set DSDL_AGENTIC_REQUIRE_TOKEN=true once the DSDL app sends it)")
    ensure_started()
    try:
        yield
    finally:
        for task in (startup["supervisor"], startup["task"]):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=15)
        if app.state.splunk_mcp is not None:
            await app.state.splunk_mcp.close()


app.router.lifespan_context = lifespan


def agentic_auth(authorization: str = Header(None)):
    """Token check of the chat routes: a sent token must be valid; a missing one is refused once
    DSDL_AGENTIC_REQUIRE_TOKEN is true."""
    if authorization is not None or REQUIRE_TOKEN:
        validate_token(authorization)


user_session_dict = {} ##"userSession": {"logs": [],"userName": userName,"convoHistory": [],"tool_call_flag": False,})
## Manage "last seen" for users
session_last_seen = {}

######################################################## HOUSE KEEPING FUNCTIONS ##########################################
## Function to manage history for token
def trim_history(hist):
    """The last MAX_MSGS messages (system messages are kept), cut before a user message so that no
    tool result loses the tool call it answers."""
    if len(hist) <= MAX_MSGS:
        return hist
    users = [i for i, m in enumerate(hist) if isinstance(m, HumanMessage)]
    start = next((i for i in users if i >= len(hist) - MAX_MSGS), users[-1] if users else len(hist) - MAX_MSGS)
    return [m for m in hist[:start] if isinstance(m, SystemMessage)] + hist[start:]

## Function to manage TTL for each user
def touch_session(userSession: str) -> None:
    """Mark a session as recently used."""
    session_last_seen[userSession] = time.time()

## Function to clean_up_expired_sessions()
def cleanup_expired_sessions() -> None:
    """Forget sessions unused for TTL_EVICT_SECONDS, and the oldest beyond MAX_SESSIONS: their chat
    state and the checkpoints every graph keeps per session (thread_id)."""
    now = time.time()
    by_age = sorted(session_last_seen, key=session_last_seen.get)
    expired = [sid for sid in by_age if now - session_last_seen[sid] > TTL_EVICT_SECONDS]
    expired += [sid for sid in by_age[:max(0, len(by_age) - MAX_SESSIONS)] if sid not in expired]
    for sid in expired:
        session_last_seen.pop(sid, None)
        user_session_dict.pop(sid, None)
        for graph in app.state.chatbot_graph_with_splunk_mcp_dict.values():
            graph.checkpointer.delete_thread(sid)

################################################################################################################
## Function to limit the number of tokens coming from logs
def count_tokens(logs_list, log):
    total_token_count = 0
    list_of_usable_logs = []
    for i in range(len(logs_list)):
        userRawLogs = str(logs_list[i].get('_raw', ''))
        len_of_log_char = len(userRawLogs) ## Count the number of characters
        token_count_for_curr_log = len_of_log_char/4 ## Assume that 1 token is 4 characters
        total_token_count += token_count_for_curr_log
        if total_token_count > MAX_LOG_TOKEN_SIZE:
            log.info(f"Number of tokens exceeds > {MAX_LOG_TOKEN_SIZE}, will truncate the logs: "
                     f"{len(list_of_usable_logs)} of {len(logs_list)} events kept.")
            return list_of_usable_logs
        else:
            list_of_usable_logs.append(logs_list[i])
    log.info(f"Number of tokens did not exceed {MAX_LOG_TOKEN_SIZE}: {len(list_of_usable_logs)} events kept.")
    return list_of_usable_logs

class ChatResponse(BaseModel):
    data: str


def drop_pending_approval(session, session_id):
    graph = session.get("pending_graph")
    if graph is not None:
        graph.checkpointer.delete_thread(session_id)
    session["tool_call_flag"] = False
    session["pending_graph"] = None


async def read_json(req, limit):
    """The request's JSON object, or None when the body is larger than limit bytes (counted while
    reading, so also for chunked uploads without a Content-Length). A body over the limit is read
    to its end but not kept: a reply sent before it is read would reset the client's upload."""
    body = bytearray()
    size = 0
    async for chunk in req.stream():
        size += len(chunk)
        if size <= limit:
            body += chunk
        elif size > 4 * limit:
            break  # give up on an upload far beyond the limit; that client's upload is reset
    if size > limit:
        return None
    try:
        data = json.loads(body or b"{}")  # on the event loop: the C decoder holds the GIL anyway
    except ValueError:
        raise HTTPException(status_code=400, detail="The request body is not valid JSON")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="The request body must be a JSON object")
    return data
################################################################################################################


@app.get("/mcp/status", dependencies=[Depends(agentic_auth)])
async def mcp_status():
    # the DSDL UIs poll this every second: answer at once, also during the start-up
    if not await started(timeout=0):
        task = startup["task"]
        detail = "start-up failed, retrying" if task.done() else "start-up in progress"
        return {"connected": False, "detail": detail, "llm_list": {}}
    mcp = app.state.splunk_mcp
    # only what the DSDL UI reads: never the serialised client configuration
    llm_list = {name: {"Able_to_hold_tools": bool(value.get("Able_to_hold_tools")),
                       "available": value.get("llm_client") is not None}
                for name, value in app.state.llm_clients.items()}
    return {"connected": bool(mcp and mcp.is_connected), "detail": f"{mcp.num_of_tools if mcp else 0} tools",
            "llm_list": llm_list}

@app.post("/logReview", dependencies=[Depends(agentic_auth)])
async def logReview(req:Request):
    body = await read_json(req, MAX_BODY_BYTES)
    if body is None:  # both DSDL UIs show the answer text, not an HTTP error
        return ChatResponse(data=f"The selected events are too large to review (over {MAX_BODY_BYTES // 1_000_000} MB). "
                                 "Select fewer events or fields.")
    userName = body.get("userName")
    userSession = body.get("sessionID")
    userRawData = body.get("logs") or []
    if not isinstance(userRawData, list):
        userRawData = []
    # at most MAX_LOG_ENTRIES events, each one a dict with its _raw text
    userRawData = [entry if isinstance(entry, dict) else {"_raw": str(entry)} for entry in userRawData[:MAX_LOG_ENTRIES]]
    log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
    print("\n========================== FASTAPI from LogReview API endpoint =========================\n")
    log.info("\n========================== FASTAPI from LogReview API endpoint =========================\n")
    ## Reset TTL for each user session
    touch_session(userSession)
    cleanup_expired_sessions()  #Clean up expired sessions

    log.info(f"User(Frontend) Logs:\nUser: {userName}\nSessionID: {userSession}\nLogs: {len(userRawData)} events\n")
    updated_logs_list = count_tokens(userRawData, log)

    append_Raw_Logs = "".join("\n" + str(entry.get("_raw", "")).replace("{", "").replace("}", "")
                              for entry in updated_logs_list)
    append_Raw_Logs_to_LLM = "Raw Logs: \n" + append_Raw_Logs ## append_Raw_Logs_to_LLM is the finalised string to be used in prompts for logs.

    ## Setting up the history for each userSession
    # Tactic: Set up the conversation chain Human->AI for asking of approval for log summary.
    query = HumanMessage('Here is my log set from Splunk. ' + append_Raw_Logs_to_LLM) ## HumanMessage to be appended (Coversation chain for logs)
    summarise_logs_query = AIMessage('Would you like to summarise the logs? I will return a summary of what happened within the logs in less than 200 words.') ## AI Message to be appended after HumanMessage(Conversation Chain simulating the LLM asking for summary)

    current_session = user_session_dict.setdefault(userSession, {
    "logs": [],
    "userName": userName,
    "convoHistory": [],
    "tool_call_flag": False,
    "pending_graph": None,
    })
    if current_session.get("busy"):  # the running turn would overwrite the history with its own
        return ChatResponse(data="Still answering your previous message. Send the events again after that answer.")
    if current_session["tool_call_flag"]:  # the next "yes" answers the log summary, not that tool call
        drop_pending_approval(current_session, userSession)
    # Update conversation history for current user
    current_session["convoHistory"].append(query)
    current_session["convoHistory"].append(summarise_logs_query)

    # Manage History Length
    current_session["convoHistory"] = trim_history(current_session["convoHistory"])

    log.info(f"------------------------- Finalised Query -------------------------")
    log.info(f"\nLogs!! HumanMessage: {query}\n")
    log.info(f"AIMessage: {summarise_logs_query}")
    log.info(f"------------------------- End Query -------------------------")
    log.info(f"===========================================================================\n")
    return ChatResponse(data=summarise_logs_query.content)



@app.post("/chatbox", dependencies=[Depends(agentic_auth)])
async def chat(req:Request):
    print("\n========================== FASTAPI from Chatbot Endpoint =========================\n")
    body = await read_json(req, CHAT_MAX_BYTES)
    if body is None:
        return ChatResponse(data=f"The message is too long (over {CHAT_MAX_BYTES // 1000} KB).")
    userName = body.get("userName")
    userSession = body.get("sessionID")
    userMessage = body.get("message", "")
    # DSDL may pass the option in literal double quotes, like DEFAULT_LLM
    llm_option = (body.get("llmOption") or "").strip().strip('"')

    ## Setup logger
    log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
    log.info(f"\n========================== FASTAPI from Chatbot Endpoint =========================\n")
    log.info(f"User(Frontend) Query:\nUser: {userName}\nSessionID: {userSession}\nMessage: {userMessage}\nLLM Option: {llm_option}")
    touch_session(userSession)
    cleanup_expired_sessions()
    if llm_option == "":
        log.error(f"LLM Option not selected! Informing user to select LLM!\n")
        log.info(f"===========================================================================\n")
        return ChatResponse(data="Please select your LLM Option!")
    turn_started = time.monotonic()
    if not await started(timeout=CHAT_WAIT_SECONDS):
        log.error("Agentic start-up not finished; asking the user to retry\n")
        return ChatResponse(data="The chat is still starting (Splunk MCP and LLM connections). Please retry in a moment.")
    if llm_option not in app.state.chatbot_graph_with_splunk_mcp_dict:
        available = ", ".join(sorted(app.state.chatbot_graph_with_splunk_mcp_dict)) or "none"
        log.error(f"LLM Option {llm_option!r} is not available (available: {available})\n")
        return ChatResponse(data=f"LLM option '{llm_option}' is not available. Available options: {available}")
    llm_client_graph = app.state.chatbot_graph_with_splunk_mcp_dict[llm_option]
    touch_session(userSession)  # again: the start-up wait may have let the session expire
    current_session = user_session_dict.setdefault(userSession, {
    "logs": [],
    "userName": userName,
    "convoHistory": [],
    "tool_call_flag": False,
    "pending_graph": None,
    })
    if current_session["tool_call_flag"] and current_session.get("pending_graph") is not llm_client_graph:
        # the approval was asked by another graph (another LLM option, or the graphs were rebuilt
        # after an MCP reconnect): this message is a new turn, not an answer to it
        log.info("The pending tool approval belongs to another graph; dropping it\n")
        drop_pending_approval(current_session, userSession)
    if current_session.get("busy"):
        return ChatResponse(data="Still answering your previous message. Please wait for that answer.")
    log.info(f"LLM Graph Chosen: {llm_option}; history {len(current_session['convoHistory'])} messages, "
             f"tool approval pending: {current_session['tool_call_flag']}\n")
    config = {"configurable": {"thread_id": userSession}}
    turn_seconds = max(5.0, CHAT_TURN_SECONDS - (time.monotonic() - turn_started))
    current_session["busy"] = True
    try:
        #Check if there is tool call approval in progress.
        if current_session["tool_call_flag"]:
            log.info(f"Current Session has tool approval in progress.\n")
            current_session["tool_call_flag"] = False
            current_session["pending_graph"] = None
            final_state = await asyncio.wait_for(llm_client_graph.ainvoke(Command(resume=userMessage), config=config),
                                                 timeout=turn_seconds)
        else:
            log.info(f"Current Session has NO tool approval in progress.\n")
            state = {
                "messages": current_session["convoHistory"] + [HumanMessage(content=userMessage)],
                "summarisation_agent_bool": False,
                "solution_agent_bool": False,
                "final_output": ""
            }
            final_state = await asyncio.wait_for(llm_client_graph.ainvoke(state, config=config), timeout=turn_seconds)
    except Exception as e:
        # keep no half-run checkpoint: the next turn starts from the stored history
        llm_client_graph.checkpointer.delete_thread(userSession)
        log.error(f"Chat turn failed: {type(e).__name__}: {e}\n")
        print(f"Chat: the turn failed: {type(e).__name__}")
        if isinstance(e, TimeoutError):
            return ChatResponse(data=f"The answer took longer than {CHAT_TURN_SECONDS} s. Please try again.")
        return ChatResponse(data=f"The chat failed ({type(e).__name__}). Please try again.")
    finally:
        current_session["busy"] = False
    interrupts = final_state.get("__interrupt__", [])
    log.info(f"LLM Agent decision on tool call or not: {interrupts}")
    if user_session_dict.get(userSession) is not current_session:
        # the session was evicted while the graph ran: keep nothing of it
        llm_client_graph.checkpointer.delete_thread(userSession)
        current_session = {"tool_call_flag": False, "pending_graph": None}
    if interrupts:
        current_session["tool_call_flag"]=True
        current_session["pending_graph"] = llm_client_graph
        payload = interrupts[0].value
        payload_question = (payload or {}).get("question", "")
        payload_details = (payload or {}).get("details", "")
        tool_msg = "\n".join(part for part in [payload_question, payload_details] if part).strip()
        log.info(f"Interrupted! Tool Message: {tool_msg}")
        print("Chat: tool call awaits approval")
        return ChatResponse(data=tool_msg)
    log.info(f"No interrupts.")
    current_session["tool_call_flag"]=False
    llm_output = final_state["messages"][-1]
    message = llm_output.content
    log.info(f"Answer: {message}")
    # the next turn passes the history in again: the graph keeps a checkpoint only while a tool
    # approval is pending
    llm_client_graph.checkpointer.delete_thread(userSession)
    current_session['convoHistory'] = trim_history(final_state['messages'])
    print("Chat: answer returned")
    return ChatResponse(data=message)
################################################################################################################
# -------------------------------------------------------------------------------
