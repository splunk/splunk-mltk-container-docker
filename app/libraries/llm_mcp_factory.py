from __future__ import annotations

from typing import Optional, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools 
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND
import asyncio
import re
import os, httpx
from datetime import timedelta

from app.model.llm_utils_chat import create_llm

def create_mcp_http_client_no_verify(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        verify=False  # Disable SSL certificate verification
    )


def redact(text) -> str:
    """Never log credentials in a URL (scheme://user:password@host): httpx error messages keep them."""
    return re.sub(r"//[^/@\s]*@", "//", str(text))


def failure_reason(error) -> str:
    """The causes of a failed MCP call: the client wraps them (refused, 401, TLS, DNS) in exception groups."""
    causes = [error]
    while any(isinstance(cause, BaseExceptionGroup) for cause in causes):
        causes = [leaf for cause in causes
                  for leaf in (cause.exceptions if isinstance(cause, BaseExceptionGroup) else [cause])]
    return redact("; ".join(f"{type(cause).__name__}: {cause}" for cause in causes))


def transient(error) -> bool:
    """An error that says nothing about tool support: the provider was not reachable yet, timed out,
    rate-limited or failed on its side (openai, httpx, ollama, botocore and google error classes)."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    for attribute in ("status_code", "code", "status"):  # HTTP status of the provider's answer
        status = getattr(error, attribute, None)
        if isinstance(status, int) and (status == 429 or 500 <= status < 600):
            return True
    names = " ".join(cls.__name__ for cls in type(error).__mro__)
    return any(word in names for word in ("Connect", "Timeout", "RateLimit", "Throttl", "Unavailable",
                                          "InternalServer", "ServerError", "ResourceExhausted", "TooManyRequests"))


class SplunkMCPManager:
    def __init__(self, *, url: str, token: str, transport: str = "streamable_http", request_timeout: float = 40):
        self.url = url
        self.token = token
        self.transport = transport
        self.request_timeout = request_timeout  # seconds an MCP request (a tool call) may take
        self.is_connected = False
        self.client: Optional[MultiServerMCPClient] = None
        self.num_of_tools = 0
        self.last_error = ""  # why the last ping failed
        self.terminated = False  # the server no longer knows the session (it restarted)
        self._runner: Optional[asyncio.Task] = None
        self._stop: Optional[asyncio.Event] = None
        self.session = None
        self.tools: Optional[List] = None
        self.llm_list = ['ollama', 'bedrock', 'azure_openai', 'openai', 'gemini']

    async def connect(self, log, timeout=60) -> List:
        """
        Create the MCP client, open a persistent session, and load tools bound to that session.
        Safe to call once at startup. The session stays open in its own task until close().
        Without tools after timeout seconds, the connection counts as failed.
        """
        if self.tools is not None:
            return self.tools  # already connected
        if not self.url:  # no Splunk MCP server configured in DSDL
            self.tools, self.num_of_tools = [], 0
            return self.tools

        self.client = MultiServerMCPClient(
            {
                "splunk-mcp": {
                    "transport": self.transport,
                    "url": self.url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                    "httpx_client_factory": create_mcp_http_client_no_verify,
                    # without it a request whose server went away (a Splunk restart) waits forever;
                    # a timed-out tool call reaches the LLM as an error it explains
                    "session_kwargs": {"read_timeout_seconds": timedelta(seconds=self.request_timeout)},
                }
            }
        )

        loaded = asyncio.get_running_loop().create_future()
        loaded.add_done_callback(lambda f: f.cancelled() or f.exception())  # no "never retrieved" warning
        self._stop = asyncio.Event()
        self._runner = asyncio.create_task(self._hold_session(loaded, log))
        try:
            self.tools = await asyncio.wait_for(asyncio.shield(loaded), timeout=timeout)
        except asyncio.CancelledError:
            self._runner.cancel()
            raise
        except Exception as e:
            self._runner.cancel()  # gives up a connection that is still trying
            self.tools = []
            self.num_of_tools = 0
            reason = f"no answer within {timeout} s" if isinstance(e, TimeoutError) else failure_reason(e)
            print(f"Splunk MCP connection to {redact(self.url)} failed: {reason}")
            log.error(f"Splunk MCP connection to {redact(self.url)} failed: {reason}")
            return self.tools

        self.is_connected = True
        self.num_of_tools = len(self.tools)
        print("\n--------------------------------------------------Splunk MCP--------------------------------------------------\n")
        print(f"\nLoaded {len(self.tools)} tools from Splunk MCP:\n")
        log.info("\n--------------------------------------------------Splunk MCP--------------------------------------------------\n")
        log.info(f"Loaded MCP Tools:\n")
        log.info(f"Number of tools: {self.num_of_tools}")
        for tool in self.tools:
            print(f"- {tool.name}: {tool.description}")
            log.info(f"- {tool.name}: {tool.description}")
        print("----------------------------------------------------------------------------------------------------")
        log.info("----------------------------------------------------------------------------------------------------")
        return self.tools

    async def ping(self, timeout=10) -> bool:
        """True when the session answers an MCP ping within timeout seconds. A JSON-RPC error reply
        from the server (one that does not implement ping) also shows the session is alive; the
        client's own errors (connection closed, request timeout, "Session terminated" when a
        restarted server no longer knows the session) do not."""
        session = self.session
        if session is None or not self.is_connected:
            self.last_error = "no open session"
            return False
        try:
            await asyncio.wait_for(session.send_ping(), timeout=timeout)
            return True
        except McpError as e:
            code = getattr(getattr(e, "error", None), "code", None)
            if code in (METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR):
                return True
            self.terminated = code == 32600  # the client's "Session terminated" (HTTP 404)
            self.last_error = failure_reason(e)
        except TimeoutError:
            self.last_error = f"no answer to a ping within {timeout} s"
        except Exception as e:
            self.last_error = failure_reason(e)
        return False

    async def _hold_session(self, loaded, log) -> None:
        """Open the MCP session, hand its tools to connect() and keep it open until close().
        The session's anyio task group must be exited in the task that entered it, so this task
        owns the session for its whole life."""
        try:
            async with self.client.session("splunk-mcp") as session:
                self.session = session
                tools = await load_mcp_tools(session)
                if not loaded.done():  # connect() may have been cancelled meanwhile
                    loaded.set_result(tools)
                await self._stop.wait()
        except Exception as e:
            if not loaded.done():
                loaded.set_exception(e)
            else:
                self.is_connected = False
                print(f"Splunk MCP session to {redact(self.url)} ended: {failure_reason(e)}")
                log.error(f"Splunk MCP session to {redact(self.url)} ended: {failure_reason(e)}")
        finally:
            self.session = None
            if not loaded.done():  # cancelled before the tools loaded
                loaded.set_exception(RuntimeError("MCP session closed before its tools loaded"))

    async def get_llm(self, tools, log, llm, probe=True, probe_timeout=30):
        """One entry per provider: {llm_client, llm_with_tools, Able_to_hold_tools}. A client that
        cannot hold tools, or that gets no tools (no MCP connection), gets llm_with_tools
        "Cannot hold tools", so its chat runs without tools. Clients are built in a thread (their
        constructors are synchronous and must not block the API's event loop). With probe, one
        short tool-calling request per configured provider checks that it accepts tool calls (a
        paid call for hosted providers); the probes run in parallel. A probe the provider rejects
        drops the tools; one that fails for a transient reason (not reachable yet, rate limit, 5xx)
        or is still running after probe_timeout seconds (a model still loading) keeps them.
        Without probe, bind_tools decides."""
        llm_clients = {}
        log.info("\n--------------------------------------------------Initialising LLM Clients--------------------------------------------------\n")

        async def initialise(name):
            try:
                client, msg = await asyncio.to_thread(create_llm, service=name)
            except Exception as e:
                log.info(f"ERROR {name}: Failed to initialise LLM, error message: {e}")
                client = None
            if client is None:
                return name, {"llm_client": None, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            if not tools:  # an empty tool list is rejected by some APIs (OpenAI: 400)
                return name, {"llm_client": client, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            try:
                llm_with_tools = client.bind_tools(tools)
            except Exception as e:
                log.info(f"{name}: client cannot use bind_tools: {e}")
                return name, {"llm_client": client, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            if probe:
                try:
                    await asyncio.wait_for(llm_with_tools.ainvoke("Can you help me get the list of indexes?"),
                                           timeout=probe_timeout)
                    log.info(f"{name}: client can invoke tool calls")
                except TimeoutError:
                    log.info(f"{name}: no answer to the tool-calling check within {probe_timeout} s; keeping the tools")
                except Exception as e:
                    if transient(e):
                        log.info(f"{name}: tool-calling check failed ({type(e).__name__}); keeping the tools")
                        return name, {"llm_client": client, "llm_with_tools": llm_with_tools, "Able_to_hold_tools": True}
                    log.info(f"{name}: client cannot invoke tool calls, chat runs without tools: "
                             f"{type(e).__name__}: {e}")
                    return name, {"llm_client": client, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            return name, {"llm_client": client, "llm_with_tools": llm_with_tools, "Able_to_hold_tools": True}

        for name, entry in await asyncio.gather(*(initialise(name) for name in self.llm_list)):
            llm_clients[name] = entry
        log.info("----------------------------------------------------------------------------------------------------")
        return llm_clients
    
    async def close(self) -> None:
        """
        Close the persistent MCP session.
        Call this on FastAPI shutdown.
        """
        if self._runner is not None:
            self._stop.set()
            _, pending = await asyncio.wait({self._runner}, timeout=10)
            for task in pending:
                task.cancel()

        self._runner = None
        self.is_connected = False
        self.session = None
        self.tools = None
        self.client = None
