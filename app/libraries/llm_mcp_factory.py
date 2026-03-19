from __future__ import annotations

from typing import Optional, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools 
import os, httpx

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


class SplunkMCPManager:
    def __init__(self, *, url: str, token: str, transport: str = "streamable_http"):
        self.url = url
        self.token = token
        self.transport = transport
        self.is_connected = False
        self.client: Optional[MultiServerMCPClient] = None
        self.num_of_tools = 0
        self._session_cm = None
        self.session = None
        self.tools: Optional[List] = None
        self.llm_list = ['ollama', 'bedrock', 'azure_openai', 'openai', 'gemini']

    async def connect(self, log) -> List:
        """
        Create MCP client, open a persistent session, and load tools bound to that session.
        Safe to call once at startup.
        """
        if self.tools is not None:
            return self.tools  # already connected

        self.client = MultiServerMCPClient(
            {
                "splunk-mcp": {
                    "transport": self.transport,
                    "url": self.url,
                    "headers": {"Authorization": f"Bearer {self.token}"},
                    "httpx_client_factory": create_mcp_http_client_no_verify,
                }
            }
        )

        # Open and keep session alive
        try:
            self._session_cm = self.client.session("splunk-mcp")
            self.session = await self._session_cm.__aenter__()

            # Load tools bound to the OPEN session
            self.tools = await load_mcp_tools(self.session)
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
            
        except Exception as e:
            self.tools = []
            self.num_of_tools = 0
            return self.tools

    async def get_llm(self, tools, log, llm):
        llm_list = self.llm_list
        llm_clients = {}

        log.info("\n--------------------------------------------------Initialising LLM Clients--------------------------------------------------\n")
        for llm in llm_list:
            try:
                client, msg = create_llm(service=llm)

                llm_clients[f"{llm}"] = {"llm_client": client, "llm_with_tools": None, "Able_to_hold_tools": False}
                log.info(f"Initialising {llm} client\n")
                ## Check that the LLM client is able to bind tools (supports tool calling)
                try:
                    log.info(f"Initialising {llm} client with tools\n")
                    llm_with_tools = client.bind_tools(tools)
                    log.info(f"1st Check: {llm} client can use bind_tools\n")
                    ## Check against the LLM platform to see that even if the client binds tools, can it actually invoke tools.
                    try:
                        query = "Can you help me get the list of indexes?"
                        reply = await llm_with_tools.ainvoke(query)
                        log.info(f"2nd Check: {llm} client can invoke tool_calls: \nQuery: {query}\n{llm} Reply: {reply}\n")
                        llm_clients[f"{llm}"] = {"llm_client": client, "llm_with_tools": llm_with_tools, "Able_to_hold_tools": True}
                    ## If LLM has tools binded, but cannot invoke, not able to hold tools.
                    except Exception as e:
                        log.info(f"2nd Check: ERROR: {llm} client cannot invoke tool_calls: \nQuery: {query}\n{llm} Error: {e}\n")
                        llm_clients[f"{llm}"]= {"llm_client": client, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
                ## If LLM client is unable to bind tool, we will set it as not able to hold tools.
                except Exception as e:
                    log.info(f"1st Check: ERROR: {llm} client cannot use bind_tools.\nError: {e}\n")
                    llm_clients[f"{llm}"]= {"llm_client": client, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            except Exception as e:
                log.info(f"ERROR {llm}: Failed to initialise LLM, error message: {e}")
                llm_clients[f"{llm}"] = {"llm_client": None, "llm_with_tools": "Cannot hold tools", "Able_to_hold_tools": False}
            log.info("----------------------------------------------------------------------------------------------------")
        return llm_clients
    
    async def close(self) -> None:
        """
        Close the persistent MCP session.
        Call this on FastAPI shutdown.
        """
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)

        self._session_cm = None
        self.session = None
        self.tools = None
        self.client = None