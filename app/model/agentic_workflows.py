#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# Basic imports
import sys
import os
import json 
import splunklib.client
from typing import Sequence, List, Any
from pydantic import Field

# llama-index core imports
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool, FunctionTool, ToolSelection, ToolOutput
from llama_index.core.agent import AgentRunner, ReActAgentWorker, FunctionCallingAgentWorker
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.vector_stores.milvus import MilvusVectorStore
from app.model.llm_utils import create_llm, create_embedding_model

# llama-index tool imports
from llama_index.tools.slack import SlackToolSpec
from llama_index.tools.google import GmailToolSpec
from llama_index.tools.google import GoogleCalendarToolSpec
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.tools.jira import JiraToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec

# llama-index workflow imports
from llama_index.utils.workflow import draw_most_recent_execution
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)

print(f"PYTHON VERSION: {sys.version}")
print("Imported agent tools packages: Google, MCP, Jira, Neo4j, Arxiv, Wiki")

import asyncio
import concurrent.futures
def run_async_in_sync(async_func):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(lambda: asyncio.run(async_func())).result()
    else:
        return asyncio.run(async_func())
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"





    
# In[2]:


# Example chat workflow using Bedrock LLM
# Example SPL:
## | makeresults
## | fit MLTKContainer algo=agentic_workflow_execution workflow_name=SimpleLLMFlow query="What does Splunk do?" * into app:agentic_workflow_execution


class SimpleLLMFlow(Workflow):
    @step
    async def generate(self, ev: StartEvent) -> StopEvent:
        llm, _ = create_llm('bedrock')
        response = await llm.acomplete(ev.query)
        return StopEvent(result=str(response))







    
# In[7]:


# Example workflow as a Splunk Cloud MCP client
# Please fill in the mcp_token and tenant_name variables and change LLM option if needed before execution
# Example SPL:
## | makeresults
## | fit MLTKContainer algo=agentic_workflow_execution workflow_name=SplunkMCPAgent query="What savesearches are there?" * into app:agentic_workflow_execution

class InputEvent(Event):
    input: list[ChatMessage]

class StreamEvent(Event):
    delta: str

class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]

class FunctionOutputEvent(Event):
    output: ToolOutput


async def get_mcp_tools():
    mcp_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    tenant_name = "XXXXXX"
    URL = f"https://{tenant_name}.api.scs.splunk.com/{tenant_name}/mcp/v1/"
    mcp_client = BasicMCPClient(
        URL,
        headers = {
            "Authorization": f"Bearer {mcp_token}",
            "Content-Type": "application/json"
        }
    )
    mcp_tool_spec = McpToolSpec(
        client=mcp_client
    )
    tools = await mcp_tool_spec.to_tool_list_async()
    return tools

class SplunkMCPAgent(Workflow):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        tools = run_async_in_sync(get_mcp_tools)
        llm, _ = create_llm('bedrock')
        # Initialize tools and LLM
        self.tools = tools
        self.llm = llm
        assert self.llm.metadata.is_function_calling_model


    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.store.set("sources", [])

        # check if memory is setup
        memory = await ctx.store.get("memory", default=None)
        if not memory:
            memory = ChatMemoryBuffer.from_defaults(llm=self.llm)

        # get user input
        user_input = ev.query
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # get chat history
        chat_history = memory.get()

        # update context
        await ctx.store.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # stream the response
        response_stream = await self.llm.astream_chat_with_tools(
            self.tools, chat_history=chat_history
        )
        async for response in response_stream:
            ctx.write_event_to_stream(StreamEvent(delta=response.delta or ""))

        # save the final response, which should have all content
        memory = await ctx.store.get("memory")
        memory.put(response.message)
        await ctx.store.set("memory", memory)

        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.store.get("sources", default=[])
            return StopEvent(result=str(response) + "\n\n\n\n" + "Tool Execution Results:\n" + str([*sources]) )
            # return StopEvent(
            #     result={"response": response, "sources": [*sources]}
            # )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.store.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        # update memory
        memory = await ctx.store.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.store.set("sources", sources)
        await ctx.store.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)





    
# In[12]:








    
# In[14]:








    
# In[16]:








    
# In[17]:








    
# In[18]:










