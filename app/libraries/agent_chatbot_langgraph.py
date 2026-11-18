### Edited to add in Splunk MCP ###

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from langgraph.prebuilt import ToolNode
# import tools
import copy, json, httpx, re
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from functools import partial



class State(TypedDict):
    # messages have the type "list".
    # The add_messages function appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]
    request_approval: bool
    decision_approval_accepted: bool
    decision_approval_rejected: bool
    tool_calls: Optional[list] = None
    tool_ran: bool


def starting_settings(state:State):
    system_prompt = SystemMessage(content="""You are a friendly chatbot that is capable of running tools.
                              If the user rejects a tool call, you MUST NOT execute that tool for that specific request.
                              However, if the user later clearly changes their mind and explicitly asks you to run the tool again
                                (for example, saying they rejected it by mistake or now approve it), you are allowed to call the relevant tool again, subject to any approval workflow.""")
    has_system_prompt = any(isinstance(m, SystemMessage) for m in state["messages"])
    if not has_system_prompt:
        # Return a state update (LangGraph will append due to add_messages)
        return {"messages": [system_prompt]}

    # No update if system message already exists
    return {}
    


######## Function to define chatbot Node. Determines if it is a normal chat or tool call.
async def chatbot(state: State, llm_with_tools, llm, log):
    try:
        last_msg = state["messages"][-1]
        
        if isinstance(last_msg, ToolMessage):
            additional_info_prompt = HumanMessage(content="The most recent ToolMessage was just executed at the user's request. Summarise that result.")
            # a failed tool call (bad SPL, no permission) has no artifact, only its error text
            tool_results = last_msg.artifact if last_msg.artifact is not None else last_msg.content
            state["messages"].append(AIMessage(content=f"Here is the results from the tool call: {tool_results}"))
            try:
                state["messages"].append(additional_info_prompt)
                response = await llm.ainvoke(state["messages"])
                # response = await llm.ainvoke(state["messages"] + [additional_info_prompt])
            except Exception as e:
                log.error(f"Error in Chatbot node -> error in tool_call because last message was true for isinstance(ToolMessage). Error: {e}")
                print(f"Chat: the LLM call after a tool call failed: {type(e).__name__}")
                return{"messages": "1. Error in Chatbot Node! Please update the admin of the problem.", "request_approval": False, "tool_calls": None}
            
            return{"messages": [response], "request_approval": False, "tool_calls": None}
            # return {"messages": [additioanl_info, response], "request_approval": False, "tool_calls": None}
        # If tool_calls, return tool_calls. Else return None
        else:
            try:
                if isinstance(llm_with_tools, str):
                    response = await llm.ainvoke(state["messages"])
                else:
                    response = await llm_with_tools.ainvoke(state["messages"])
            except Exception as e:
                log.error(f"Error in Chatbot node -> error in invoking normal call. Error: {e}")
                print(f"Chat: the LLM call failed: {type(e).__name__}")
                return {"messages": "2. Error in Chatbot Node! Please update the admin of the problem.", "request_approval": False, "tool_calls": None}
            if getattr(response, "tool_calls", None):
                return {"messages": [response], "request_approval": True, "tool_calls": response.tool_calls}
            # Normal reply, no tool call. Remember to wrap messages in a list, because the state is in a list.
            return {"messages": [response], "request_approval": False, "tool_calls": None}
    except Exception as e:
        log.error(f"Error in Chatbot node! Error is {e}")
        print(f"Chat: error in the chatbot node: {type(e).__name__}")
        return {"messages": "3. Error in Chatbot Node! Please update the admin of the problem.", "request_approval": False, "tool_calls": None}


######## Function to define state after chatbot (determines if got chatbot is seeking approval, running tools OR just normal chat)
def route_after_chatbot(state:State):
    ## Check based on chat node to see if we go to approval node based on the current state
    if state.get("request_approval"):
        return "request_approval"
    ## Check to see if there is anything to do  based on the current state
    return "end"

######## Function for the Approval Node
def approval(state:State, log):
    
    ## Take the last message to see if user has approved the tool use.
    last_msg = state["messages"][-1]
    tool_call_content = last_msg.tool_calls
    tool_names = ""
    arg_list = ""

    ## Getting the tool call arguments to be printed for approval.
    try:
        tool_question = f""
        for i in range(len(tool_call_content)):
            current_tool_name = f"Tool Name: " + tool_call_content[i]["name"]
            tool_names += tool_call_content[i]["name"] + " "
            if len(tool_call_content[i]["args"]) == 0 or None:
                pass
            else:
                for key, value in tool_call_content[i]["args"].items():
                    arg_list += f"\n    {key}: {value}"
            if not arg_list:
                tool_question += current_tool_name + "\nTool Arguments: None"  + " \n"
            else:
                tool_question += current_tool_name +"\nTool Arguments:" + arg_list + " \n"
            
    except Exception as e:
        log.error(f"Error in Approval Node, unable to print out the tool arguments. Error: {e}")
        return {"request_approval": False, "decision_approval_accepted": False, "decision_approval_rejected": True, "messages": [AIMessage(content="There is something wrong with the tool calling. Let the admin know!")]}
    
    approval_decision = interrupt({"question": f"Do you approve this tool use? Reply yes. Any other answers will be treated as no.", "details": f"{tool_question}"}) # {"question": f"Do you approve this tool use? Reply yes. Any other answers will be treated as no.", "details": f"Tool details: {content}"}
    
        
    ## If approved, set the state of approval to False (has been approved).
    ## As the prompt says, only "yes" approves ("Yes", "yes."), not a reply that contains it
    ## ("what about yesterday?") or qualifies it ("yes, but only index=main").
    if re.fullmatch(r"\s*yes\s*[.!]?\s*", str(approval_decision), re.IGNORECASE):
        return {"request_approval": False, "decision_approval_accepted": True, "decision_approval_rejected": False}

    ## If not approved, set the state of approval to False (has made the decision to not be approved)
    else:
        # We must reply to each tool_call with a ToolMessage so the protocol is valid
        tool_msgs = []
        try:
            tool_rejections_reply = f""
            tool_rejected_names = f""
            for tc in state.get("tool_calls", []):
                name_of_tc = tc['name']
                tool_rejected_names += name_of_tc + " "
                tool_msgs.append(
                    ToolMessage(
                        content=f"""The tool '{name_of_tc}' was NOT executed for that specific request because the user rejected it at that moment.
                                    If the user later **changes their mind** and **explicitly asks you to run this tool again or explicitly approves it**, you MAY call this tool in a new request.
                                """,
                        name=tc.get("name"),
                        tool_call_id=tc.get("id"),
                    )
                )
            
        except Exception as e:
            log.error(f"Error in Approval Node. User rejected tools, but not replied properly. Error: {e}")
            return {"request_approval": False, "messages": [AIMessage(content="There is something wrong with the tool calling. Let the admin know!")], "decision_approval_rejected": True, "decision_approval_accepted": False}

        tool_rejections_reply = AIMessage(content = f"Understood. We will not run the tools: {tool_names}")
        tool_msgs.append(tool_rejections_reply)
        return {"request_approval": False, "messages": tool_msgs, "decision_approval_rejected": True, "decision_approval_accepted": False}
    
def route_after_approval(state:State):
    if state.get("decision_approval_accepted"):
        return "decision_approval_accepted"
    if state.get("decision_approval_rejected"):
        return "decision_approval_rejected"
    else:
        return "end"


def build_chatbot_graph(mcp_tools, llm, llm_with_tools, log):
    ## Setting up Tool Node that references to the tool_list from the state "tool_calls". Serves as the executor of the tools.
    # a failing tool call (bad SPL, no permission, a closed MCP session) becomes a ToolMessage with
    # the error, which the LLM explains, instead of an exception that ends the turn mid-graph
    tool_node = ToolNode(mcp_tools, handle_tool_errors=True)
    # Set entry and finish points
    graph_builder = StateGraph(State)
    graph_builder.add_edge(START, "starting_settings")
    graph_builder.add_node("starting_settings", partial(starting_settings))
    graph_builder.add_edge("starting_settings", "chatbot")
    graph_builder.add_node("chatbot", partial(chatbot, llm = llm, llm_with_tools=llm_with_tools, log=log)) ## Either tool_call or no tool_call
    graph_builder.add_conditional_edges(
        "chatbot",
        route_after_chatbot,
        {
            "request_approval": "approval",
            "end": END
        }
    )
    graph_builder.add_node("approval", partial(approval, log=log))
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges(
        "approval",
        route_after_approval,
        {
            "decision_approval_accepted": "tools",
            "decision_approval_rejected": END,
            "end": END
        }
    )
    graph_builder.add_edge("tools", "chatbot")

    checkpointer = MemorySaver()
    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph
