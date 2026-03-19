# Splunk App for Data Science and Deep Learning 5.2.3
# Creator: Philipp Drieger, Principal AI Architect
# Authors: Huaibo Zhao
# 2018-2026
# -------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command
from app.model.llm_utils_chat import create_llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from contextlib import asynccontextmanager
# from dotenv import load_dotenv
from app.libraries.logging_function import get_logger
from app.libraries.agent_chatbot_langgraph import build_chatbot_graph
from app.libraries.llm_mcp_factory import SplunkMCPManager
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from fastapi import HTTPException, status

from importlib import import_module, reload
import pandas as pd
import json
import csv
import os
import time
import uvicorn

try:
    MAX_MSGS = int(os.environ['MAX_MSGS']) 
except:
    MAX_MSGS = 20

try:
    TTL_EVICT_SECONDS = int(os.environ['TTL_EVICT_SECONDS']) 
except:
    TTL_EVICT_SECONDS = 3600   

try:
    MAX_LOG_TOKEN_SIZE = int(os.environ['MAX_LOG_TOKEN_SIZE']) 
except:
    MAX_LOG_TOKEN_SIZE = 3000

try:
    DEFAULT_LLM = os.environ['DEFAULT_LLM'].strip('"')
except:
    DEFAULT_LLM = 'bedrock'

try:
    SYSTEM_PROMPT = os.environ['SYSTEM_PROMPT'].strip('"')
    if len(SYSTEM_PROMPT) > 0:
        system_prompt_chat = SystemMessage(content=SYSTEM_PROMPT)
    else:
        system_prompt_chat = SystemMessage('''You are a friendly chatbot that is well-verse in Splunk and logs. You are here to help people ''')
except:
    system_prompt_chat = SystemMessage('''You are a friendly chatbot that is well-verse in Splunk and logs. You are here to help people ''')

try:
    mcp_list = []
    for item in json.loads(os.environ["llm_config"])['mcp']:
        if item["enabled"] == '1' or item["enabled"] == 'true' or item["enabled"] == True:
            mcp_list.append(item)
    SPLUNK_MCP_URL = mcp_list[0]["url"]
    SPLUNK_MCP_TOKEN = mcp_list[0]["token"]
except:
    SPLUNK_MCP_URL = None
    SPLUNK_MCP_TOKEN = None

## Base Directory for Logging
base_dir = os.getcwd()

## Initialise all the graphs
@asynccontextmanager
async def lifespan(app: FastAPI):
    # app.state.summarisation_graph = None
    app.state.splunk_mcp = None
    app.state.chatbot_graph_with_splunk_mcp_dict = {}
    chatbot_graph_with_splunk_mcp_dict = {}
    current_chatbox_backend_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log = get_logger(userName="Chatbox_Container", userSession=current_chatbox_backend_time, base_dir=base_dir)
    log.info(f"\n========================== CONTAINER STARTUP: {current_chatbox_backend_time} =========================\n")
    print(f"\n========================== CONTAINER STARTUP: {current_chatbox_backend_time} =========================\n")
    
    try: 
        # summarisation_graph = build_summarisation_graph()
        # app.state.summarisation_graph = summarisation_graph
        
        app.state.splunk_mcp = SplunkMCPManager(
            url=SPLUNK_MCP_URL,
            token=SPLUNK_MCP_TOKEN,
        )
        log.info("Starting MCP Connection...")
        print("Starting MCP Connection...")

        tools = await app.state.splunk_mcp.connect(log=log)
        log.info("Success: MCP Connection!")
        print("Success: MCP Connection!")

        log.info("Starting Testing of LLM Clients...")
        print("Starting Testing of LLM Clients...")

        llm = "ollama" ## No actual meaning. Return full set of LLMs.
        llm_clients = await app.state.splunk_mcp.get_llm(tools=tools,llm=llm, log=log)
        app.state.llm_clients = llm_clients
        log.info("Success: LLM Clients Connected!")
        print("Success: LLM Clients Connected!")
        print(f"{llm_clients}")
        log.info(f"LLM Clients: {llm_clients}")
        log.info("Starting Langgraph buidling for Chatbot...")
        print("Starting Langgraph buidling for Chatbot...")
        for llm_name, value in llm_clients.items():
            try:
                log.info(f"Graph for {llm_name}...")
                print(f"Graph for {llm_name}...")
                chatbot_graph_with_splunk_mcp_dict[llm_name] = build_chatbot_graph(llm=value["llm_client"], llm_with_tools= value["llm_with_tools"],mcp_tools=tools, log=log)
                log.info(f"Graph for {llm_name} Success!")
                print(f"Graph for {llm_name} Success!")
            except Exception as e:
                log.info(f"Graph {llm_name} failed to build. Error: {e}")
                print(f"Graph {llm_name} failed to build. Error: {e}")
        app.state.chatbot_graph_with_splunk_mcp_dict = chatbot_graph_with_splunk_mcp_dict
        yield
        log.info("\n========================== CONTAINER Shutdown =========================\n")
    finally:
        if app.state.splunk_mcp is not None:
            await app.state.splunk_mcp.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_session_dict = {} ##"userSession": {"logs": [],"userName": userName,"convoHistory": [],"tool_call_flag": False,})
## Manage "last seen" for users
session_last_seen = {}

######################################################## HOUSE KEEPING FUNCTIONS ##########################################
## Function to manage history for token
def trim_history(hist):
    """Trims the conversation history after it goes past certain length of messages"""
    # keep system prompt at index 0, keep last MAX_MSGS-1 others
    if len(hist) > MAX_MSGS:
        print(f"Length of history: {len(hist)}")
        print(f"History: {hist}")
        return [hist[0]] + hist[-(MAX_MSGS-1):]
    return hist

## Function to manage TTL for each user
def touch_session(userSession: str) -> None:
    """Mark a session as recently used."""
    session_last_seen[userSession] = time.time()

## Function to clean_up_expired_sessions()
def cleanup_expired_sessions() -> int:
    now = time.time()
    expired = []
    for sid, last in session_last_seen.items():
        if now - last > TTL_EVICT_SECONDS:
            expired.append(sid)

    for sid in expired:
        session_last_seen.pop(sid, None)
        user_session_dict.pop(sid, None)

################################################################################################################
## Function to limit the number of tokens coming from logs
def count_tokens(logs_list, log):
    total_token_count = 0
    list_of_usable_logs = []
    for i in range(len(logs_list)):
        userRawLogs = logs_list[i]['_raw']
        len_of_log_char = len(userRawLogs) ## Count the number of characters
        print(f"log: {userRawLogs}")
        print(f"{i}: Word Count = {len_of_log_char}")
        token_count_for_curr_log = len_of_log_char/4 ## Assume that 1 token is 4 characters
        total_token_count += token_count_for_curr_log
        print(f"Total Token Count: {total_token_count}")
        if total_token_count > MAX_LOG_TOKEN_SIZE:
            log.info(f"Number of tokens exceeds > {MAX_LOG_TOKEN_SIZE}, will truncate the logs.")
            log.info(f"Trunchated Logs: {list_of_usable_logs}")
            return list_of_usable_logs
        else:
            list_of_usable_logs.append(logs_list[i])
    log.info(f"Number of tokens did not exceed {MAX_LOG_TOKEN_SIZE}.")
    log.info(f"Logs: {list_of_usable_logs}")
    return list_of_usable_logs

class ChatResponse(BaseModel):
    data: str
################################################################################################################


@app.get("/mcp/status")
def mcp_status(request: Request):
    mcp = request.app.state.splunk_mcp
    # implement is_connected() in your manager
    curr_llm_clients = app.state.llm_clients
    # print(f"Type of LLM: {curr_llm_clients}")
    ok =  mcp.is_connected
    detail = mcp.num_of_tools
    return {"connected": ok, "detail": f"{detail} tools", "llm_list": curr_llm_clients}

@app.post("/logReview")
async def logReview(req:Request):
    body = await req.json()
    userName = body.get("userName")
    userSession = body.get("sessionID")
    userRawData = body.get("logs", "")
    log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
    print("\n========================== FASTAPI from LogReview API endpoint =========================\n")
    log.info("\n========================== FASTAPI from LogReview API endpoint =========================\n")
    ## Reset TTL for each user session
    touch_session(userSession)
    cleanup_expired_sessions()  #Clean up expired sessions
    print(f"Request = {body}\n")
    print(f"User: {userName}\n")
    print(f"SessionID: {userSession}\n")
    print(f"Logs: {userRawData}\n")

    log.info(f"User(Frontend) Logs:\nUser: {userName}\nSessionID: {userSession}\nLogs{userRawData}\n")
    updated_logs_list = count_tokens(userRawData, log)

    print(f"Updated_logs_list: {updated_logs_list}\n Type of updated_logs_list: {type(updated_logs_list)}")
    log.info(f"Updated_logs_list: {updated_logs_list}\n Type of updated_logs_list: {type(updated_logs_list)}")
    append_Raw_Logs = ""

    for i in range(len(updated_logs_list)):
        userRawLogs = userRawData[i]['_raw']
        if "{" in userRawLogs:
            userRawLogs = userRawLogs.replace("{", "")
        if "}" in userRawLogs:
            userRawLogs = userRawLogs.replace("}", "")
        append_Raw_Logs = append_Raw_Logs + "\n" + userRawLogs
    append_Raw_Logs_to_LLM = "Raw Logs: \n" + append_Raw_Logs ## append_Raw_Logs_to_LLM is the finalised string to be used in prompts for logs.

    ## Setting up the history for each userSession
    # Tactic: Set up the conversation chain Human->AI for asking of approval for log summary.
    query = HumanMessage('Here is my log set from Splunk. ' + append_Raw_Logs_to_LLM) ## HumanMessage to be appended (Coversation chain for logs)
    summarise_logs_query = AIMessage('Would you like to summarise the logs? I will return a summary of what happened within the logs in less than 200 words.') ## AI Message to be appended after HumanMessage(Conversation Chain simulating the LLM asking for summary)

    # last_msg = final_state["final_output"]
    # print(f"\n\nOutput from LLM: {last_msg}")

    current_session = user_session_dict.setdefault(userSession, {
    "logs": [],
    "userName": userName,
    "convoHistory": [],
    "tool_call_flag": False,
    })
    # Update Logs History for current user
    current_session["logs"].append(updated_logs_list)
    # Update conversation history for current user
    current_session["convoHistory"].append(query)
    current_session["convoHistory"].append(summarise_logs_query)

    # Manage History Length
    # print(type(manage_session_states_history[userSession]))
    current_session["convoHistory"] = trim_history(current_session["convoHistory"])

    log.info(f"------------------------- Finalised Query -------------------------")
    print(f"Current Session Details: {current_session}\n")
    log.info(f"\nLogs!! HumanMessage: {query}\n")
    log.info(f"AIMessage: {summarise_logs_query}")
    log.info(f"------------------------- End Query -------------------------")

    print(f"\n\nCurrentSession Convo History: {current_session}")
    print("===========================================================================\n")
    log.info(f"===========================================================================\n")
    return ChatResponse(data=summarise_logs_query.content)



@app.post("/chatbox")
async def chat(req:Request):
    print("\n========================== FASTAPI from Chatbot Endpoint =========================\n")
    body = await req.json()
    userName = body.get("userName")
    userSession = body.get("sessionID")
    userMessage = body.get("message", "")
    llm_option = body.get("llmOption")
    # currentUser = body.get("currentUser", "default")
    
    ## Setup logger
    log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
    log.info(f"\n========================== FASTAPI from Chatbot Endpoint =========================\n")
    log.info(f"User(Frontend) Query:\nUser: {userName}\nSessionID: {userSession}\nMessage: {userMessage}\nLLM Option: {llm_option}")
    touch_session(userSession)
    cleanup_expired_sessions()
    print(f"Request = {body}\n")
    print(f"User: {userName}\n")
    print(f"SessionID: {userSession}\n")
    print(f"Type for User Message {type(userMessage)}")
    print(f"User Message: {userMessage}\n")
    print(f"LLM Option: {llm_option}")
    if llm_option == None or llm_option == "":
        log.error(f"LLM Option not selected! Informing user to select LLM!\n")
        log.info(f"===========================================================================\n")
        return ChatResponse(data="Please select your LLM Option!")
    else:
        print("===========================================================================\n")
        llm_client_graph = app.state.chatbot_graph_with_splunk_mcp_dict[llm_option]
        current_session = user_session_dict.setdefault(userSession, {
        "logs": [],
        "userName": userName,
        "convoHistory": [],
        "tool_call_flag": False,
        })
        log.info(f"LLM Graph Chosen: {llm_client_graph}\n")
        log.info(f"Current User Session Details: {current_session}\n")
        #Check if there is tool call approval in progress.
        if current_session["tool_call_flag"]:
            log.info(f"Current Session has tool approval in progress.\n")
            current_session["tool_call_flag"]=False
            final_state = await llm_client_graph.ainvoke(
                Command(resume=userMessage),
                config={"configurable": {"thread_id": userSession}},
            )
        else:
            log.info(f"Current Session has NO tool approval in progress.\n")
            state = {
                "messages": current_session["convoHistory"] + [HumanMessage(content=userMessage)],
                "summarisation_agent_bool": False,
                "solution_agent_bool": False,
                "final_output": ""
            }

            final_state = await llm_client_graph.ainvoke(
                state,
                config={"configurable": {"thread_id": userSession}}
            )
            interrupts = final_state.get("__interrupt__", [])
            log.info(f"LLM Agent decision on tool call or not: {interrupts}")
            print(f"Exited the interrupt: {interrupts}")
            if interrupts:
                current_session["tool_call_flag"]=True
                payload = interrupts[0].value
                payload_question=payload.get("question")
                payload_details = payload.get("details")
                tool_msg = payload_question + "\n" + payload_details
                log.info(f"Interrupted! Tool Message: {tool_msg}")
                print(f"\n\n\nINTERRUPTED!!!! Tool Message: {tool_msg}")
                return ChatResponse(data=tool_msg)
            else:
                log.info(f"No interrupts.")
                print(f"\n\n\nNot Interrupted \n\n\n")
                current_session["tool_call_flag"]=False
        log.info(f"Current User's Final state in graph: {final_state}")
        print(f"\n\nFinal State after seeking approval: {final_state}")
        llm_output = final_state["messages"][-1]
        message = llm_output.content
        print(f"\nFinal States: {final_state['messages']}\n")
        print(f"\n\nOutput from LLM: {message}\n\n")
        print(f"Type of last output: {type(message)}")

        current_session['convoHistory'] = final_state['messages']
        print("\n\n------------------------------------Convo History------------------------------------\n\n")
        print(f"{current_session['convoHistory']}")
        print("------------------------------------------------------------------------")
        return ChatResponse(data=message)
################################################################################################################
# -------------------------------------------------------------------------------
# GLOBAL variables 
# -------------------------------------------------------------------------------
app.Model = {}
app.NotebookDataPath = '/srv/notebooks/data/'
app.favicon_path = '/srv/app/static/favicon.ico'
app.data_path = '/srv/app/data/'
app.graphics_path = '/srv/app/graphics/'

@app.on_event("startup")
def setup_tracing():
    if 'olly_enabled' in os.environ:
        if 'true' in os.environ['olly_enabled'] or os.environ['olly_enabled'] == "1":
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from splunk_otel.tracing import start_tracing
            start_tracing()
            FastAPIInstrumentor.instrument_app(app)

# -------------------------------------------------------------------------------
# STATIC mounts
# -------------------------------------------------------------------------------
app.mount("/graphics", StaticFiles(directory=app.graphics_path), name="graphics")

# -------------------------------------------------------------------------------
# HELPER functions
# -------------------------------------------------------------------------------
# helper function: clean param
def get_clean_param(p):
    return p.lstrip('\"').rstrip('\"')

# perform token validation
def validate_token(header):
    token = ''
    # check if token is setup and defined correctly
    if header==None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No header with token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # check if token is setup and defined correctly
    if not 'api_token' in os.environ:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token setup",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = os.environ['api_token']
    if len(token)==0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token defined",
            headers={"WWW-Authenticate": "Bearer"},
        )    
    # check if token is matching
    if not token in header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # success if we did not fail and raise exceptions on the steps before
    return True

# -------------------------------------------------------------------------------
# MAIN APP enpoint definitions
# -------------------------------------------------------------------------------
# set an icon if requested
@app.get('/favicon.ico')
async def favicon():
    return FileResponse(app.favicon_path)

# -------------------------------------------------------------------------------
# get general information
@app.get('/')
def get_root(authorization: str = Header(None)):
    return get_summary(authorization)

# -------------------------------------------------------------------------------
# get model summary
@app.get('/summary')
def get_summary(authorization: str = Header(None)):
    return_object = {
        'app': 'Splunk App for Data Science and Deep Learning',
        'version': '5.2.3',
        'model': 'no model exists'
    }
    if validate_token(authorization):
        return_object["token"] = "valid"
        if "model" in app.Model:
            return_object["model"] = str(app.Model["model"])
            if "algo" in app.Model:
                return_object["model_summary"] = app.Model["algo"].summary(app.Model["model"])
    return json.dumps(return_object)


# -------------------------------------------------------------------------------
# fit endpoint 
# expects json object { 'data' : '<string of csv serialized pandas dataframe>', 'meta' : {<json dict object for parameters>}}
@app.post('/fit')
async def set_fit(request : Request, authorization: str = Header(None)):
    # prepare a return object
    response = {}
    response['status'] = 'error'
    response['message'] = '/fit: ERROR: '
    # 0. validate endpoint security token
    if not validate_token(authorization):
        response["message"] += 'unauthorized: invalid or missing token'
        return response

    # 1. validate input POST data
    try:
        dp = await request.json()
        #dp = request.get_json()
        app.Model["data"] = dp["data"]
        #print("/fit: raw data: ", str(app.Model["data"]))
        print("/fit: raw data size: ", len(str(app.Model["data"])))
        app.Model["meta"] = dp["meta"]
        print("/fit: meta info: ", str(app.Model["meta"]))
    except Exception as e:
        response["message"] += 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: ' + str(e)
        return response

    # 2. convert to dataframe 
    try:
        # TODO check with compression option and chunked mode
        try:
            from pd.compat import StringIO
        except ImportError:
            from io import StringIO
        app.Model["df"] = pd.read_csv(StringIO(app.Model["data"]))
        print("/fit: dataframe shape: ", str(app.Model["df"].shape))
        # free memory from raw data
        del(app.Model["data"])
        # memorize model name
        app.Model["model_name"] = "default"
        if "model_name" in app.Model["meta"]["options"]:
            app.Model["model_name"] = app.Model["meta"]["options"]["model_name"]
        print("/fit: model name: " + app.Model["model_name"])

    except Exception as e:
        response["message"] += 'unable to convert raw data to pandas dataframe. Ended with exception: ' + str(e)
        return response
    
    # 3. check for mode = stage and if so early exit the fit
    try:
        if "params" in app.Model["meta"]["options"]:
            params = app.Model["meta"]["options"]["params"]
            if 'mode' in params:
                # get, trim and lowercase the mode flag
                params_mode = get_clean_param(params["mode"])
                # if not production then just copy data and early exit
                if params_mode=="stage":
                    print("/fit: in staging mode: staging input dataframe for model (" + app.Model["model_name"] + ")")
                    path = app.NotebookDataPath+app.Model["model_name"]+'.csv'
                    app.Model["df"].to_csv(path, index=False)
                    path_json = app.NotebookDataPath+app.Model["model_name"]+'.json'
                    with open(path_json, 'w') as param_file:
                        json.dump(app.Model["meta"], param_file)
                    response["message"] = "Model data staged successfully in " + path + " - no model was built yet."
                    response["status"] = "staged"
                    return response
    except Exception as e:
        response["message"] += 'unable to stage dataframe. Ended with exception: ' + str(e)
        return response


    # 4. create and import module code
    try:
        if "algo" in app.Model["meta"]["options"]["params"]:
            algo_name = get_clean_param(app.Model["meta"]["options"]["params"]["algo"])
            if "algo" in app.Model:
                if app.Model["algo_name"] == algo_name:
                    reload(app.Model["algo"])
                    print("/fit: algo reloaded from module " + algo_name + ")")
                else:
                    del(app.Model["algo"])
                    app.Model["algo_name"] = algo_name
                    app.Model["algo"] = import_module("app.model." + algo_name)
                    print("/fit: algo with different name loaded from module " + algo_name + "")
            else:
                app.Model["algo"] = import_module("app.model." + algo_name)
                app.Model["algo_name"] = algo_name
                print("/fit: algo loaded from module " + algo_name + ": " + str(app.Model["algo"]))
            model_summary = app.Model["algo"].summary()
            if "version" in model_summary:
                print("/fit: version info: " + str(model_summary["version"]) + "")
            
    except Exception as e:
        response["message"] += 'unable to load algo code from module. Ended with exception: ' + str(e)
        return response

    # 5. init model from algo module
    try:
        app.Model["model"] = app.Model["algo"].init(app.Model["df"],app.Model["meta"])
        print("/fit: " + str(app.Model["model"]) + "")
        model_summary = app.Model["algo"].summary(app.Model["model"])
        if "summary" in model_summary:
            print("/fit: model summary: " + str(model_summary["summary"]) + "")
    except Exception as e:
        response["message"] += 'unable to initialize module. Ended with exception: ' + str(e)
        return response

    
    # 6. fit model 
    try:
        app.Model["fit_info"] = app.Model["algo"].fit(app.Model["model"], app.Model["df"], app.Model["meta"])
        print("/fit: " + str(app.Model["fit_info"]) + "")
    except Exception as e:
        response["message"] += 'unable to fit model. Ended with exception: ' + str(e)
        return response

    # 7. save model if into keyword is present with model_name
    try:
        name = "default"
        name = app.Model["algo_name"] + "_" + app.Model["model_name"]
        app.Model["algo"].save(app.Model["model"], name)
    except Exception as e:
        response["message"] += 'unable to save model. Ended with exception: ' + str(e)
        return response

    
    # 8. apply model 
    try:
        df_result = pd.DataFrame(app.Model["algo"].apply(app.Model["model"], app.Model["df"], app.Model["meta"]))
        print("/fit: returned result dataframe with shape " + str(df_result.shape) + "")
    except Exception as e:
        response["message"] += 'unable to apply model. Ended with exception: ' + str(e)
        return response
    
    response["results"] = df_result.to_csv(index=False)
    
    # end with a successful response
    response["status"] = "success"
    response["message"] = "/fit done successfully"
    return response


# -------------------------------------------------------------------------------
# fit routine 
# expects json object { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}
@app.post('/apply')
async def set_apply(request : Request, authorization: str = Header(None)):
    # prepare a return object
    response = {}
    response["status"] = "error"
    response["message"] = "/apply: ERROR: "
    # 0. validate endpoint security token
    if not validate_token(authorization):
        response["message"] += 'unauthorized: invalid or missing token'
        return response

    # 1. validate input POST data
    try:
        dp = await request.json()
        #dp = request.get_json()
        app.Model["data"] = dp["data"]
        print("/apply: raw data size: ", len(str(app.Model["data"])))
        app.Model["meta"] = dp["meta"]
        print("/apply: meta info: ", str(app.Model["meta"]))
    except Exception as e:
        response["message"] += 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: ' + str(e)
        return response

    # 2. convert to dataframe 
    try:
        # TODO check with compression option and chunked mode
        try:
            from pd.compat import StringIO
        except ImportError:
            from io import StringIO
        app.Model["df"] = pd.read_csv(StringIO(app.Model["data"]))
        print("/apply: dataframe shape: ", str(app.Model["df"].shape))
        # free memory from raw data
        del(app.Model["data"])
        # memorize model name
        app.Model["model_name"] = "default"
        if "model_name" in app.Model["meta"]["options"]:
            app.Model["model_name"] = app.Model["meta"]["options"]["model_name"]
        print("/apply: model name: " + app.Model["model_name"])

    except Exception as e:
        response["message"] += 'unable to convert raw data to pandas dataframe. Ended with exception: ' + str(e)
        return response

    if "algo" in app.Model:
        del(app.Model["algo"])
    
    
    # 3. check if model is initialized and load if not
    if "algo" not in app.Model:
        try:
            algo_name = get_clean_param(app.Model["meta"]["options"]["params"]["algo"])
            app.Model["algo_name"] = algo_name
            app.Model["algo"] = import_module("app.model." + algo_name)
            load_name = algo_name + "_" + app.Model["model_name"]
            app.Model["model"] = app.Model["algo"].load(load_name)
            print("/apply: loaded model: " + load_name)
            model_summary = app.Model["algo"].summary(app.Model["model"])
            if "summary" in model_summary:
                print("/apply: model summary: " + str(model_summary["summary"]) + "")
        except Exception as e:
            response["message"] += 'unable to initialize module. Ended with exception: ' + str(e)
            return response
    
    # 3. apply model
    if "algo" in app.Model:
        # TODO check if same algo and model name otherwise hard load by default
        try:
            df_result = pd.DataFrame(app.Model["algo"].apply(app.Model["model"], app.Model["df"], app.Model["meta"]))
            print("/apply: returned result dataframe with shape " + str(df_result.shape) + "")
        except Exception as e:
            response["message"] += 'unable to apply model. Ended with exception: ' + str(e)
            return response
    
    response["results"] = df_result.to_csv(index=False)
    
    # end with a successful response
    response["status"] = "success"
    response["message"] = "/apply done successfully"
    return response

# -------------------------------------------------------------------------------
# compute routine (experimental)
# expects json object { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}
@app.post('/compute')
async def set_compute(request : Request):
    # prepare a return object
    response = {}
    response["status"] = "error"
    response["message"] = "/compute: ERROR: "
    #print("This is a new compute function")
    # 1. validate input POST data
    try:
        dp = await request.json()
        #print("/compute: raw data: ", str(dp))
        app.Model["data"] = dp["data"]
        print("/compute: raw data type: ", str(type(app.Model["data"])))
        print("/compute: raw data size: ", len(str(app.Model["data"])))
        app.Model["meta"] = dp["meta"]
        print("/compute: meta info: ", str(app.Model["meta"]))

    except Exception as e:
        response["message"] += 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: ' + str(e)
        print("/compute: data input error: " + str(e))
        return response
    
    # 2. convert to dataframe 
    try:
        print("/compute: enter conversion block")
        # TODO check with compression option and chunked mode
        #app.Model["df"] = csv.DictReader(app.Model["data"], app.Model["meta"]["fieldnames"], dialect='unix', delimiter=',', quotechar='"')
        
        app.Model["df"] = app.Model["data"]
        #print("/compute: DictReader object: " + str(type(app.Model["df"])))
        #print("/compute: DictReader content: " + str(list(app.Model["df"])))

        del(app.Model["data"])
        # memorize model name
        app.Model["algo_name"] = app.Model["meta"]['algo']
        app.Model["algo"] = import_module("app.model." + app.Model["algo_name"])
        #if "model_name" in app.Model["meta"]["options"]:
        #    app.Model["model_name"] = app.Model["meta"]["options"]["model_name"]
        print("/compute: model name: " + app.Model["algo_name"])

    except Exception as e:
        response["message"] += 'unable to convert raw data to DictReader object. Ended with exception: ' + str(e)
        print("/compute: conversion error: " + str(e))
        return response
    
    df_result = app.Model["algo"].compute(None, app.Model["df"], app.Model["meta"])
    #print("Finished computation")
    response["results"] = json.dumps(df_result)
    
    # end with a successful response
    response["status"] = "success"
    response["message"] = "/compute done successfully"
    return response