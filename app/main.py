# Deep Learning Toolkit for Splunk 5.2.0
# Author: Philipp Drieger, Principal Machine Learning Architect, 2018-2024
# -------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from app.model.llm_utils_chat import create_llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from dotenv import load_dotenv
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
from app.libraries.logging_function import get_logger

app = FastAPI()

# CORS for Splunk JavaScript
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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



LLM_LIST = ['ollama', 'bedrock', 'azure_openai', 'openai', 'gemini']
llm_clients = {}

## manage_session_states_logs={userSession:[(Summary_of_Log_flag, raw logs, summary_of_log)]}
manage_session_states_logs = {}
## Manage_session_states_history = {userSession: [{"role": "user", "content": user_query}]}
manage_session_states_history = {}
## Manage "last seen" for users
session_last_seen = {}


for LLM in LLM_LIST:
    try:
        client, msg = create_llm(service=LLM)
        print(msg)
        llm_clients[f"llm_client_{LLM}"] = client
    except Exception as e:
        llm_clients[f"llm_client_{LLM}"] = None

## Base Directory for Logging
base_dir = os.getcwd()

######################################################## HOUSE KEEPING FUNCTIONS #############################################################
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
        manage_session_states_history.pop(sid, None)

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

##############################################################################################################################################

######################################################## Endpoints and Functional Code ########################################################

## API endpoint for the Logs in the SPL to be recorded.
@app.post("/logReview")
async def logReview(req:Request):
    try:
        ## Getting the data from the API call.
        body = await req.json()
        # print(f"/logReview body: {body}")
        userName = body.get("userName")
        userSession = body.get("sessionID")
        userRawData = body.get("logs", "")
        # userRawData = json.dumps(temp_logs, indent=2)
        # print(f"{type(temp_logs)}")
        ## Setup logger
        log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
        touch_session(userSession)
        cleanup_expired_sessions()
        ## Concatenating all the raw logs together into a string and removing curly braces.
        append_Raw_Logs = ""
        userRawData = count_tokens(userRawData, log)
        for i in range(len(userRawData)):
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
        summarise_logs_query = AIMessage(f'I have received {len(userRawData)} lines of logs. Would you like to summarise the logs? I will return a summary of what happened within the logs in less than 200 words.') ## AI Message to be appended after HumanMessage(Conversation Chain simulating the LLM asking for summary)

        # Check if it is user's first time having the conversation.
        # if userSession in manage_session_states_logs: # Not user's first time
        #     manage_session_states_logs[userSession].append((False, append_Raw_Logs_to_LLM, llm_summary)) # Manage history of Splunk Results(logs)
        if userSession not in manage_session_states_history: # If it is user's first time
            manage_session_states_history[userSession] = []
            manage_session_states_history[userSession].append(system_prompt_chat) # Add in the system prompt
            manage_session_states_history[userSession].append(query) # Add in the human message about the raw logs
            manage_session_states_history[userSession].append(summarise_logs_query) # Add in the AI message about the logs
        # Not the user's first message        
        else:  
            manage_session_states_history[userSession].append(query) # Append the Human message about the raw logs
            manage_session_states_history[userSession].append(summarise_logs_query) # Append the AI message about the logs.

        # Manage History Length
        # print(type(manage_session_states_history[userSession]))
        manage_session_states_history[userSession] = trim_history(manage_session_states_history[userSession])

        # print("############################# Summary of user session's list of logs ##############################")
        # print(f"User session Number: {userSession} \n")
        # print(f"Number of Raw Logs: {len(userRawData)}")
        # print(f"Conversation history of {userSession}: {manage_session_states_history[userSession]}")
        # print(f"\nAI message returning back to user: {summarise_logs_query.content}")
        log.info(f"------------------------- Received Logs -------------------------")
        log.info(f"\nLogs!! HumanMessage: {query}\n")
        log.info(f"AIMessage: {summarise_logs_query}")
        log.info(f"------------------------- End Logs -------------------------")

        return {"status": "received", "data": summarise_logs_query.content}
    except Exception as e:
        print("Error:", str(e))
        log.error(f"Error in /logReview: {str(e)}\n")
        return {"status": "error", "error": str(e)}
    
## Function to organise the LLM prompt for invocation.
async def chatbox_callLLM(human_msg, userSession, llm_option):
    # print("################################################################ LLM Initialised ##############################################################")
    manage_session_states_history[userSession].append(human_msg) # Append the Userquery (Human Message) into the history
    history = manage_session_states_history[userSession] # Extract the full Conversation History to be used as the prompt
    # print(f"Final prompt to LLM: {history}")
    try:
        llm_client = llm_clients[f"llm_client_{llm_option}"]

        if llm_client is None:
            raise RuntimeError(f"SYSTEM: LLM option {llm_option} has not been successfully configured. Please try another LLM option.")
        print(f"LLM client {llm_option} is available")
    except Exception as e:
        raise RuntimeError(f"SYSTEM: Failed to initiate LLM option {llm_option}. The following error occurred: {e}.  Please switch to a different LLM. ")
    result = await llm_client.ainvoke(history) # Send the Conversation History to the Client.
    # print("################################################################ LLM Reply ##############################################################")
    # print(f"LLM Reply: {result}")
    # print("###############################################################User History###############################################################")
    manage_session_states_history[userSession].append(result) # Append AI's reply to the Conversation History
    # Manage History Length
    # print(type(manage_session_states_history[userSession]))
    manage_session_states_history[userSession] = trim_history(manage_session_states_history[userSession])

    # print(f"{manage_session_states_history[userSession]}")
    return result


## Endpoint for the chatbox
@app.post("/chatbox")
async def chat(req:Request):
    try:
        body = await req.json()
        # print(f"Request body: {body}")
        userName = body.get("userName")
        userSession = body.get("sessionID")
        user_query = body.get("message", "")
        llm_option = body.get("llmOption", DEFAULT_LLM)

        ## Setup logger
        log = get_logger(userName=userName, userSession=userSession, base_dir=base_dir)
        touch_session(userSession)
        cleanup_expired_sessions()
        ## Check if user history exists. If not, templatise it.
        if userSession not in manage_session_states_history:
            # first_human_msg = HumanMessage(user_query)
            manage_session_states_history[userSession] = []
            # Append system prompt first
            manage_session_states_history[userSession].append(system_prompt_chat)
        # print("------------------------------- START of DATA RETRIEVAL:Chatbox API Call -------------------------------")
        # print(f"\nUser's Name: {userName}\n")
        # print(f"Current user session ID: {userSession}\n")
        # print(f"Current user query: {user_query}\n")
        # print("------------------------------- END of DATA RETRIEVAL:Chatbox API Call -------------------------------")
        # print("################################################################ Enter AI Function ##############################################################")
        human_msg = HumanMessage(user_query) # Convert it into the langchain HumanMessage template for prompt.
        ai_response = await chatbox_callLLM(human_msg, userSession, llm_option)

        log.info("------------------------------- Start: Chatbox API Call -------------------------------")
        log.info(f"LLM option: {llm_option}")
        log.info(f"HumanMessage: {human_msg}")
        log.info(f"AIMessage: {ai_response}")
        log.info("------------------------------- End:Chatbox API Call -------------------------------")


        return {"status": "received", "data": ai_response.content}

    except Exception as e:
        print(f"Error in CHATBOX: {e}")
        log.error(f"Error in /chatbox endpoint: {str(e)}\n")
        return {"status": "error", "data": "Problem occured in chatbox endpoint"}
    
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
        'version': '5.2.0',
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

