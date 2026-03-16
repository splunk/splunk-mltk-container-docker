# Splunk App for Data Science and Deep Learning 5.2.3 (With Custom Memory Optimization)
# Creator: Philipp Drieger, Principal AI Architect
# Authors: Huaibo Zhao
# modified: Guillaume Radecki
# 2018-2026
# -------------------------------------------------------------------------------

from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, ORJSONResponse
from app.model.llm_utils_chat import create_llm
try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    HAS_langchain=True
except ImportError: HAS_langchain=False

from importlib import import_module, reload
import pandas as pd
import orjson
# import csv
import os
import time
# import uvicorn
from app.libraries.logging_function import get_logger
import gc  # Garbage Collection for memory management

# Initialize FastAPI with ORJSONResponse for faster serialization
app = FastAPI(default_response_class=ORJSONResponse)

# CORS for Splunk JavaScript
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables configuration
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

if HAS_langchain:
    try:
        SYSTEM_PROMPT = os.environ['SYSTEM_PROMPT'].strip('"')
        if len(SYSTEM_PROMPT) > 0:
            system_prompt_chat = SystemMessage(content=SYSTEM_PROMPT)
        else:
            system_prompt_chat = SystemMessage(
                '''You are a friendly chatbot that is well-verse in Splunk and logs. You are here to help people ''')
    except:
        system_prompt_chat = SystemMessage(
            '''You are a friendly chatbot that is well-verse in Splunk and logs. You are here to help people ''')
else:
    print("WARNING: longchain_core is not installed; chat features is disabled.")


LLM_LIST = ['ollama', 'bedrock', 'azure_openai', 'openai', 'gemini']
llm_clients = {}

## manage_session_states_logs={userSession:[(Summary_of_Log_flag, raw logs, summary_of_log)]}
manage_session_states_logs = {}
## Manage_session_states_history = {userSession: [{"role": "user", "content": user_query}]}
manage_session_states_history = {}
## Manage "last seen" for users
session_last_seen = {}

# Initialize LLM Clients
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
        return [hist[0]] + hist[-(MAX_MSGS - 1):]
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


def count_tokens_vectorized(logs_list, log):
    """
    Optimized token counting using Pandas vectorization.
    Assumes 1 token ≈ 4 characters.
    """
    if not logs_list:
        return []

    df = pd.DataFrame(logs_list)
    # Calculate character lengths for all logs at once (CPU optimized)
    lengths = df['_raw'].str.len()
    cumulative_tokens = lengths.cumsum() / 4

    # Find the index where we exceed the limit
    mask = cumulative_tokens <= MAX_LOG_TOKEN_SIZE
    truncated_df = df[mask]

    if len(truncated_df) < len(df):
        log.info(f"Token limit exceeded. Truncated {len(df) - len(truncated_df)} logs.")

    return truncated_df.to_dict('records')


##############################################################################################################################################

######################################################## Endpoints and Functional Code ########################################################

## API endpoint for the Logs in the SPL to be recorded.
@app.post("/logReview")
@app.post("/logReview")
async def logReview(req: Request):
    if not HAS_langchain:
        error_msg="ERROR: longchain_core is not installed; chat features is disabled."
        print(error_msg)
        return {"status": "error", "error": error_msg}

    try:
         body = orjson.loads(await req.body())
         userSession = body.get("sessionID")
         userRawData = body.get("logs", [])
         log = get_logger(userName=body.get("userName"), userSession=userSession, base_dir=base_dir)
         touch_session(userSession)
         cleanup_expired_sessions()
         # 1. Vectorized token counting
         userRawData = count_tokens_vectorized(userRawData, log)
         # 2. Vectorized cleaning: Use regex to remove both { and } in one pass
         df_temp = pd.DataFrame(userRawData)
         if not df_temp.empty:
             # CPU Optimization: Combined regex replace is faster than two separate calls
             cleaned_series = df_temp['_raw'].str.replace(r'[{}]', '', regex=True)
             append_Raw_Logs = "\n".join(cleaned_series.tolist())
         else:
             append_Raw_Logs = ""
         append_Raw_Logs_to_LLM = f"Raw Logs: \n{append_Raw_Logs}"
         query = HumanMessage(f"Here is my log set from Splunk. {append_Raw_Logs_to_LLM}")
         summarise_logs_query = AIMessage(f"I received {len(userRawData)} lines. Would you like a summary?")
         # Session history management
         if userSession not in manage_session_states_history:
             manage_session_states_history[userSession] = [system_prompt_chat, query, summarise_logs_query]
         else:
             manage_session_states_history[userSession].extend([query, summarise_logs_query])
         manage_session_states_history[userSession] = trim_history(manage_session_states_history[userSession])
         log.info(f"------------------------- Received Logs -------------------------")
         log.info(f"\nLogs!! HumanMessage: {query}\n")
         log.info(f"AIMessage: {summarise_logs_query}")
         log.info(f"------------------------- End Logs -------------------------")
         # Cleanup local heavy objects
         del df_temp
         gc.collect()
         return {"status": "received", "data": summarise_logs_query.content}
    except Exception as e:
         print("Error:", str(e))
         if log:
             log.error(f"Error in /logReview: {str(e)}\n")
         return {"status": "error", "error": str(e)}

## Function to organise the LLM prompt for invocation.
async def chatbox_callLLM(human_msg, userSession, llm_option):
    # print("################################################################ LLM Initialised ##############################################################")
    manage_session_states_history[userSession].append(
        human_msg)  # Append the Userquery (Human Message) into the history
    history = manage_session_states_history[
        userSession]  # Extract the full Conversation History to be used as the prompt
    # print(f"Final prompt to LLM: {history}")
    try:
        llm_client = llm_clients[f"llm_client_{llm_option}"]
        if llm_client is None:
            raise RuntimeError(
                f"SYSTEM: LLM option {llm_option} has not been successfully configured. Please try another LLM option.")
        print(f"LLM client {llm_option} is available")
    except Exception as e:
        raise RuntimeError(f"SYSTEM: Failed to initiate LLM option {llm_option}. Error: {e}")

    result = await llm_client.ainvoke(history)
    manage_session_states_history[userSession].append(result)
    manage_session_states_history[userSession] = trim_history(manage_session_states_history[userSession])

    # MEMORY OPTIMIZATION: Free memory after LLM call
    gc.collect()
    return result


## Endpoint for the chatbox
@app.post("/chatbox")
async def chat(req: Request):
    if not HAS_langchain:
        error_msg = "ERROR: longchain_core is not installed; chat features is disabled."
        print(error_msg)
        return {"status": "error", "error": error_msg}
    try:
        # MEMORY/CPU: Faster JSON parsing
        body = orjson.loads(await req.body())
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
        human_msg = HumanMessage(user_query)  # Convert it into the langchain HumanMessage template for prompt.
        ai_response = await chatbox_callLLM(human_msg, userSession, llm_option)
        log.info("------------------------------- Start: Chatbox API Call -------------------------------")
        log.info(f"LLM option: {llm_option}")
        log.info(f"HumanMessage: {human_msg}")
        log.info(f"AIMessage: {ai_response}")
        log.info("------------------------------- End:Chatbox API Call -------------------------------")
        # MEMORY OPTIMIZATION
        del body
        gc.collect()
        return {"status": "received", "data": ai_response.content}
    except Exception as e:
        print(f"Error in CHATBOX: {e}")
        if log:
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

import sys
import gc

def cleanup_memory():
    """
    Force to clean memory (RAM and VRAM GPU) 
    TensorFlow and PyTorch
    """
    # 2. Clean PyTorch GPU
    if 'torch' in sys.modules:
        try:
            torch = sys.modules['torch']
            # Clean VRAM cache GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            # Clean VRAM on  Mac Apple Silicon (M1/M2/M3)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"ERROR  clean PyTorch memory: {e}")
    # 3. Clean TensorFlow / Keras 
    if 'tensorflow' in sys.modules:
        try:
            tf = sys.modules['tensorflow']
            # Remove graph in RAM/VRAM
            tf.keras.backend.clear_session()
        except Exception as e:
            print(f"ERROR clean TensorFlow memory: {e}")
    # 4 Del all no use var in RAM
    gc.collect()


# perform token validation
def validate_token(header):
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
    # CPU/RAM: Using orjson for serialization
    return ORJSONResponse(return_object)


# -------------------------------------------------------------------------------
# fit endpoint
# expects json object { 'data' : '<string of csv serialized pandas dataframe>', 'meta' : {<json dict object for parameters>}}
@app.post('/fit')
async def set_fit(request: Request, authorization: str = Header(None)):
    response = {'status': 'error', 'message': '/fit: ERROR: '}
    # 0. validate endpoint security token
    if not validate_token(authorization):
        response["message"] += 'unauthorized: invalid or missing token'
        return response

    error_msg =""
    try:
        # 1. validate input POST data
        error_msg ='unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: '
        # Use orjson for heavy payload parsing
        dp = orjson.loads(await request.body())
        raw = dp["data"]
        app.Model["meta"] = dp["meta"]
        del dp  # Free request body memory


        # 2. convert to dataframe
        error_msg='unable to convert raw data to pandas dataframe. Ended with exception: '
        from io import StringIO
        # CPU/MEMORY: Read CSV and immediately delete the raw string
        df_local = pd.read_csv(StringIO(raw))
        print("/fit: dataframe shape: ", str(df_local.shape))
        # free memory from raw data
        del raw
        # memorize model name
        app.Model["model_name"] = app.Model["meta"]["options"].get("model_name", "default")
        print("/fit: model name: " + app.Model["model_name"])


        # 3. check for mode = stage and if so early exit the fit
        error_msg='unable to stage dataframe. Ended with exception: '
        if "params" in app.Model["meta"]["options"]:
            params = app.Model["meta"]["options"]["params"]
            if get_clean_param(params.get('mode', '')) == "stage":
                print("/fit: in staging mode: staging input dataframe for model (" + app.Model["model_name"] + ")")
                path = f"{app.NotebookDataPath}{app.Model['model_name']}.csv"
                df_local.to_csv(path, index=False)
                path_json = f"{app.NotebookDataPath}{app.Model['model_name']}.json"
                with open(path_json, 'wb') as f:
                    f.write(orjson.dumps(app.Model["meta"]))
                response["message"] = "Model data staged successfully in " + path + " - no model was built yet."
                response["status"] = "staged"
                return response


        # 4. create and import module code
        error_msg='unable to load algo code from module. Ended with exception: '
        if "algo" in app.Model["meta"]["options"]["params"]:
            algo_name = get_clean_param(app.Model["meta"]["options"]["params"]["algo"])
            if "algo" in app.Model and app.Model.get("algo_name") == algo_name:
                reload(app.Model["algo"])
                print("/fit: algo reloaded from module " + algo_name + ")")
            else:
                del (app.Model["algo"])
                app.Model["algo_name"] = algo_name
                app.Model["algo"] = import_module("app.model." + algo_name)
                print("/fit: algo loaded from module " + algo_name + ": " + str(app.Model["algo"]))
            model_summary = app.Model["algo"].summary()
            if "version" in model_summary:
                print("/fit: version info: " + str(model_summary["version"]) + "")


        # 5. init model from algo module
        error_msg = 'unable to initialize module. Ended with exception: '
        app.Model["model"] = app.Model["algo"].init(df_local, app.Model["meta"])
        print("/fit: " + str(app.Model["model"]) + "")
        model_summary = app.Model["algo"].summary(app.Model["model"])
        if "summary" in model_summary:
            print("/fit: model summary: " + str(model_summary["summary"]) + "")


        # 6. fit model
        error_msg='unable to fit model. Ended with exception: '
        app.Model["fit_info"] = app.Model["algo"].fit(app.Model["model"], df_local, app.Model["meta"])
        print("/fit: " + str(app.Model["fit_info"]) + "")


        # 7. save model if into keyword is present with model_name
        error_msg='unable to save model. Ended with exception: '
        name = f"{app.Model['algo_name']}_{app.Model['model_name']}"
        app.Model["algo"].save(app.Model["model"], name)


        # Apply and cleanup
        error_msg='unable to apply model. Ended with exception: '
        df_result = pd.DataFrame(app.Model["algo"].apply(app.Model["model"], df_local, app.Model["meta"]))
        response["results"] = df_result.to_csv(index=False)
        del df_local, df_result


        # end with a successful response
        response['status']= 'success'
        response['message']= '/fit done successfully'
        gc.collect()  # Final memory sweep
        return response
    
    except Exception as e:
        response["message"] += error_msg + str(e)
        return response
    
    finally:
        cleanup_memory()


# -------------------------------------------------------------------------------
# fit routine
# expects json object { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}
@app.post('/apply')
async def set_apply(request: Request, authorization: str = Header(None)):
    response = {"status": "error", "message": "/apply: ERROR: "}
    # 0. validate endpoint security token
    if not validate_token(authorization):
        response["message"] += 'unauthorized: invalid or missing token'
        return response

    error_msg =""
    try:
        # 1. validate input POST data
        error_msg='unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: '
        dp = orjson.loads(await request.body())
        raw = dp["data"]
        print("/apply: raw data size: ", len(str(raw)))
        app.Model["meta"] = dp["meta"]
        print("/apply: meta info: ", str(app.Model["meta"]))


        # 2. convert to dataframe
        error_msg='unable to convert raw data to pandas dataframe. Ended with exception: '
        try:
            from pd.compat import StringIO
        except ImportError:
            from io import StringIO
        df_local = pd.read_csv(StringIO(raw))
        print("/apply: dataframe shape: ", str(df_local.shape))
        # free memory from raw data
        del raw
        # memorize model name
        app.Model["model_name"] = app.Model["meta"]["options"].get("model_name", "default")

        if "algo" in app.Model:
            del (app.Model["algo"])


        # 3. check if model is initialized and load if not
        error_msg='unable to initialize module. Ended with exception: '
        if "algo" not in app.Model:
            algo_name = get_clean_param(app.Model["meta"]["options"]["params"]["algo"])
            app.Model["algo_name"] = algo_name
            app.Model["algo"] = import_module("app.model." + algo_name)
            load_name = f"{algo_name}_{app.Model['model_name']}"
            app.Model["model"] = app.Model["algo"].load(load_name)
            print("/apply: loaded model: " + load_name)
            model_summary = app.Model["algo"].summary(app.Model["model"])
            if "summary" in model_summary:
                print("/apply: model summary: " + str(model_summary["summary"]) + "")


        # 3. apply model
        error_msg='unable to apply model. Ended with exception: '
        if "algo" in app.Model:
            df_result = pd.DataFrame(app.Model["algo"].apply(app.Model["model"], df_local, app.Model["meta"]))
            response["results"] = df_result.to_csv(index=False)
            response["status"] = "success"
            response["message"] = "/apply done successfully"
            print("/apply: returned result dataframe with shape " + str(df_result.shape) + "")
            del df_result, df_local


        response['status']= 'success'
        response['message']= '/apply done successfully'
    
        return response
    
    except Exception as e:
        response["message"] += error_msg + str(e)
        return response
    
    finally:
        cleanup_memory()

# -------------------------------------------------------------------------------
# compute routine (experimental)
# expects json object { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}
@app.post('/compute')
async def set_compute(request: Request):
    response = {"status": "error", "message": "/compute: ERROR: "}

    error_msg =""
    try:
        # 1. validate input POST data
        error_msg= 'unable to parse json from POST data. Provide a JSON object with structure { "data" : "<string of csv serialized pandas dataframe>", "meta" : {<json dict object for parameters>}}. Ended with exception: '
        dp = orjson.loads(await request.body())
        raw = dp["data"]
        print("/compute: raw data type: ", str(type(raw)))
        print("/compute: raw data size: ", len(str(raw)))
        app.Model["meta"] = dp["meta"]
        print("/compute: meta info: ", str(app.Model["meta"]))


        # 2. convert to dataframe
        error_msg='unable to convert raw data to DictReader object. Ended with exception: '
        df_local = raw
        del raw
        # memorize model name
        app.Model["algo_name"] = app.Model["meta"]['algo']
        app.Model["algo"] = import_module("app.model." + app.Model["algo_name"])
        # if "model_name" in app.Model["meta"]["options"]:
        #    app.Model["model_name"] = app.Model["meta"]["options"]["model_name"]
        print("/compute: model name: " + app.Model["algo_name"])


        # CPU/RAM: Direct computation with minimal object copying
        df_result = app.Model["algo"].compute(None, df_local, app.Model["meta"])
        response["results"] = orjson.dumps(df_result)
        response['status']= 'success'
        response['message']= '/compute done successfully'
        del df_result, df_local

        return response

    except Exception as e:
        response["message"] += error_msg+ str(e)
        print("/compute: conversion error: " + str(e))
        return response
    
    finally:
        cleanup_memory()
