# Deep Learning Toolkit for Splunk 3.8.0
# Author: Philipp Drieger, Principal Machine Learning Architect, 2018-2021
# -------------------------------------------------------------------------------

from fastapi import FastAPI, Request
from starlette.responses import FileResponse

from importlib import import_module, reload
import pandas as pd
import json
import os

app = FastAPI()

# -------------------------------------------------------------------------------
# GLOBAL variables 
# -------------------------------------------------------------------------------
app.Model = {}
app.NotebookDataPath = '/srv/notebooks/data/'
app.favicon_path = '/srv/app/static/favicon.ico'

@app.on_event("startup")
def setup_tracing():
    if 'olly_enabled' in os.environ:
        if 'true' in os.environ['olly_enabled'] or os.environ['olly_enabled'] == "1":
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from splunk_otel.tracing import start_tracing
            start_tracing()
            FastAPIInstrumentor.instrument_app(app)


# -------------------------------------------------------------------------------
# HELPER functions
# -------------------------------------------------------------------------------
# helper function: clean param
def get_clean_param(p):
    return p.lstrip('\"').rstrip('\"')
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
def get_root():
    return get_summary()

# -------------------------------------------------------------------------------
# get model summary
@app.get('/summary')
def get_summary():
    return_object = {
        'app': 'Deep Learning Toolkit for Splunk',
        'version': '3.8.0',
        'model': 'no model exists'
    }
    if "model" in app.Model:
        return_object["model"] = str(app.Model["model"])
        if "algo" in app.Model:
            return_object["model_summary"] = app.Model["algo"].summary(app.Model["model"])
    return json.dumps(return_object)


# -------------------------------------------------------------------------------
# fit endpoint 
# expects json object { 'data' : '<string of csv serialized pandas dataframe>', 'meta' : {<json dict object for parameters>}}
@app.post('/fit')
async def set_fit(request : Request):
    # prepare a return object
    response = {}
    response['status'] = 'error'
    response['message'] = '/fit: ERROR: '
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
async def set_apply(request : Request):
    # prepare a return object
    response = {}
    response["status"] = "error"
    response["message"] = "/apply: ERROR: "

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
            print("/fit: returned result dataframe with shape " + str(df_result.shape) + "")
        except Exception as e:
            response["message"] += 'unable to apply model. Ended with exception: ' + str(e)
            return response
    
    response["results"] = df_result.to_csv(index=False)
    
    # end with a successful response
    response["status"] = "success"
    response["message"] = "/apply done successfully"
    return response
