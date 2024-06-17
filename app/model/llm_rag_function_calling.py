#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import pymilvus
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, ServiceContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import textwrap
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.agent import FunctionCallingAgentWorker
from typing import Sequence, List
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import AgentRunner, ReActAgentWorker
from pydantic import Field
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"

def search_splunk_events(
    index: str, 
    sourcetype: str, 
    earliest_time: str, 
    latest_time: str, 
    source: str = None, 
    keyword: str =None
):
    '''
    Description on input fields
    earliest_time: Time specifier for earliest event to search, formatted like '[+|-]<time_integer><time_unit>@<time_unit>'. For example, '-12h@h' for the past 12 hours, '-5m@m' for the last 5 minutes and '-40s@s' for the last 40 seconds
    latest_time: Time specifier for latest event search, formatted like '[+|-]<time_integer><time_unit>@<time_unit>'. For example, '-12h@h' for the past 12 hours, '-5m@m' for the last 5 minutes and '-40s@s' for the last 40 seconds. For searching events up to now, set this field to 'now'
    '''
    # Imports
    import splunklib.client as splunk_client
    import splunklib.results as splunk_results
    import time 
    import pandas as pd
    # Load Splunk server info and create service
    token = os.environ["splunk_access_token"]
    host = os.environ["splunk_access_host"]
    port = os.environ["splunk_access_port"]
    service = splunk_client.connect(host=host, port=port, token=token)
    if index is not None:
        index = index
    else:
        index= ' *'
    if source is not None:
        source = source
    else:
        source = ' *'
    if sourcetype is not None:
        sourcetype = sourcetype
    else:
        sourcetype = ' *'
    if keyword is not None:
        keyword = keyword
    else:
        keyword = ' *'
    if earliest_time is not None:
        earliest = earliest_time
    else:
        earliest = '-24h@h'
    if latest_time is not None:
        latest = latest_time
    else:
        latest = "now"

    query = f"index={index} sourcetype={sourcetype} source={source} {keyword} earliest={earliest} latest={latest}"
    query_cleaned = query.strip()
    # add search keyword before the SPL
    query_cleaned="search "+query_cleaned
    
    job = service.jobs.create(
        query_cleaned,
        earliest_time=earliest, 
        latest_time=latest, 
        adhoc_search_level="smart",
        search_mode="normal")
    while not job.is_done():
        time.sleep(0.1)
    resultCount = int(job.resultCount)
    diagnostic_messages = []
    resultset = []
    processed = 0
    offset = 0
    while processed < resultCount:
        for event in splunk_results.JSONResultsReader(job.results(output_mode='json', offset=offset, count=0)):
            if isinstance(event, splunk_results.Message):
                # Diagnostic messages may be returned in the results
                diagnostic_messages.append(event.message)
                #print('%s: %s' % (event.type, event.message))
            elif isinstance(event, dict):
                # Normal events are returned as dicts
                resultset.append(event['_raw'])
                #print(result)
            processed += 1
        offset = processed   
    results = f'The list of events searched from Splunk is {str(resultset)}'
    return results

# Milvus search function
def search_record_from_vector_db(log_message: str, collection_name: str):
    from pymilvus import connections, Collection
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    transformer_embedder = HuggingFaceEmbedding(model_name='all-MiniLM-L6-v2')
    connections.connect("default", host="milvus-standalone", port="19530")
    collection = Collection(collection_name)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    log_message = transformer_embedder.get_text_embedding(log_message)
    results = collection.search(data=[log_message], anns_field="embeddings", param=search_params, limit=1, output_fields=["_key","label"])
    l = []
    for result in results:
        t = ""
        for r in result:
            t += f"For the log message {log_message}, the recorded similar log message is: {r.entity.get('label')}."
        l.append(t)
    return l[0]

search_splunk_tool = FunctionTool.from_defaults(fn=search_splunk_events)
search_record_from_vector_db_tool = FunctionTool.from_defaults(fn=search_record_from_vector_db)







    
# In[6]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param





    
# In[10]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['hyperparameter'] = 42.0
    
    return model





    
# In[19]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    info = {"message": "model trained"}
    return info









    
# In[6]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Example: 'all-MiniLM-L6-v2'
    query = param['options']['params']['query'].strip('\"')
    # Case of only two functions
    try:
        func1 = int(param['options']['params']['func1'])
        func2 = int(param['options']['params']['func2'])
    except:
        func1 = 1
        func2 = 1
    tool_list = []

    if func1:
        tool_list.append(search_splunk_tool)
    if func2:
        tool_list.append(search_record_from_vector_db_tool)
    
    try:
        model = param['options']['params']['model_name'].strip('\"')
    except:
        model="mistral"
    
    url = "http://ollama:11434"
    llm = Ollama(model=model, base_url=url, request_timeout=6000.0)

    
    worker = ReActAgentWorker.from_tools(tool_list, llm=llm)
    agent = AgentRunner(worker)     
    response = agent.chat(query)
    
    cols = {"Response": [response.response]}
    for i in range(len(response.sources)):
        if response.sources[i].tool_name != "unknown":
            cols[response.sources[i].tool_name] = [response.sources[i].content]
    result = pd.DataFrame(data=cols)
    return result

















    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        json.dump(model, file)
    return model





    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model = json.load(file)
    return model





    
# In[18]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

















