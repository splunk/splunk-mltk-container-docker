#!/usr/bin/env python
# coding: utf-8


    
# In[50]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Union
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
import textwrap
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.agent import FunctionCallingAgentWorker
from typing import Sequence, List
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import AgentRunner, ReActAgentWorker
from pydantic import Field
from app.model.llm_utils import create_llm, create_embedding_model
import splunklib.client
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"

## Acknowledgement: The example tools are following the Splunk MCP at https://github.com/livehybrid/splunk-mcp

def get_splunk_connection() -> splunklib.client.Service:
    """
    Get a connection to the Splunk service.
    
    Returns:
        splunklib.client.Service: Connected Splunk service
    """
    try:
        print(f"🔌 Connecting to Splunk")
        
        # Connect to Splunk
        service = splunklib.client.connect(
            host=os.environ["splunk_access_host"],
            port=os.environ["splunk_access_port"],
            token=os.environ["splunk_access_token"],
            scheme="https",
            verify=False
        )
        
        print(f"Connected to Splunk successfully")
        return service
    except Exception as e:
        print(f"Failed to connect to Splunk: {str(e)}")
        raise

def search_splunk(search_query: str, earliest_time: str = "-24h", latest_time: str = "now", max_results: int = 100) -> List[Dict[str, Any]]:
    """
    Execute a Splunk search query and return the results.
    
    Args:
        search_query: The search query to execute
        earliest_time: Start time for the search (default: 24 hours ago)
        latest_time: End time for the search (default: now)
        max_results: Maximum number of results to return (default: 100)
        
    Returns:
        List of search results
    """
    if not search_query:
        raise ValueError("Search query cannot be empty")
        
    try:
        service = get_splunk_connection()
        print(f"🔍 Executing search: {search_query}")
        
        # Create the search job
        kwargs_search = {
            "earliest_time": earliest_time,
            "latest_time": latest_time,
            "preview": False,
            "exec_mode": "blocking"
        }
        
        job = service.jobs.create(search_query, **kwargs_search)
        
        # Get the results
        result_stream = job.results(output_mode='json', count=max_results)
        results_data = json.loads(result_stream.read().decode('utf-8'))
        
        return results_data.get("results", [])
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        raise

def list_indexes() -> Dict[str, List[str]]:
    """
    Get a list of all available Splunk indexes.
    
    Returns:
        Dictionary containing list of indexes
    """
    try:
        service = get_splunk_connection()
        indexes = [index.name for index in service.indexes]
        print(f"📊 Found {len(indexes)} indexes")
        return {"indexes": indexes}
    except Exception as e:
        print(f"❌ Failed to list indexes: {str(e)}")
        raise

def get_index_info(index_name: str) -> Dict[str, Any]:
    """
    Get metadata for a specific Splunk index.
    
    Args:
        index_name: Name of the index to get metadata for
        
    Returns:
        Dictionary containing index metadata
    """
    try:
        service = get_splunk_connection()
        index = service.indexes[index_name]
        
        return {
            "name": index_name,
            "total_event_count": str(index["totalEventCount"]),
            "current_size": str(index["currentDBSizeMB"]),
            "max_size": str(index["maxTotalDataSizeMB"]),
            "min_time": str(index["minTime"]),
            "max_time": str(index["maxTime"])
        }
    except KeyError:
        print(f"❌ Index not found: {index_name}")
        raise ValueError(f"Index not found: {index_name}")
    except Exception as e:
        print(f"❌ Failed to get index info: {str(e)}")
        raise

def list_saved_searches() -> List[Dict[str, Any]]:
    """
    List all saved searches in Splunk
    
    Returns:
        List of saved searches with their names, descriptions, and search queries
    """
    try:
        service = get_splunk_connection()
        saved_searches = []
        
        for saved_search in service.saved_searches:
            try:
                saved_searches.append({
                    "name": saved_search.name,
                    "description": saved_search.description or "",
                    "search": saved_search.search
                })
            except Exception as e:
                print(f"⚠️ Error processing saved search: {str(e)}")
                continue
            
        return saved_searches
        
    except Exception as e:
        print(f"❌ Failed to list saved searches: {str(e)}")
        raise

def list_users() -> List[Dict[str, Any]]:
    """List all Splunk users (requires admin privileges)"""
    try:
        service = get_splunk_connection()
        print("👥 Fetching Splunk users...")
                
        users = []
        for user in service.users:
            try:
                if hasattr(user, 'content'):
                    # Ensure roles is a list
                    roles = user.content.get('roles', [])
                    if roles is None:
                        roles = []
                    elif isinstance(roles, str):
                        roles = [roles]
                    
                    # Ensure capabilities is a list
                    capabilities = user.content.get('capabilities', [])
                    if capabilities is None:
                        capabilities = []
                    elif isinstance(capabilities, str):
                        capabilities = [capabilities]
                    
                    user_info = {
                        "username": user.name,
                        "real_name": user.content.get('realname', "N/A") or "N/A",
                        "email": user.content.get('email', "N/A") or "N/A",
                        "roles": roles,
                        "capabilities": capabilities,
                        "default_app": user.content.get('defaultApp', "search") or "search",
                        "type": user.content.get('type', "user") or "user"
                    }
                    users.append(user_info)
                    print(f"✅ Successfully processed user: {user.name}")
                else:
                    # Handle users without content
                    user_info = {
                        "username": user.name,
                        "real_name": "N/A",
                        "email": "N/A",
                        "roles": [],
                        "capabilities": [],
                        "default_app": "search",
                        "type": "user"
                    }
                    users.append(user_info)
                    print(f"⚠️ User {user.name} has no content, using default values")
            except Exception as e:
                print(f"⚠️ Error processing user {user.name}: {str(e)}")
                continue
            
        print(f"✅ Found {len(users)} users")
        return users
        
    except Exception as e:
        print(f"❌ Error listing users: {str(e)}")
        raise

def get_indexes_and_sourcetypes() -> Dict[str, Any]:
    """
    Get a list of all indexes and their sourcetypes.
    
    This endpoint performs a search to gather:
    - All available indexes
    - All sourcetypes within each index
    - Event counts for each sourcetype
    - Time range information
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - indexes: List of all accessible indexes
            - sourcetypes: Dictionary mapping indexes to their sourcetypes
            - metadata: Additional information about the search
    """
    try:
        service = get_splunk_connection()
        print("📊 Fetching indexes and sourcetypes...")
        
        # Get list of indexes
        indexes = [index.name for index in service.indexes]
        print(f"Found {len(indexes)} indexes")
        
        # Search for sourcetypes across all indexes
        search_query = """
        | tstats count WHERE index=* BY index, sourcetype
        | stats count BY index, sourcetype
        | sort - count
        """
        
        kwargs_search = {
            "earliest_time": "-24h",
            "latest_time": "now",
            "preview": False,
            "exec_mode": "blocking"
        }
        
        print("🔍 Executing search for sourcetypes...")
        job = service.jobs.create(search_query, **kwargs_search)
        
        # Get the results
        result_stream = job.results(output_mode='json')
        results_data = json.loads(result_stream.read().decode('utf-8'))
        
        # Process results
        sourcetypes_by_index = {}
        for result in results_data.get('results', []):
            index = result.get('index', '')
            sourcetype = result.get('sourcetype', '')
            count = result.get('count', '0')
            
            if index not in sourcetypes_by_index:
                sourcetypes_by_index[index] = []
            
            sourcetypes_by_index[index].append({
                'sourcetype': sourcetype,
                'count': count
            })
        
        response = {
            'indexes': indexes,
            'sourcetypes': sourcetypes_by_index,
            'metadata': {
                'total_indexes': len(indexes),
                'total_sourcetypes': sum(len(st) for st in sourcetypes_by_index.values()),
                'search_time_range': '24 hours'
            }
        }
        
        print(f"✅ Successfully retrieved indexes and sourcetypes")
        return response
        
    except Exception as e:
        print(f"❌ Error getting indexes and sourcetypes: {str(e)}")
        raise

def health_check() -> Dict[str, Any]:
    """Get basic Splunk connection information and list available apps"""
    try:
        service = get_splunk_connection()
        print("🏥 Performing health check...")
        
        # List available apps
        apps = []
        for app in service.apps:
            try:
                app_info = {
                    "name": app['name'],
                    "label": app['label'],
                    "version": app['version']
                }
                apps.append(app_info)
            except Exception as e:
                print(f"⚠️ Error getting info for app {app['name']}: {str(e)}")
                continue
        
        response = {
            "status": "healthy",
            "connection": {
                "host": os.environ["splunk_access_host"],
                "port": os.environ["splunk_access_port"],
                "scheme": "https",
                "username": "admin",
                "ssl_verify": False
            },
            "apps_count": len(apps),
            "apps": apps
        }
        
        print(f"✅ Health check successful. Found {len(apps)} apps")
        return response
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        raise

        
search_splunk_tool = FunctionTool.from_defaults(fn=search_splunk)
list_indexes_tool = FunctionTool.from_defaults(fn=list_indexes)
get_index_info_tool = FunctionTool.from_defaults(fn=get_index_info)
list_saved_searches_tool = FunctionTool.from_defaults(fn=list_saved_searches)
list_users_tool = FunctionTool.from_defaults(fn=list_users)
get_indexes_and_sourcetypes_tool = FunctionTool.from_defaults(fn=get_indexes_and_sourcetypes)
health_check_tool = FunctionTool.from_defaults(fn=health_check)







    
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







    
# In[4]:


def apply(model,df,param):
    try:
        query = param['options']['params']['prompt'].strip('\"')
    except:
        result = pd.DataFrame({'Message': "ERROR: Please input a parameter \'prompt\'."})
        return result
        
    tool_list = [
        search_splunk_tool,
        list_indexes_tool,
        get_index_info_tool,
        list_saved_searches_tool,
        list_users_tool,
        get_indexes_and_sourcetypes_tool,
        health_check_tool
    ]

    try:
        service = param['options']['params']['llm_service'].strip("\"")
        print(f"Using {service} LLM service.")
    except:
        service = "ollama"
        print("Using default Ollama LLM service.")

    try:
        model_name = param['options']['params']['model_name'].strip("\"")
    except:
        model_name = None
        print("No model name specified")
        
    llm, m = create_llm(service=service, model=model_name)

    if llm is None:
        cols={'Message': [m]}
        returns=pd.DataFrame(data=cols)
        return returns

    
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





    
# In[11]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns

def compute(model,df,param):
    try:
        query = param['options']['params']['prompt'].strip('\"')
    except:
        result = pd.DataFrame({'Message': "ERROR: Please input a parameter \'prompt\'."})
        return result
    # Case of only two functions. Please customize for your own functions
    tool_list = [
        search_splunk_tool,
        list_indexes_tool,
        get_index_info_tool,
        list_saved_searches_tool,
        list_users_tool,
        get_indexes_and_sourcetypes_tool,
        health_check_tool
    ]

    try:
        service = param['options']['params']['llm_service'].strip("\"")
        print(f"Using {service} LLM service.")
    except:
        service = "ollama"
        print("Using default Ollama LLM service.")

    try:
        model_name = param['options']['params']['model_name'].strip("\"")
    except:
        model_name = None
        print("No model name specified")
        
    llm, m = create_llm(service=service, model=model_name)

    if llm is None:
        cols={'Message': [m]}
        returns=pd.DataFrame(data=cols)
        return returns

    
    worker = ReActAgentWorker.from_tools(tool_list, llm=llm)
    agent = AgentRunner(worker)     
    response = agent.chat(query)
    
    cols = {"Response": [response.response]}
    for i in range(len(response.sources)):
        if response.sources[i].tool_name != "unknown":
            cols[response.sources[i].tool_name] = [response.sources[i].content]
    result = pd.DataFrame(data=cols)
    return result

















