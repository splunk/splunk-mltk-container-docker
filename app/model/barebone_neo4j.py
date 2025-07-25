#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
import neo4j
from neo4j import GraphDatabase
# read configuration from additional DSDL services settings
configurations = json.loads(os.environ['llm_config'])
neo4j_config = configurations['graph_db']['neo4j'][0]
neo4j_uri = neo4j_config['url']
neo4j_auth = (neo4j_config['username'], neo4j_config['password'])

# or use other ways to retrieve desired configurations
#neo4j_uri = "neo4j://neo4j:7687"
#neo4j_auth = ("neo4j", "changeme")
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"















    
# In[20]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param













    
# In[68]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['neo4j'] = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)    
    return model

# define any helper functions for neo4j
def create_connection(tx, row):
    # Create a new connection for each row in the dataframe
    cypher_query = f"""
        MERGE (src:ip_address {{ip: '{row['src_ip']}'}})
        MERGE (dst:ip_address {{ip: '{row['dst_ip']}'}})
        CREATE (src)-[conn:connection {{
          count: {row['count']},
          sum_bytes_sent: {row['sum_bytes_sent']},
          sum_bytes_received: {row['sum_bytes_received']},
          sum_packets_received: {row['sum_packets_received']},
          sum_packets_sent: {row['sum_packets_sent']}
        }}]->(dst)
        """
    result = tx.run(cypher_query)







    
# In[62]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    with model['neo4j'].session(database="neo4j") as session:
        for _, row in df.iterrows():
            org_id = session.execute_write(create_connection, row)
            #print(f"Added: {row}")
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[66]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    records, summary, keys = model['neo4j'].execute_query(
        "MATCH (src:ip_address)-[conn:connection]->(dst:ip_address)RETURN count(conn)",
        database_="neo4j",
    )
    result = pd.DataFrame(records[0], columns=['result'])
    return result







    
# In[50]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    return model





    
# In[51]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init(None,None)
    return model





    
# In[52]:


# return a model summary
def summary(model=None):
    returns = {"version": {"neo4j": neo4j.__version__, "pandas": pd.__version__} }
    return returns

















