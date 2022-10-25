#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import string
import urllib.request
import hashlib
import ssl
import os.path
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import umap
from dsdlsupport.SplunkGenerateGraphicsObjects import SplunkGenerateGraphicsObjects

# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"
CACHE_DIRECTORY = "/srv/app/model/data/ja3_cache"

# SUPPORTING FUNCTION TO PARSE JA3 INTO A DATAFRAME
def ja3_list_to_df(ja3_list):
    ja3_sig_cols=pd.DataFrame(ja3_list.str.split(",").tolist())
    ja3_sig_cols.columns = ["SSLVersion","Cipher","SSLExtension","EllipticCurve","EllipticCurvePointFormat"]
    ja3_sig_cols=ja3_sig_cols.fillna('None')
    return ja3_sig_cols

# SUPPORTING FUNCTIONS TO CREATE CIPHER FREQUENCY LISTS
def get_value_counts(df,col):
    return df[col].str.split("-",expand=True).stack().value_counts()

def create_common_lists(ja3_df,dc=100):
    counts = dict()
    common_lists = dict()
    for column in ja3_df.columns:
        print("finding common "+column+"s")
        counts[column] = get_value_counts(ja3_df,column)
        common_lists[column] = counts[column].head(dc).index.values.tolist()
        print(common_lists[column])
    return common_lists

# SUPPORTING FUNCTIONS FOR ONEHOT ENCODING
def onehot_from_multivalue(df,col):
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop(col)),columns=col+"_"+mlb.classes_,index=df.index))
    return df

def onehot_encode_ja3(df,common_lists):
    df2=df
    for col in df.columns:
        df2[col]=df[col].str.split("-")
        df2[col] = df2[col].transform(lambda x: list(set(x) & set(common_lists[col])))
        df2 = onehot_from_multivalue(df2,col)
    return df2





















    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
# this cell "stages" the data pushed from splunk by the above SPL command, which needs to be run in a connected splunk environment whilst the dev container is running
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param









    
# In[6]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    
    model['onehot_max_cols'] = 100
    model['output_mode'] = "umap" # "umap" or "onehot"
    
    # https://umap-learn.readthedocs.io/en/latest/parameters.html
    model['umap_min_dist'] = 1.0
    model['umap_neighbours'] = 50
    model['umap_components'] = 2
    model['umap_metric']="hamming"
    
    if 'options' in param:
        if 'params' in param['options']:
            if 'onehot_max_cols' in param['options']['params']:
                model['onehot_max_cols'] = int(param['options']['params']['onehot_max_cols'])
            if 'umap_min_dist' in param['options']['params']:
                model['umap_min_dist'] = float(param['options']['params']['umap_min_dist'])
            if 'umap_neighbours' in param['options']['params']:
                model['umap_neighbours'] = int(param['options']['params']['umap_neighbours'])
            if 'output_mode' in param['options']['params']:
                model['output_mode'] = param['options']['params']['output_mode']
    
    model['reducer']=umap.UMAP(n_neighbors=model['umap_neighbours'],\
                               min_dist=model['umap_min_dist'],\
                               n_components=model['umap_components'],\
                               metric=model['umap_metric'])
    
    return model







    
# In[8]:


# train the model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    ja3_df = ja3_list_to_df(df[param['feature_variables'][0]])

    # calculate commonly used options for each signature column
    model['common_option_lists'] = create_common_lists(ja3_df, model['onehot_max_cols'])
    # turn the ja3 col into a dataframe with the correct columns
    ja3_df = ja3_list_to_df(df[param['feature_variables'][0]])
    # onehot encode on the ja3 dataframe
    ja3_df_onehot = onehot_encode_ja3(ja3_df,model['common_option_lists'])
    model['reducer'].fit(ja3_df_onehot)
    
    model['fit_data']=df
    model['fit_onehot_features']=ja3_df_onehot
    
    info = {"message": "model trained"}    
    return info







    
# In[10]:


# returns the calculated results
def apply(model,df,param):
    if df.equals(model["fit_data"]):
        ja3_df_onehot = model['fit_onehot_features']
    else:
        ja3_df = ja3_list_to_df(df[param['feature_variables'][0]])
        ja3_df_onehot = onehot_encode_ja3(ja3_df,model['common_option_lists'])
    
    result={}
    
    if model["output_mode"]=="umap":
        umap_reduced_features=pd.DataFrame( model["reducer"].transform(ja3_df_onehot) )
        result = umap_reduced_features
        fig = plt.figure(figsize=(5,5))
        plt.scatter(result[0],result[1],s=2,color=[0,0.3,0.9,0.1])
        SplunkGenerateGraphicsObjects(model,"ja3_umap_apply",fig)
    else:
        if model["output_mode"]=="onehot":
            result = ja3_df_onehot
        else:
            result = {"message": "unknown output mode, use \"umap\" or \"onehot\""}
        
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
#    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
#        json.dump(model, file)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
#    model = {}
#    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
#        model = json.load(file)
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__}}
    return returns







