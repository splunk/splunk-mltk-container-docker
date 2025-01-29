#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import pandas as pd
import spacy
import en_core_web_sm
from spacytextblob.spacytextblob import SpacyTextBlob

# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"









    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[7]:


# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    # Load English parser and text blob (for sentiment analysis)
    model = spacy.load('en_core_web_sm')
    #spacy_text_blob = SpacyTextBlob()
    #model.add_pipe(spacy_text_blob)
    model.add_pipe('spacytextblob')
    return model







    
# In[9]:


# returns a fit info json object
def fit(model,df,param):
    returns = {}
    return returns





    
# In[14]:


def apply(model,df,param):
    X = df[param['feature_variables']].values.tolist()
    temp_data=list()
    
    for i in range(len(X)):
        doc = model(str(X[i]))
        polarity=doc._.blob.polarity
        subjectivity=doc._.blob.subjectivity
        assessments=doc._.blob.sentiment_assessments.assessments
        temp_data.append([polarity,subjectivity,assessments])
        
    column_names=["polarity","subjectivity","assessments"]
    returns=pd.DataFrame(temp_data, columns=column_names)
        
    return returns







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # model will not be saved or reloaded as it is pre-built
    return model







    
# In[17]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    # model will not be saved or reloaded as it is pre-built
    return model





    
# In[18]:


# return model summary
def summary(model=None):
    returns = {"version": {"spacy": spacy.__version__} }
    if model is not None:
        s = []
        returns["summary"] = ''.join(s)
    return returns













