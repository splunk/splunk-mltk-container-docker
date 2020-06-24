#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import sys
import json
import pandas as pd
import numpy as np
from random import random
from operator import add
from pyspark.sql import SparkSession
# ...
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









    
# In[6]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    appName = "barebone_spark_model"
    if 'options' in param:
        if 'model_name' in param['options']: 
            appName = param['options']['model_name']
    spark = SparkSession        .builder        .appName(appName)        .getOrCreate()
    model['spark'] = spark
    return model











    
# In[10]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    # model.fit()
    info = {"message": "model trained"}
    return info







    
# In[12]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    spark = model['spark']
    partitions = 4
    sdf = spark.createDataFrame(df)
    rdd = sdf.rdd.map(lambda row: 1 if row["feature_0"] ** 2 + row["feature_1"] ** 2 <= 1 else 0)
    result = pd.DataFrame(rdd.collect())
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
    model = {}
#    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
#        model = json.load(file)
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"spark": "no model"}
    if model:
        returns = {"spark_info": str(model['spark'].sparkContext.getConf().getAll()) }
    return returns





