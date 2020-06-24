#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import sys
import json
import pandas as pd
import numpy as np
from random import random
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[4]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[8]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    appName = "fp_growth_spark_model"
    if 'options' in param:
        if 'model_name' in param['options']: 
            appName = param['options']['model_name']
    sparkConf = SparkConf().setAll([('spark.executor.memory', '1g'), ('spark.executor.cores', '1'), ('spark.cores.max', '4'), ('spark.driver.memory','4g'), ('spark.driver.maxResultSize','4g')])
    spark = SparkSession.builder.appName(appName).config(conf=sparkConf).getOrCreate()
    model['spark'] = spark
    return model











    
# In[12]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    spark = model['spark']
    sc = spark.sparkContext
    feature_variables = param['feature_variables']
    target_variable = param['target_variables'][0]

    min_support = 0.10
    min_confidence = 0.10
    if 'options' in param:
        if 'params' in param['options']:
            if 'min_support' in param['options']['params']:
                min_support = float(param['options']['params']['min_support'])
            if 'min_confidence' in param['options']['params']:
                min_confidence = float(param['options']['params']['min_confidence'])

    df['_items'] = df[target_variable].map(lambda l: l.split(' '))
    sdf = spark.createDataFrame(df)

    model['fpgrowth'] = FPGrowth(itemsCol='_items', minSupport=min_support, minConfidence=min_confidence)
    model['model'] = model['fpgrowth'].fit(sdf)

    info = {"message": "model trained"}
    return info











    
# In[14]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    limit = 10
    min_items = 2
    max_items = 2
    if 'options' in param:
        if 'params' in param['options']:
            if 'limit' in param['options']['params']:
                limit = int(param['options']['params']['limit'])
            if 'min_items' in param['options']['params']:
                min_items = int(param['options']['params']['min_items'])
            if 'max_items' in param['options']['params']:
                max_items = int(param['options']['params']['max_items'])

    spark = model['spark']
    freqItems = model['model'].freqItemsets
    freqItems.createOrReplaceTempView("frequentItems")
    results = spark.sql("select items, freq from frequentItems where size(items) between "+str(min_items)+" and "+str(max_items)+" order by freq desc limit "+str(limit))
    result = results.toPandas()
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    #import shutil
    #from pathlib import Path
    #if Path(MODEL_DIRECTORY + name).is_dir():
    #    shutil.rmtree(MODEL_DIRECTORY + name)
    #model['model'].save(model['spark'].sparkContext, MODEL_DIRECTORY + name)
    return model







    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init({},{})
    #model['model'] = GradientBoostedTreesModel.load(model['spark'].sparkContext, MODEL_DIRECTORY + name)
    return model







    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"spark": "no model"}
    if model:
        returns = {"spark_info": str(model['spark'].sparkContext.getConf().getAll()) }
    return returns









