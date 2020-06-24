#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import sys
import json
import pandas as pd
import numpy as np
from random import random
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[13]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[7]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    appName = "recommendation"
    if 'options' in param:
        if 'model_name' in param['options']: 
            appName = param['options']['model_name']
    sparkConf = SparkConf().setAll([('spark.executor.memory', '1g'), ('spark.executor.cores', '1'), ('spark.cores.max', '4'), ('spark.driver.memory','4g'), ('spark.driver.maxResultSize','4g')])
    spark = SparkSession.builder.appName(appName).config(conf=sparkConf).getOrCreate()
    model['spark'] = spark
    return model











    
# In[118]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    spark = model['spark']
    sc = spark.sparkContext
    feature_variables = param['feature_variables']
    
    rank=10
    numIterations=10
    if 'options' in param:
        if 'params' in param['options']:
            if 'rank' in param['options']['params']:
                rank = int(param['options']['params']['rank'])
            if 'numIterations' in param['options']['params']:
                numIterations = int(param['options']['params']['numIterations'])

    sdf = spark.createDataFrame(df)
    ratings = sdf.rdd.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    model['als'] = ALS.train(ratings, rank, numIterations)

    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model['als'].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    model['MSE'] = MSE
    info = {"message": "model trained", "Mean Squared Error": str(MSE)}
    return info







    
# In[107]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    spark = model['spark']
    sdf = spark.createDataFrame(df)
    limit = None
    if 'options' in param:
        if 'params' in param['options']:
            if 'limit' in param['options']['params']:
                limit = int(param['options']['params']['limit'])
    ratings = sdf.rdd.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    testdata = ratings.map(lambda p: (p[0], p[1]))
    predictions = model['als'].predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    if limit==None:
        results = predictions.map(lambda r: (r[1])).collect()
    else:
        results = predictions.takeOrdered(limit, key = lambda x: -x[1])
    result = pd.DataFrame(results)
    return result







    
# In[125]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    sc = model['spark'].sparkContext
    import shutil
    from pathlib import Path
    if Path(MODEL_DIRECTORY + name).is_dir():
        shutil.rmtree(MODEL_DIRECTORY + name)
    model['als'].save(sc, MODEL_DIRECTORY + name)
    return model







    
# In[123]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init({},{})
    model['als'] = MatrixFactorizationModel.load(model['spark'].sparkContext, MODEL_DIRECTORY + name) 
    return model







    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"spark": "no model"}
    if model:
        returns = {"spark_info": str(model['spark'].sparkContext.getConf().getAll()) }
    return returns









