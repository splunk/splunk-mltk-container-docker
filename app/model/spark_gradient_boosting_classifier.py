#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# this definition exposes all python module imports that should be available in all subsequent commands
import sys
import json
import pandas as pd
import numpy as np
from random import random
from pyspark.sql import SparkSession
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[ ]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[ ]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    appName = "gradient_boosting_classifer_spark_model"
    if 'options' in param:
        if 'model_name' in param['options']: 
            appName = param['options']['model_name']
    spark = SparkSession        .builder        .appName(appName)        .getOrCreate()
    model['spark'] = spark
    return model











    
# In[ ]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    spark = model['spark']
    sc = spark.sparkContext
    feature_variables = param['feature_variables']
    target_variable = param['target_variables'][0]
    iterations = 10
    if 'options' in param:
        if 'params' in param['options']:
            if 'iterations' in param['options']['params']:
                iterations = int(param['options']['params']['iterations'])
    sdf = spark.createDataFrame(df)
    rdd = sdf.rdd.map(lambda row: LabeledPoint(row[target_variable], [row[x] for x in feature_variables]) ) 

    model['model'] = GradientBoostedTrees.trainClassifier(rdd, categoricalFeaturesInfo={}, numIterations=iterations)

    info = {"message": "model trained"}
    return info







    
# In[ ]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    spark = model['spark']
    sdf = spark.createDataFrame(df)
    feature_variables = param['feature_variables']
    predictions = model['model'].predict(sdf.rdd.map(lambda row: [row[x] for x in feature_variables]))
    result = pd.DataFrame(predictions.collect())
    return result







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    import shutil
    from pathlib import Path
    if Path(MODEL_DIRECTORY + name).is_dir():
        shutil.rmtree(MODEL_DIRECTORY + name)
    model['model'].save(model['spark'].sparkContext, MODEL_DIRECTORY + name)
    return model







    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = init({},{})
    model['model'] = GradientBoostedTreesModel.load(model['spark'].sparkContext, MODEL_DIRECTORY + name)
    return model







    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"spark": "no model"}
    if model:
        returns = {"spark_info": str(model['spark'].sparkContext.getConf().getAll()) }
    return returns









