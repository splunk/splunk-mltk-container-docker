#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import time
import copy
import pandas as pd
import tarfile
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore
# tensorboard related
from torch.utils.tensorboard import SummaryWriter
import tensorboard
import datetime
import logging
import sys
import io
import os
import psutil
import shutil
from sumeval.metrics.rouge import RougeCalculator





    
# In[3]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("/srv/notebooks/data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("/srv/notebooks/data/"+name+".json", 'r') as f:
        param = json.load(f) 
#         param = {}
    return df, param





    
# In[4]:


def init(df,param):
    model = {}
    return model





    
# In[5]:


# No model loaded. Stage holder
def fit(model,df,param):  
    returns = {}
    return returns





    
# In[8]:


def apply(model,df,param):
    tag = "-- process=summarization_evaluation "
    X = df["summary"].values.tolist()
    Y = df["extracted_summary"].values.tolist()
    temp_rouge=list()
    pattern = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]')
    if pattern.findall(X[0]):
        language = "jp"
        rouge = RougeCalculator(stopwords=True, lang="ja")
    else:
        language = "en"
        rouge = ROUGEScore()
    # rouge = ROUGEScore()
    print(tag + "scoring begins")
    for i in range(len(X)):
        print(tag + "scoring {}-th utterance over {}. {}% finished.".format(i+1, len(X), round(i/len(X)*100)))
        # r = rouge(X[i], Y[i])
        if language == "jp":
            rouge_1 = rouge.rouge_n(
                summary=Y[i],
                references=X[i],
                n=1)
            # temp_rouge.append(round(r[param['options']['params']['metrics']].item(),2))
            temp_rouge.append(rouge_1)
        else:
            r = rouge(X[i], Y[i])
            temp_rouge.append(round(r[param['options']['params']['metrics']].item(),2))
    cols={param['options']['params']['metrics']: temp_rouge}
    returns=pd.DataFrame(data=cols)
    print(tag + "scoring successfully finished")
        
    return returns







    
# In[14]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model, name):
    return {}





    
# In[15]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(path):
    model = {}
    return model





    
# In[16]:


# return model summary
def summary(model=None):
    returns = {}
    return returns









