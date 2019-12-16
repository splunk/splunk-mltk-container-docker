#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import torch
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


def init(df,param):
    X = df[param['feature_variables']]
    Y = df[param['target_variables']]
    input_size = int(X.shape[1])
    num_classes = len(np.unique(Y.to_numpy()))
    learning_rate = 0.001
    mapping = { key: value for value,key in enumerate(np.unique(Y.to_numpy().reshape(-1))) }
    print("FIT build logistic regression model with input shape " + str(X.shape))
    print("FIT build model with target classes " + str(num_classes))
    model = {
        "input_size": input_size,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "mapping": mapping,
        "num_epochs": 10000,
        "batch_size": 100,
    }
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model['num_epochs'] = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model['batch_size'] = int(param['options']['params']['batch_size'])
    # Simple logistic regression model
    model['model'] = torch.nn.Linear(input_size, num_classes)
    # Define loss and optimizer
    model['criterion'] = torch.nn.CrossEntropyLoss()  
    model['optimizer'] = torch.optim.SGD(model['model'].parameters(), lr=learning_rate)      
    return model







    
# In[ ]:


def fit(model,df,param):
    returns = {}
    X = df[param['feature_variables']].astype('float32').to_numpy()
    Y = df[param['target_variables']].to_numpy().reshape(-1)
    mapping = { key: value for value,key in enumerate(np.unique(Y)) }
    Y = df[param['target_variables']].replace( {param['target_variables'][0]:mapping } ).to_numpy().reshape(-1)
    #Y = df[param['target_variables']].to_numpy().reshape(-1)
    #Y = pd.get_dummies(Y).astype('float32').to_numpy()
    #Ymap = df[param['target_variables']].replace( {param['target_variables'][0]:mapping } ).to_numpy()
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model['num_epochs'] = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model['batch_size'] = int(param['options']['params']['batch_size'])
    print(model['num_epochs'])
    for epoch in range(model['num_epochs']):
        inputs = torch.from_numpy(X)
        targets = torch.from_numpy(Y)
        outputs = model['model'](inputs)
        loss = model['criterion'](outputs, targets)
        model['optimizer'].zero_grad()
        loss.backward()
        model['optimizer'].step()
        if (epoch+1) % (model['num_epochs']/10) == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, model['num_epochs'], loss.item()))                
    # memorize parameters
    returns['model_epochs'] = model['num_epochs']
    returns['model_batch_size'] = model['batch_size']
    returns['model_loss_acc'] = loss.item()
    return returns







    
# In[ ]:


def apply(model,df,param):
    X = df[param['feature_variables']].astype('float32').to_numpy()
    classes = {v: k for k, v in model['mapping'].items()}
    with torch.no_grad():
        input = torch.from_numpy(X)
        output = model['model'](input)
        y_hat = output.data
        _, predicted = torch.max(output.data, 1)
        prediction = [classes[key] for key in predicted.numpy()]
    return prediction







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    torch.save(model, MODEL_DIRECTORY + name + ".pt")
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    model = torch.load(MODEL_DIRECTORY + name + ".pt")
    return model





    
# In[ ]:


# return model summary
def summary(model=None):
    returns = {"version": {"pytorch": torch.__version__} }
    if model is not None:
        if 'model' in model:
            returns["summary"] = str(model)
    return returns





