#!/usr/bin/env python
# coding: utf-8


    
# In[18]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from causalnex.structure import DAGRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[22]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param

















    
# In[24]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = DAGRegressor(
                alpha=0.1,
                beta=0.9,
                fit_intercept=True,
                hidden_layer_units=None,
                dependent_target=True,
                enforce_dag=True,
                 )
    return model







    
# In[26]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    
    target=param['target_variables'][0]
    
    #Data prep for processing
    y_p = df[target]
    y = y_p.values

    X_p = df[param['feature_variables']]
    X = X_p.to_numpy()
    X_col = list(X_p.columns)

    #Scale the data
    ss = StandardScaler()
    X_ss = ss.fit_transform(X)
    y_ss = (y - y.mean()) / y.std()
    
    scores = cross_val_score(model, X_ss, y_ss, cv=KFold(shuffle=True, random_state=42))
    print(f'MEAN R2: {np.mean(scores).mean():.3f}')

    X_pd = pd.DataFrame(X_ss, columns=X_col)
    y_pd = pd.Series(y_ss, name=target)

    model.fit(X_pd, y_pd)
    
    info = pd.Series(model.coef_, index=X_col)
    #info = pd.Series(model.coef_, index=list(df.drop(['_time'],axis=1).columns))
    return info







    
# In[28]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    data = []

    for col in list(df.columns):
        s = model.get_edges_to_node(col)
        for i in s.index:
            data.append([i,col,s[i]]);

    graph = pd.DataFrame(data, columns=['src','dest','weight'])

    #results to send back to Splunk
    graph_output=graph[graph['weight']>0]
    return graph_output







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    #with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
    #    json.dump(model, file)
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    #with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
    #    model = json.load(file)
    return model





    
# In[ ]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns















