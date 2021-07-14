#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import rrcf as rcf
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







    
# In[5]:


# Create the random cut forest from the source data
def init(df,param):
    # Set model parameters
    features=len(df)
    num_trees=15
    tree_size=30
    sample_size_range=(features // tree_size, tree_size)
    
    if 'options' in param:
        if 'params' in param['options']:
            if 'num_trees' in param['options']['params']:
                num_trees = int(param['options']['params']['num_trees'])
            if 'tree_size' in param['options']['params']:
                tree_size = int(param['options']['params']['tree_size'])
    
    # Convert data to nparray
    variables=[]
    
    if 'target_variables' in param:
        variables=param['target_variables']
        
    other_variables=[]
    
    if 'feature_variables' in param:
        other_variables=param['feature_variables']

    for item in other_variables:
        variables.append(item)
    
    data=df[variables].to_numpy().astype(float)
    
    # Create the random cut forest
    forest = []
    while len(forest) < num_trees:
        # Select random subsets of points uniformly
        ixs = np.random.choice(features, size=sample_size_range,
                               replace=False)
        # Add sampled trees to forest
        trees = [rcf.RCTree(data[ix], index_labels=ix)
                 for ix in ixs]
        forest.extend(trees)
    return forest







    
# In[7]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    
    return len(model)







    
# In[9]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    # Calculate the collusive displacement of the points in the random trees
    features=len(df)
    threshold=0.01
    
    if 'options' in param:
        if 'params' in param['options']:
            if 'threshold' in param['options']['params']:
                threshold = float(param['options']['params']['threshold'])
    
    avg_codisp = pd.Series(0.0, index=np.arange(features))
    index = np.zeros(features)

    for tree in model:
        codisp = pd.Series({leaf : tree.codisp(leaf)
                           for leaf in tree.leaves})

        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
    avg_codisp /= index
    
    # Identify outliers based on the collusive displacement values
    threshold_percentage=int(threshold*features)
    threshold = avg_codisp.nlargest(n=threshold_percentage).min()
    
    outlier=(avg_codisp >= threshold).astype(float)
    
    result=pd.DataFrame({'outlier':outlier,'collusive_displacement':avg_codisp})
    return result







    
# In[11]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    return model





    
# In[12]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    return model





    
# In[13]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns





