#!/usr/bin/env python
# coding: utf-8


    
# In[2]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
import tensorflow as tf
# restrict GPU memory https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ...
# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[8]:


# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[10]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    # Determine the number of features in the data
    n_features = df[param['feature_variables']].shape[1]
    
    # Set the model parameters depending on the input variables
    previous_steps = 3
    hidden_layers = 30
    activation_func = 'tanh'
    
    if 'options' in param:
        if 'params' in param['options']:
            if 'previous_steps' in param['options']['params']:
                previous_steps = int(param['options']['params']['previous_steps'])
            if 'hidden_layers' in param['options']['params']:
                hidden_layers = int(param['options']['params']['hidden_layers'])
            if 'activation' in param['options']['params']:
                activation_func = param['options']['params']['activation']
    
    model = Sequential()
    model.add(LSTM(units=hidden_layers, activation=activation_func, return_sequences=True, input_shape=(previous_steps, n_features)))
    model.add(LSTM(units=hidden_layers))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model







    
# In[12]:


# train your model
# returns a fit info json object and may modify the model object
def fit(model,df,param):
    X = df[param['feature_variables']]
    y = df[param['target_variables']]
    
    # Determine how many features are in the dataset
    n_features = X.shape[1]
    
    # Determine the batch size and epochs
    previous_steps=3
    model_batch_size=10
    model_epochs=100

    if 'options' in param:
        if 'params' in param['options']:
            if 'previous_steps' in param['options']['params']:
                previous_steps = int(param['options']['params']['previous_steps'])
            if 'epochs' in param['options']['params']:
                model_epochs = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])

    # Scale the input data
    scaler = MinMaxScaler()
    X_ss = scaler.fit_transform(X)

    # Loop through the data to ensure you have the correct input and output for the LSTM
    input_data=[]
    output_data=[]
    for i in range(X_ss.shape[0]-previous_steps-1):
        t=[]
        for j in range(0,previous_steps):
            t.append(X_ss[i+j])

        input_data.append(t)
        output_data.append(y.iloc[i+previous_steps])

    X = np.array(input_data)
    y = np.array(output_data)

    X = X.reshape(X.shape[0],previous_steps, n_features)

    print("Training data contains ", X.shape[0], " records of shape ", X.shape)
    
    # fit network
    history = model.fit(X, y, epochs=model_epochs, batch_size=model_batch_size, validation_data=(X, y), verbose=2, shuffle=False)
    info = {"message": "model trained"}
    return info







    
# In[14]:


# apply your model
# returns the calculated results
def apply(model,df,param):
    y_hat = np.zeros(df.shape[0]) 
    
    X = df[param['feature_variables']]
    n_features = X.shape[1]
    
    # Determine the batch size and epochs
    previous_steps=3

    if 'options' in param:
        if 'params' in param['options']:
            if 'previous_steps' in param['options']['params']:
                previous_steps = int(param['options']['params']['previous_steps'])
    
    # Scale the input data
    scaler = MinMaxScaler()
    X_ss = scaler.fit_transform(X)

    # Loop through the data to ensure you have the correct input and output for the LSTM
    input_data=[]
    output_data=[]
    for i in range(X_ss.shape[0]-previous_steps-1):
        t=[]
        for j in range(0,previous_steps):
            t.append(X_ss[i+j])

        input_data.append(t)

    X = np.array(input_data)
    X = X.reshape(X.shape[0],previous_steps, n_features)
    
    predictions = model.predict(X)
    
    for k in range(y_hat.shape[0]):
        if k > previous_steps:
            y_hat[k]=predictions[k-previous_steps-1]
    
    result = pd.DataFrame(y_hat, columns=['prediction'])
    return result







    
# In[16]:


# save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model.save(MODEL_DIRECTORY + name + ".keras")
    return model







    
# In[19]:


# load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = load_model(MODEL_DIRECTORY + name + ".keras")
    return model







    
# In[22]:


# return a model summary
def summary(model=None):
    returns = {"version": {"numpy": np.__version__, "pandas": pd.__version__} }
    return returns







