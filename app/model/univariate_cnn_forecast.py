#!/usr/bin/env python
# coding: utf-8


    
# In[ ]:


# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

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


# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    # Collect variables
    model_batch_size = 3
    n_features = 1
    hidden_layers = 50
    activation_func = 'sigmoid'
    if 'options' in param:
        if 'params' in param['options']:
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
            if 'hidden_layers' in param['options']['params']:
                hidden_layers = int(param['options']['params']['hidden_layers'])
            if 'activation' in param['options']['params']:
                activation_func = param['options']['params']['activation']
    
    # define model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=64, kernel_size=2, activation=activation_func, input_shape=(model_batch_size, n_features)))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hidden_layers,activation=activation_func))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model







    
# In[ ]:


# returns a fit info json object
# split a univariate sequence into samples
def split_sequence(sequence, batch_size):
    
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + batch_size
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def fit(model,df,param):
    returns = {}
    
    # Collect variables from param file
    model_epochs = 100
    model_batch_size = 3
    holdback = 30
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model_epochs = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
            if 'holdback' in param['options']['params']:
                holdback = int(param['options']['params']['holdback'])
    
    
    # flatten data frame into an array and extract the training set
    full_data = df[param['options']['split_by']].values.tolist()
    train_set = list(full_data[:len(full_data)-holdback])
    
    # split data into samples
    X, y = split_sequence(train_set, model_batch_size)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], 1))
    

    # connect model training to tensorboard
    log_dir="/srv/notebooks/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # run the training
    returns['fit_history'] = model.fit(x=X,
                                       y=y, 
                                       verbose=2, 
                                       epochs=model_epochs,
                                       shuffle=False,
                                       callbacks=[tensorboard_callback])
    # memorize parameters
    returns['model_epochs'] = model_epochs
    returns['model_batch_size'] = model_batch_size
    returns['model_loss_acc'] = model.evaluate(x = X, y = y)
    return returns







    
# In[ ]:


def apply(model,df,param):
    
    # Collect variables
    model_batch_size = 3
    future_steps = 30
    holdback = 30
    if 'options' in param:
        if 'params' in param['options']:
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
            if 'future_steps' in param['options']['params']:
                future_steps = int(param['options']['params']['future_steps'])
            if 'holdback' in param['options']['params']:
                holdback = int(param['options']['params']['holdback'])
    
    # select training data
    X = df[param['options']['split_by']].values.tolist()
    
    test_set = list(X[len(X)-holdback-model_batch_size:])
    predictions = list(X[:len(X)-holdback])
    
    # generate forecast
    for i in range(0, holdback+future_steps):
        if i<holdback:
            X_batch = np.asarray(test_set[i:i+model_batch_size]).reshape(1,model_batch_size,1)
            y_pred = model.predict(x = X_batch, verbose=1)
            predictions.append(list(y_pred[0]))
        else:
            X_batch = np.asarray(test_set[i:i+model_batch_size]).reshape(1,model_batch_size,1)
            y_pred = model.predict(x = X_batch, verbose=1)
            predictions.append(list(y_pred[0]))
            test_set.append(y_pred)
            
    # append predictions to time series to return a data frame
    return predictions







    
# In[ ]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # save keras model to hdf5 file
    # https://www.tensorflow.org/beta/tutorials/keras/save_and_restore_models
    model.save(MODEL_DIRECTORY + name + ".h5")
    return model





    
# In[ ]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    model = keras.models.load_model(MODEL_DIRECTORY + name + ".h5")
    return model





    
# In[ ]:


# return model summary
def summary(model=None):
    returns = {"version": {"tensorflow": tf.__version__, "keras": keras.__version__} }
    if model is not None:
        # Save keras model summary to string:
        s = []
        model.summary(print_fn=lambda x: s.append(x+'\n'))
        returns["summary"] = ''.join(s)
    return returns













