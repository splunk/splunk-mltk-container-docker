#!/usr/bin/env python
# coding: utf-8


    
# In[1]:


# mltkc_import
# this definition exposes all python module imports that should be available in all subsequent commands
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow import keras
import keras
from tensorflow import feature_column
#from tensorflow.keras import layers
from keras import layers
from sklearn.model_selection import train_test_split
import shap

# restrict GPU memory https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# global constants
MODEL_DIRECTORY = "/srv/app/model/data/"

def df_to_dataset(dataframe, target_label_name, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target_label_name)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def df_to_dataset_apply(dataframe, batch_size=32):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    ds = ds.batch(batch_size)
    return ds







    
# In[3]:


# mltkc_stage
# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param







    
# In[5]:


# mltkc_init
# initialize the model
# params: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    model['param'] = param
    #y = df[param['target_variables'][0]]
    #X = df[param['feature_variables']] #.astype(float)
    #print("FIT build model with input shape " + str(X.shape))
    #input_shape = int(X.shape[1])
    
    model_structure = '256-128'
    numeric_features = None
    embedding_features = None
    embedding_dimensions = 8
    categorical_features = None

    feature_columns = []
    
    if 'options' in param:
        if 'params' in param['options']:
            if 'structure' in param['options']['params']:
                model_structure = str(param['options']['params']['structure']).lstrip("\"").rstrip("\"").lstrip(" ").rstrip(" ")
            if 'numeric_features' in param['options']['params']:
                numeric_features = str(param['options']['params']['numeric_features']).lstrip("\"").rstrip("\"").lstrip(" ").rstrip(" ").replace(" ", ",").split(",")
                for feature in numeric_features:
                    if '*' in feature:
                        wildcards = df.filter(like=feature.replace('*','')).columns
                        for wildcard in wildcards:
                            feature_columns.append(feature_column.numeric_column(wildcard))
                    elif feature in df:
                        feature_columns.append(feature_column.numeric_column(feature))
            if 'embedding_features' in param['options']['params']:
                embedding_features = str(param['options']['params']['embedding_features']).lstrip("\"").rstrip("\"").lstrip(" ").rstrip(" ").replace(" ", ",").split(",")
                for feature in embedding_features:
                    if '*' in feature:
                        wildcards = df.filter(like=feature.replace('*','')).columns
                        for wildcard in wildcards:
                            feature_embedding = feature_column.categorical_column_with_vocabulary_list(wildcard, df[wildcard].unique())
                            feature_embedding = feature_column.embedding_column(feature_embedding, dimension=embedding_dimensions)
                            feature_columns.append(feature_embedding)
                    elif feature in df:
                        feature_embedding = feature_column.categorical_column_with_vocabulary_list(feature, df[feature].unique())
                        feature_embedding = feature_column.embedding_column(feature_embedding, dimension=embedding_dimensions)
                        feature_columns.append(feature_embedding)
            if 'categorical_features' in param['options']['params']:
                categorical_features = str(param['options']['params']['categorical_features']).lstrip("\"").rstrip("\"").lstrip(" ").rstrip(" ").replace(" ", ",").split(",")
                for feature in categorical_features:
                    if '*' in feature:
                        wildcards = df.filter(like=feature.replace('*','')).columns
                        for wildcard in wildcards:
                            categorical_column = feature_column.categorical_column_with_vocabulary_list(wildcard, df[wildcard].unique())
                            categorical_column = feature_column.indicator_column(categorical_column)
                            feature_columns.append(categorical_column)
                    elif feature in df:
                        categorical_column = feature_column.categorical_column_with_vocabulary_list(feature, df[feature].unique())
                        categorical_column = feature_column.indicator_column(categorical_column)
                        feature_columns.append(categorical_column)
                    
    model['feature_columns'] = feature_columns
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model['feature_layer'] = feature_layer
    
    hidden_factors = np.floor(np.array(model_structure.split("-"), dtype="float"))
    keras_model = tf.keras.Sequential()
    keras_model.add(feature_layer)
    for hidden in hidden_factors:
        keras_model.add(layers.Dense(int(hidden), activation=tf.nn.relu))
        keras_model.add(layers.Dropout(0.01))
    keras_model.add(layers.Dense(1, activation=tf.nn.sigmoid))
        
    keras_model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model['keras_model'] = keras_model
        
    return model









    
# In[8]:


# mltkc_stage_create_model_fit
# returns a fit info json object
def fit(model,df,param):
    returns = {}
    #X = df[param['feature_variables']]
    #Y = df[param['target_variables']]
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
        
    model_epochs = 10
    model_batch_size = 1
    if 'options' in param:
        if 'params' in param['options']:
            if 'epochs' in param['options']['params']:
                model_epochs = int(param['options']['params']['epochs'])
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
    
    train_ds = df_to_dataset(df, param['target_variables'][0], batch_size=model_batch_size)
    val_ds = df_to_dataset(val, param['target_variables'][0], shuffle=False, batch_size=model_batch_size)
    test_ds = df_to_dataset(test, param['target_variables'][0], shuffle=False, batch_size=model_batch_size)

    # connect model training to tensorboard
    log_dir="/srv/notebooks/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # run the training
    returns['fit_history'] = model['keras_model'].fit(train_ds,
        validation_data=val_ds,
        epochs=model_epochs,
        verbose=2,
        callbacks=[tensorboard_callback])    

    returns['model_epochs'] = model_epochs
    returns['model_batch_size'] = model_batch_size
    model['model_epochs'] = model_epochs
    model['model_batch_size'] = model_batch_size
    
    returns['model_loss_acc'] = model['keras_model'].evaluate(test_ds)
    return returns







    
# In[10]:


# mltkc_stage_create_model_apply
def apply(model,df,param):
    X = df[param['feature_variables']]
    model_batch_size = 1
    print("APPLY PARAMS: " + str(param))
    if 'options' in param:
        if 'params' in param['options']:
            if 'batch_size' in param['options']['params']:
                model_batch_size = int(param['options']['params']['batch_size'])
    # TODO
    apply_dataset = df_to_dataset_apply(X, batch_size=model_batch_size)
    y_hat = model['keras_model'].predict(apply_dataset, verbose=1)
    return y_hat







    
# In[12]:


# save model to name in expected convention "<algo_name>_<model_name>.h5"
def save(model,name):
    # save keras model to hdf5 file
    # https://www.tensorflow.org/beta/tutorials/keras/save_and_restore_models
    if 'keras_model' in model:
        tf.keras.models.save_model(model['keras_model'], MODEL_DIRECTORY + name)
    return model







    
# In[21]:


# load model from name in expected convention "<algo_name>_<model_name>.h5"
def load(name):
    model = {}
    #with open(MODEL_DIRECTORY + name + ".feature_layer_config.json", 'r') as file:
    #    feature_layer_config = json.load(file)

    # #model = tf.keras.models.load_model(MODEL_DIRECTORY + name + '.h5') #, custom_objects=feature_layer_config)
    model['keras_model'] = tf.keras.models.load_model(MODEL_DIRECTORY + name)
    
    return model









    
# In[25]:


# return model summary
def summary(model=None):
    returns = {"version": {"tensorflow": tf.__version__, "keras": keras.__version__} }
    return returns







