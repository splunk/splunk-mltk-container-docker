#!/usr/bin/env python
# coding: utf-8


    
# In[306]:


# This definition exposes all python module imports that should be available in all subsequent commands
import json
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# Custom Lion Optimizer
class Lion(optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute update
                update = exp_avg.clone().mul_(beta2).add_(grad, alpha=1 - beta2).sign_()

                # Weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                # Update parameters
                p.data.add_(update, alpha=-group['lr'])

        return loss

# Define the DNN model
class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, nodes_per_layer, activation_func, dropout_rate):
        super(SimpleDNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, nodes_per_layer))
        layers.append(activation_func)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(activation_func)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(nodes_per_layer, 2))  # Binary classification
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Global constants
nodes_per_layer = 128
batch_size = 256
class_weight = .3
num_epochs = 10
MODEL_DIRECTORY = "/srv/app/model/data/"







    
# In[308]:


# This cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("data/"+name+".json", 'r') as f:
        param = json.load(f)
    return df, param











    
# In[312]:


# initialize your model
# available inputs: data and parameters
# returns the model object which will be used as a reference to call fit, apply and summary subsequently
def init(df,param):
    model = {}
    input_dim = len(df.columns)-2 #remove training and flag field in input dimensionality
    num_hidden_layers = int(param['options']['params']['num_hidden_layers'])
    activation_name = param['options']['params']['activation_name'].strip('\"')
    
    # Map activation functions
    activation_mapping = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh()
    }
    activation_func = activation_mapping[activation_name]
    dropout_rate = float(param['options']['params']['dropout_rate'])
    
    # Convert to PyTorch tensors
    device = torch.device('cpu')
    model['num_hidden_layers'] = num_hidden_layers
    model['input_dim'] = input_dim
    model['nodes_per_layer'] = nodes_per_layer
    model['activation_name'] = activation_name
    model['dropout_rate'] = dropout_rate
    model['dnn'] = SimpleDNN(input_dim, num_hidden_layers, nodes_per_layer, activation_func, dropout_rate).to(device)
    return model







    
# In[314]:


# Train your model
# Returns a fit info json object and may modify the model object
def fit(model,df,param):
    summary_list = {}
    df_train = df[df['is_train'] == 1]
    df_test = df[df['is_train'] == 0]
    X_train = df_train.drop('label', axis=1).drop('is_train', axis=1)
    X_test = df_test.drop('label', axis=1).drop('is_train', axis=1)
    y_train = df_train['label']
    y_test = df_test['label']
    
    print("\nShape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    device = torch.device('cpu')
    
    # Convert pandas Series to NumPy arrays before creating tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device) # Convert Series to numpy array
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)   # Convert Series to numpy array
    
    # Create TensorDataset and DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    learning_rate = float(param['options']['params']['learning_rate'])
    optimizer_name = param['options']['params']['optimizer_name'].strip('\"')

    # Calculate class weights
    total_samples = len(y_train)
    num_class_0 = np.sum(y_train == 0)
    num_class_1 = np.sum(y_train == 1)
    weight_for_class_0 = total_samples / (2.0 * num_class_0)
    weight_for_class_1 = (total_samples / (2.0 * num_class_1)) * class_weight
    class_weights = torch.tensor([weight_for_class_0, weight_for_class_1], dtype=torch.float32).to(device)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model['dnn'].parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model['dnn'].parameters(), lr=learning_rate)
    elif optimizer_name == 'Lion':
        optimizer = Lion(model['dnn'].parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Training loop
    print(f"\nTraining with: Layers={model['num_hidden_layers']}, Nodes={model['nodes_per_layer']}, LR={learning_rate}, "
          f"Batch={batch_size}, Epochs={num_epochs}, Dropout={model['dropout_rate']}, Activation={model['activation_name']}, "
          f"Optimizer={optimizer_name}, Class Weight={class_weight}")
    summary_list['training_settings'] = f"\nTraining with: Layers={model['num_hidden_layers']}, Nodes={model['nodes_per_layer']}, LR={learning_rate}, Batch={batch_size}, Epochs={num_epochs}, Dropout={model['dropout_rate']}, Activation={model['activation_name']}, Optimizer={optimizer_name}, Class Weight={class_weight}"
    start_time = time.time()
    model['dnn'].train()
    loss_list = []
    epoch_list = []
    for epoch in range(num_epochs):
        epoch_list.append(epoch+1)
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model['dnn'](inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model['dnn'].parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
    
        epoch_loss = epoch_loss / len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        loss_list.append(round(epoch_loss,4))
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    summary_list['training_time'] = f"Training completed in {training_time:.2f} seconds"
    summary_list['epoch_number'] = epoch_list
    summary_list['loss_list'] = loss_list
    summary_list['final_loss'] = round(epoch_loss,4)
    with open(MODEL_DIRECTORY + "dnn_lab_loss.json", 'w') as file:
        json.dump(summary_list, file)
    
    return model







    
# In[316]:


# Apply your model
# Returns the calculated results
def apply(model,df,param):
    try:
        X = df.drop('label', axis=1)
    except:
        X = df
    try:
        X = X.drop('is_train', axis=1) 
    except:
        X = df
    try:
        device = torch.device('cpu')
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
        model['dnn'].eval()
        predictions = []
        with torch.no_grad(): 
            for i in range(X_tensor.shape[0]): 
                inputs = X_tensor[i:i+1] 
                output = model['dnn'](inputs)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.tolist()[0])   
        cols = {"Result": predictions}
        result = pd.DataFrame(data=cols)
    except Exception as e:
        cols = {"Error in Inference": [str(model)]}
        result = pd.DataFrame(data=cols)
    return result







    
# In[318]:


# Save model to name in expected convention "<algo_name>_<model_name>"
def save(model,name):
    model_path = MODEL_DIRECTORY + name + ".pth"
    torch.save(model['dnn'], model_path)
    with open(MODEL_DIRECTORY + name + ".json", 'w') as file:
        model_files = model.copy()
        model_files.pop('dnn', None)
        json.dump(model_files, file)
    return model







    
# In[320]:


# Load model from name in expected convention "<algo_name>_<model_name>"
def load(name):
    model = {}
    with open(MODEL_DIRECTORY + name + ".json", 'r') as file:
        model_params = json.load(file)

    input_dim = model_params['input_dim']
    num_hidden_layers = int(model_params['num_hidden_layers'])
    activation_name = model_params['activation_name']
    # Map activation functions
    activation_mapping = {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh()
    }
    activation_func = activation_mapping[activation_name]
    dropout_rate = float(model_params['dropout_rate'])
    device = torch.device('cpu')
    nodes_per_layer = model_params['nodes_per_layer']

    model['dnn'] = SimpleDNN(input_dim, num_hidden_layers, nodes_per_layer, activation_func, dropout_rate).to(device)
    model_path = MODEL_DIRECTORY + name + ".pth"
    model['dnn'] = torch.load(model_path, weights_only=False)
    return model









    
# In[323]:


# Return a model summary
def summary(model=None):
    try:
        with open(MODEL_DIRECTORY + "dnn_lab_loss.json", 'r') as file:
            loss_info = json.load(file)
    except:
        loss_info = {'training_settings': "None", 'training_time': "None", 'epoch_number':"None", 'loss_list':"None", 'final_loss': "None"}
    returns = loss_info
    return returns







