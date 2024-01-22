import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import BertModel, BertTokenizerFast
import os
import tqdm
import time

# specify GPU
device = torch.device("cpu")
max_seq_len = 500

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model  
        cls_hs = self.bert(sent_id, attention_mask=mask)
#         print(cls_hs[0][:,0,:])
        x = self.fc1(cls_hs[0][:,0,:])

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x





# this cell is not executed from MLTK and should only be used for staging data into the notebook environment
def stage(name):
    with open("/srv/notebooks/data/"+name+".csv", 'r') as f:
        df = pd.read_csv(f)
    with open("/srv/notebooks/data/"+name+".json", 'r') as f:
        param = json.load(f) 
#         param = {}
    return df, param

def init(df,param):
    model = {}
    return model

def fit(model,df,param):
    model = {}
    return model




def apply(model,df,param):

    # import BERT-base pretrained model
    MODEL_NAME="/srv/app/model/data/classification/en/bert_classification_en"
    bert = BertModel.from_pretrained(MODEL_NAME, local_files_only=True)
    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    df_test = df
    labels = 'ground_truth'
    test_text = df_test.text.values
    # test_labels = df_test[labels].values   

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # for test set
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    # test_y = torch.tensor(test_labels.tolist())
    #load weights of best model
    model_par = BERT_Arch(bert)
    device = torch.device("cpu")
    model_par = nn.DataParallel(model_par)

    # Load fine-tuned model file
    MODEL_DIRECTORY = "/srv/app/model/data/classification/en/bert_phishing"
    model_par.load_state_dict(torch.load(os.path.join(MODEL_DIRECTORY, "pytorch_model.pt"), map_location=torch.device(device)),strict=False)
    state_dict = model_par.module.state_dict()

    model = BERT_Arch(bert)
    model.load_state_dict(state_dict)
    model = model.to(device)

    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
    # Create the DataLoader for our training set
    # data_test_data = TensorDataset(test_seq, test_mask, test_y)
    data_test_data = TensorDataset(test_seq, test_mask)
    data_test_dataloader = DataLoader(data_test_data, batch_size=4, shuffle=False)

    total_batches = len(data_test_dataloader)

    all_logits = []
    # get predictions for test data
    for step, batch in enumerate(data_test_dataloader):

        # b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        b_input_ids, b_attn_mask= tuple(t.to(device) for t in batch)
        with torch.no_grad():
            preds = model(b_input_ids.to(device), b_attn_mask.to(device))
            all_logits.append(preds)

    all_logits = torch.cat(all_logits, dim=0)
    preds = all_logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    ret = []
    for pred in preds:
        if pred == 0:
            ret.append("non-phishing")
        else:
            ret.append("phishing")
    returns = pd.DataFrame(ret,columns=['prediction'])

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



