# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:37:29 2025

@author: Admin
"""
import os
import torch
from torch import nn
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from pytorch_metric_learning import reducers
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import MultipleReducers
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from pytorch_metric_learning.distances import LpDistance
from datetime import datetime
directory = r"C:\Users\Admin\Desktop\Leonardo\Summary Networks"  # Replace with your desired path
os.chdir(directory)
from Data_Extraction_Augmentation import *
'''
Critical Points:
    
    1) Batch Normalization




Some notes:
    1) X = input data (batch or minibatch) size: nxd where n = # of examples and d = # of features.
    
    2) Input variables:
        input_size – The number of expected features in the input x
    
        hidden_size – The number of features in the hidden state h
    
        num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together
            to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
    
        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) 
            instead of (seq, batch, feature). Note that this does not apply to hidden or cell states.
                See the Inputs/Outputs sections below for details. Default: False
    
        dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer,
            with dropout probability equal to dropout. Default: 0
    
        bidirectional – If True, becomes a bidirectional GRU. Default: False

    3) Usually the hidden size is in the range (64,2056) while the depth of a deep network is in the range (1,8).
    
    4) In a deep network we might be interested in consulting all the layers' final state. INdeed:
        out, h_n = self.gru(x, h0) where out =  It contains the hidden state for every single time step of the last GRU layer
                                         h_n = This is the final hidden state of the GRU. It holds the hidden state of each 
                                             stacked GRU layer at the very end of the sequence. For a single-layer GRU (num_layers=1),
                                             h_n is effectively the same as out[:, -1, :] but in a different tensor format.


    5) Tunable hyperparameters:
    
        Model: Hidden size, embedding size, depth, dropout, if LeakyReLu is used: negative slope magnitude, 
        
        Optimizator: learning rate, weight decays (the two betas), etc...
        
        
    6) Data labels: The traces of reference (control or specific pathologies) have ALL the same label. While surrogate ones MUST have only some common labels
            to optimize training. Indeed if the Negative labels are ALL the same the trining would be inefficient and the might even not
            converge at all. Therefore the strategy is to construct groups of negative instances by permutating the orginal mini-batch and then 
            generate some relative positives.
                
                
    7) Dataloader: we are handling timeseries data. The shuffle method randomly shuffles the single samples leading to a complete destruction
            of the intrinsic temporal relationships. DO NOT USE IT if the dataseet is passed to the default DataLoader. With the costume one,
            however, we devide the dataset into chunks (windows) which instead can be randomly chosen and shuffle set to True.
            Batch_size determines the number of samples per batch, in the costum function a single sample is an entire window of size defined
            (in seconds) fixed in the dataset_--- variable initialized before.
            
            
    8) Validation loss function: The aim of the entire script is to set up a network that performs an embedding that lays the target neuronal dynamic 
                    traces nearby and all others quite far. For this reason validation loss function is an euclidean distance. The 'CONSTRASTIVE LOSS'
                    function utilises the L2 norm (Euclidean distance)
                    
                    
                
    9) In validation settings the Positive instances are not much variated from the referenc, just the shift is incremented to simulate the lag on an in-silico network. 
    We will take two instances of control traces (two different control networks traces concatenated) and evaluate the tendency of the summary network to produce similar embeddings for such traces
       

        
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
    10) The ADAMW algorithms and the embedding regularizer both have regularization mechanisms, it's not clear on what they act.
'''




# ------------- Free cuda -------------
torch.cuda.empty_cache()

#%%

# -------------------------- TRAINING DATA --------------------------

# ------------- Load and compute data -------------
%matplotlib
Char_folder = r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11'
Char_base = 'ptrain_Control00_Well11_'


# --- Dynamics ---
Projected_trajectories,Variance_explained,fs_downsampled = Neuronal_traces(Visible=False,Char_folder=Char_folder,Char_base=Char_base)

# --- Standardization ---<
Projected_trajectories = Standardization(Projected_trajectories[:,0:3])

# Torch format
Training_data = torch.from_numpy(Projected_trajectories)



# --------- Construct Dataset and Dataloader ---------


# Create an instance of your custom dataset
dataset_train = TimeSeriesDataset(Training_data,fs=fs_downsampled,window_size_s=200)




#%
# Create the DataLoader
dataloader_train = DataLoader(dataset_train, 
                        batch_size=1, 
                        sampler=RandomSampler(dataset_train,replacement=True, num_samples=100),
                        shuffle=False, 
                        drop_last=True) # drop_last is important to ensure all batches have the same size




# -------------------------- VALIDATION DATA --------------------------

# ------------- Load and compute data -------------
# %matplotlib
Char_folder = r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well23'
Char_base = 'ptrain_Control00_Well23_'


# --- Dynamics ---
Projected_trajectories,Variance_explained,fs_downsampled = Neuronal_traces(Visible=False,Char_folder=Char_folder,Char_base=Char_base)

# --- Standardization ---<
Projected_trajectories = Standardization(Projected_trajectories[:,0:3])

# Torch format
Validation_data = torch.from_numpy(Projected_trajectories)



concatenated_data = torch.cat((Training_data, Validation_data), dim=0)

# Create an instance of your custom dataset
dataset_validation= TimeSeriesDataset(concatenated_data,fs=fs_downsampled,window_size_s=400)




#%
# Create the DataLoader
dataloader_validation = DataLoader(dataset_validation, 
                        batch_size=1, 
                        sampler=RandomSampler(dataset_train,replacement=True, num_samples=100),
                        shuffle=False, 
                        drop_last=True) # drop_last is important to ensure all batches have the same size






# --- SET DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------- TRAINING FUNCTIONS ---------

model = GRUNetwork(3,64,16)

reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1,
                                 distance=LpDistance(),
                                 reducer = reducer,
                                 embedding_regularizer = LpRegularizer())



                                
# --- Optimizer ---
optimizer_fn = torch.optim.AdamW(model.parameters())

# --- Move objects in the set device ---
model = model.to(device)



# --- Online miners ---  NOT USED FOR NOW!
# miner_fn = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all",)


# iterations = 10
# avg_loss = train_one_epoch(model,dataloader_train,loss_fn,optimizer_fn,iterations,fs_downsampled)






# --------------- EPOCHS ---------------


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    
    iterations = 10
    avg_loss = train_one_epoch(model,dataloader_train,loss_fn,optimizer_fn,iterations,fs_downsampled)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(dataloader_validation):
            
            vdata = torch.squeeze(vdata).to(device)
            
            final_embeddings,final_labels = model(vdata,fs_downsampled,State='Validation')
        
            
            vloss = loss_fn(final_embeddings, final_labels)
            
            
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
   

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1










#%%
del model
del loss_fn
