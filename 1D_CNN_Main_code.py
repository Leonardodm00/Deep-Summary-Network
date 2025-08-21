# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 20:59:29 2025

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
from CNN_functions import *





# ------------- Free cuda -------------
torch.cuda.empty_cache()


Dataset_training_window_size_s = 200



# --------------------- PREPARE THE DATASET ---------------------
Char_folder_array = [r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11',r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well17']
Char_base_array = ['ptrain_Control00_Well11_','ptrain_Control00_Well17_']

data_array = []
for j in range(2):
    
    
    smoothed_cumulative,fs_downsampled,MBD = Neuronal_traces(Visible=False,Char_folder=Char_folder_array[j],Char_base=Char_base_array[j])
    
    
    cumulative_stdz = Standardization(smoothed_cumulative)
    
    data_array.append( torch.unsqueeze(torch.from_numpy(cumulative_stdz),0).float() )

data = torch.cat((data_array[0], data_array[1]), dim=1)



# Set the length of trianing data to correctly initialize the network
window_size_temp = Dataset_training_window_size_s*fs_downsampled # in samples

# Find the window size closest to a power of two
Training_data_length = closest_power_of_2(window_size_temp)



# ---------------- TRAINING DATA LOADER ----------------

Dataset_training = TimeSeriesDataset(data,fs=fs_downsampled,window_size_s=Dataset_training_window_size_s)
# Create the DataLoader
Dataloader_training = DataLoader(Dataset_training, 
                        batch_size=1, 
                        sampler=RandomSampler(Dataset_training,replacement=True, num_samples=200),
                        shuffle=False, 
                        drop_last=True) # drop_last is important to ensure all batches have the same size


# ---------------------------------------------------------------

#%%


# --------------------- INITIALIZE THE NETWORK ---------------------

# --- SET DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
       
        
network = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                   head_dropout = True, downsampling_rate=2,groups=8,dropout_pers=0.2,Block_Type = 'ResNeXt_Block',
                   width_shrink = 4, Network_depth = 16, Stage_kernel = 3, embedding_size = 16, 
                   Stem_augmentation = 16, Stem_kernel = 5, Stem_stride = 4, device = device)

      
print(network)       

# ------- Load the network on the device -------
network = network.to(device)
        


# --- Training algorithms ---
reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1,
                                 distance=LpDistance(),
                                 reducer = reducer,
                                 embedding_regularizer = LpRegularizer())



                                
# --- Optimizer ---
optimizer_fn = torch.optim.AdamW(network.parameters())

#%%
       

    



#!!!!!!!!!!!!! DEBUG








def train_one_epoch(model,dataloader,loss_fn,optimizer_fn,fs,device):
    running_loss = 0.
    last_loss = 0.
    model.train(True)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    
    # The training loop now iterates over the dataloader
    for i, data_batch in enumerate(dataloader):
        
        # data_batch has shape (batch_size, window_size, features)
        
        # Move the mini-batch to the GPU
        data_batch = data_batch.to(device)


        # Zero your gradients for every batch!
        optimizer_fn.zero_grad()

        # Make predictions for this batch. Expected size: [1 x sequence_length]
        final_embeddings,final_labels = model(torch.squeeze(data_batch),fs,State='Training')



        #!!!: PUT SOMEWHERE THE MINER



        # Compute the loss and its gradients
        loss = loss_fn(final_embeddings, final_labels)
        loss.backward()

        # Adjust learning weights
        optimizer_fn.step()

        # Gather data and report
        running_loss += loss.item()
        
        
        avg_loss = running_loss / (i+1) # loss per batch
        print('  batch {} loss: {}'.format(i + 1, avg_loss))
       
    return avg_loss     
     
     
avg_loss = train_one_epoch(network,Dataloader_training,loss_fn,optimizer_fn,fs_downsampled,device)

