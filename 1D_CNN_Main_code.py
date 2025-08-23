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
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from datetime import datetime
directory = r"C:\Users\Admin\Desktop\Leonardo\Summary Networks"  # Replace with your desired path
os.chdir(directory)
from CNN_functions import *

from skopt.space import Real, Integer
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
# import torch
import scipy.io
import pdb

# '''

# This code implements a flexible 1D_CNN architecure which itself can undergo hyperparameter tuning. It is insired on the work of Radosavovic et Al. (2020)

# ---------------------------------- HYPERPARAMETER TUNING ----------------------------------
# I can also make a shallow analysis on the various searched parameters influence on the reduction of the loss. Looking at the marginal profile. If it is flat than no correlation stands
# otherwise the HP afffects the performance.
# Two different tuning procedures:
    # 1) Network architecture: for the body this focuses on depth, wm and block type. Gropus will be fixed to the suggested size of 16 and bottleneck fixed to 1. (d2l)
    #                       Instead for the head: width_shrink and the embedding size.
    # 2) Training HP: ...   !!!




# ------------- Network Architecture -------------


# Parameters' name:
    
#     Depth: d  ... It actually is the power of two --> if d=3 than the depth is 8 integer
#     Width multiplier: wm [1,4] integer 
#     Block type: blk [0,1] integer ... 0 = ResNet, 1 = ResNeXt
#     Width shrink: ws  [2,6] Integer
#     Embedding size: es [8,16] integer
    
# NOTE: I don't think there are interdependent HP in this case, I just have some doubts about depth and width.

# Main steps:
    
# Random initialization of the kernels and weigths results in Different network realizations which have different optimal hyperparameters which can 
# significanlty vary from one newtork to another. This suggests that different randomly generated networks need to be optimize independently.  

# Different trials with different seed shall be done, for this reason for each HP set 10 separate trainings will be done and the oveall mean final loss evaluated.
# In this pipeline there is a foundamental differnece with the epoch-based approach. At every iteration the model parameters are reset. Instead in th eepoch-based training the 
# params are kept toroughout the epochs.



# - First random search with full range sizes;
# - Second random search: unimportant HP are remuved and the remaining ones' 
#                 ranges are narrowed within the 10% of the best values;
# - Choose median or best HP values;


# ------------- Network/Training HP -------------

# Network and miner's margins

#                    ......







# Searching algorithm: Bayesian Optimization by Gaussian Process.

# This pipeline requires to use K-fold cross validation. (Doing some sort of it)  

# Evaluation metric is the average loss_fun. 


# REFERENCES:
    
#     ESN Overview: A Practical Guide to Applying Echo State Networks.
    
#     Evaluation approach: Which Hype for my New Task? Hints and Random Search for Reservoir Computing Hyperparameters.
    
#     Bayesian Optimization algorithm: Scikit-optimize library.















# NOTES:


#     1) Tunable hyperparameters:
    
#         Model: width, Kernel sizes, embedding size, depth, dropout, if LeakyReLu is used: negative slope magnitude etc...
        
#         Optimizator: learning rate, weight decays (the two betas), etc...
        
        
#     2) Data labels: The traces of reference (control or specific pathologies) have ALL the same label. While surrogate ones MUST have only some common labels
#             to optimize training. Indeed if the Negative labels are ALL the same the trining would be inefficient and the might even not
#             converge at all. Therefore the strategy is to construct groups of negative instances by permutating the orginal mini-batch and then 
#             generate some relative positives.
                
                
#     3) Dataloader: we are handling timeseries data. The shuffle method randomly shuffles the single samples leading to a complete destruction
#             of the intrinsic temporal relationships. DO NOT USE IT if the dataseet is passed to the default DataLoader. With the costume one,
#             however, we devide the dataset into chunks (windows) which instead can be randomly chosen and shuffle set to True.
#             Batch_size determines the number of samples per batch, in the costum function a single sample is an entire window of size defined
#             (in seconds) fixed in the dataset_--- variable initialized before.
            
            
#     4) Validation loss function: The aim of the entire script is to set up a network that performs an embedding that lays the target neuronal dynamic 
#                     traces nearby and all others quite far. For this reason validation loss function is an euclidean distance. The 'MARGIN LOSS'
#                     function utilises the L2 norm (Euclidean distance),it works better than the contrastive loss. 
#                     Refer to: https://gombru.github.io/2019/04/03/ranking_loss/
                    
                    
                
#     5) In validation settings the Positive instances are not much variated from the referenc, just the shift is incremented to simulate the lag on an in-silico network. 
#     We will take two instances of control traces (two different control networks traces concatenated) and evaluate the tendency of the summary network to produce similar embeddings for such traces

 
#     6) The ADAMW algorithms and the embedding regularizer both have regularization mechanisms, it's not clear on what they act.
    
#     7) The MSE threshold used to distinguish Positive and Negative examples is highly sensitive to the overall time series length.

#     8) The architecutre (especially the depth and width) differently process the sequence of data given the fs_downsample.

#     9) When embeddings are normalized, the euclidean norm equals the cosine similarity as distance metrics for the embeddings

#    10) The 'choose_subset' parameter in Data Augmentation function allows to take out randomly a certain number of positive and negative instances



    
# '''









# ------------- Free cuda -------------
torch.cuda.empty_cache()


Dataset_training_window_size_s = 200



# --------------------- PREPARE THE DATASET ---------------------
Char_folder_array = [r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11',r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well17']
Char_base_array = ['ptrain_Control00_Well11_','ptrain_Control00_Well17_']

data_array = []
for j in range(2):
    
    
    smoothed_cumulative,fs_downsampled,MBD = Neuronal_traces(Visible=True,Char_folder=Char_folder_array[j],Char_base=Char_base_array[j],w_size=0.02,Gaussian_window=0.04)
    
    
    cumulative_stdz = Standardization(smoothed_cumulative)
    
    data_array.append( torch.unsqueeze(torch.from_numpy(cumulative_stdz),0).float() )

data = torch.cat((data_array[0], data_array[1]), dim=1)


#%


#%
# ---------------- TRAINING DATA LOADER ----------------

Dataset_training = TimeSeriesDataset(data,fs=fs_downsampled,window_size_s=Dataset_training_window_size_s)
# Create the DataLoader
Dataloader_training = DataLoader(Dataset_training, 
                        batch_size=1, 
                        sampler=RandomSampler(Dataset_training,replacement=True, num_samples=200),
                        shuffle=False, 
                        drop_last=True) # drop_last is important to ensure all batches have the same size


# Set the length of trianing data to correctly initialize the network
window_size_temp = Dataset_training_window_size_s*fs_downsampled # in samples

# Find the window size closest to a power of two
Training_data_length = closest_power_of_2(window_size_temp)


#%

#%
# ---------------------------------------------------------------

#%%


# --------------------- INITIALIZE THE NETWORK ---------------------

# --- SET DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
       
        
network = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                   head_dropout = True, downsampling_rate=2,groups=8,dropout_pers=0.2,Block_Type = 'ResNeXt_Block',
                   width_shrink = 4, Network_depth = 16, Stage_kernel = 3, embedding_size = 16, 
                   Stem_augmentation = 16
                   
                   , Stem_kernel = 5, Stem_stride = 4, device = device)

      
print(network)       

# ------- Load the network on the device -------
network = network.to(device)
        


# --- Training algorithms ---

reducer = reducers.AvgNonZeroReducer()
loss_fn = losses.TripletMarginLoss(margin=0.4,
                                 distance=CosineSimilarity(), #  is preferred in embedding contexts.
                                 reducer = reducer,
                                 embedding_regularizer = None) # See in the obsidian page 'Deep learning Training Regularizers' the reason


# --- Miner ---
miner_hard = miners.TripletMarginMiner(margin=0.2, type_of_triplets="hard",distance=CosineSimilarity()) 
                                
# --- Optimizer ---
optimizer_fn = torch.optim.AdamW(network.parameters())

#%%
'''
The main parameter to control is the MSE


'''
%matplotlib
n_versions_insatances = 10
n_param_vector = 5
# ----------- CHECK AUGMENTATION VALUES 
Positives,Negatives,Pos_Labels,Neg_Labels = Data_Augmentation(data[:,0:Training_data_length],n_versions_insatances,n_param_vector,intra_knot_dist_range= [0.35,0.1],sigma_scale_range= [0.4,0.01],
                                                              MSE_threshold= 0.01,fs = fs_downsampled, shift_magnitude_s=30,Visible = True,choose_subset=10)



    


#%%
#!!!!!!!!!!!!! DEBUG





%matplotlib


def train_one_epoch(model,dataloader,loss_fn,optimizer_fn,fs,device,miner_hard):
    running_loss = 0.
    last_loss = 0.
    model.train(True)
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    
    # The training loop now iterates over the dataloader
    for i, data_batch in enumerate(dataloader):
        
        # data_batch has shape (1, window_size)
        
        # The mini-batch is moved  to the GPU inside the foward pass
       


        # Zero your gradients for every batch!
        optimizer_fn.zero_grad()

        # Make predictions for this batch. Expected size: [1 x 1 x sequence_length]
        final_embeddings,final_labels = model(torch.squeeze(data_batch,1),fs,State='Training',choose_subset = 15)


        # Miner
        hard_pairs = miner_hard(final_embeddings, final_labels)

        # Compute the loss and its gradients
        loss = loss_fn(final_embeddings, final_labels,hard_pairs)
        loss.backward()

        # Adjust learning weights
        optimizer_fn.step()

        # Gather data and report
        running_loss += loss.item()
        
        
        avg_loss = running_loss / (i+1) # loss per batch
        print('  batch {} loss: {}'.format(i + 1, avg_loss))
       
    return avg_loss     
     






     
        
