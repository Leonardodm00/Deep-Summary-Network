# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:37:29 2025

@author: Admin
"""

import torch
from torch import nn
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from pytorch_metric_learning import reducers

'''
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
    
        Model: Hidden size, embedding size, depth, dropout.
        Optimizator: learning rate, weight decays (the two betas), etc...
        
        
    6) Data labels: The traces of reference (control or specific pathologies) have ALL the same label. While surrogate ones have ALL different 
                labels.
'''





class GRUNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, embedding_size, num_layers=1, dropout=0.2):
        super(GRUNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the GRU layer
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        
       # Define a sequence of three fully connected layers
        # with ReLU activation functions in between
        self.fc = nn.Sequential(
            # First fully connected layer
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            
            # Second fully connected layer
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # Third and final fully connected layer
            nn.Linear(64, embedding_size)
        )


    def forward(self, x, h0=None):
            if h0 is None:
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            out, h_n = self.gru(x, h0)
            
            last_hidden_state = out[:, -1, :]
            
            final_embedding = self.fc(last_hidden_state)
            
            return final_embedding
    


# --------- TRAINING FUNCTIONS ---------

model = GRUNetwork(3,128,16)


# --- Design the loss function ---

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.reducers import MultipleReducers
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer



reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1,
                                 distance=LpDistance(),
                                 reducer = reducer,
                                 embedding_regularizer = LpRegularizer())



                                
# --- Optimizer ---
optimizer_fn = torch.optim.AdamW(model.parameters())

# --- Online miners ---  NOT USED FOR NOW!
# miner_fn = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all",)



def train_one_epoch(model,loss_fn,optimizer_fn,miner_fn,training_loader):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer_fn.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer_fn.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0

    return last_loss
