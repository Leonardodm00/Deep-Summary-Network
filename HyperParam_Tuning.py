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

Parameters' name:
    
    Hidden size : GRU width
    Layers GRU : GRU depth
    Layers MLP : MLP depth
    Embedding size : size of the embedding vector
    



Main steps:
    
    
- Fix the reservoir size;
- Fix one of the interdependendt hyperparameters;
- First random search with full range sizes;
- Second random search: unimportant HP are remuved and the remaining ones' 
                ranges are narrowed within the 10% of the best values;
- Choose median or best HP values but the ridge;
- Check robustness of previously fixed HPs;
- Evaluate the best reservoir's size (trade-off btw loss and computation time);
- Find the best regularization parameter for the chosen size. 



Searching algorithm: Bayesian Optimization by Gaussian Process.

This pipeline requires to use K-fold cross validation.   

Evaluation metrics are Normalized Root Mean Squared Error and R^2 correlation coefficient.

REFERENCES:
    
    ESN Overview: A Practical Guide to Applying Echo State Networks.
    
    Evaluation approach: Which Hype for my New Task? Hints and Random Search for Reservoir Computing Hyperparameters.
    
    Bayesian Optimization algorithm: Scikit-optimize library.
    



'''

def GRU_WRAPPER(Params):
    
    
    # ------------- Free cuda -------------
    torch.cuda.empty_cache()



    # -------------------------- TRAINING DATA --------------------------

    # ------------- Load and compute data -------------

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
    dataset_train = TimeSeriesDataset(Training_data,fs=fs_downsampled,window_size_s=100)




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
    # print(f"Using device: {device}")


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
    

    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        

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
        

        # Log the running loss averaged per batch
       

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
         
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        epoch_number += 1
        
    del model
    del loss_fn   
    return best_vloss


