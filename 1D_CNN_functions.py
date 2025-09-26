
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler
from torch import nn
import torch.nn.functional as F

import math
from collections import Counter
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import scipy.io

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import pdb

from skopt.space import Real, Integer

from pytorch_metric_learning import losses, miners, reducers
from pytorch_metric_learning.distances import LpDistance, BatchedDistance, CosineSimilarity
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
from pytorch_metric_learning.regularizers import LpRegularizer


'''

The following code is insired to the AnyNet architecure, whith some changes. The key idea is to 
set up aa flexible network design space. Such space is defined as a distribution of networks 
which is optimizes in a way to obtain good performance for the entire families of networks.


Following the design principles suggested by Radosavovic:
    1) Keep the bottleneck ratio equal to 1 for all stages.
    2) All stages share the same group width.
    3) Increase linearly the netwrok width ascross stages.
    4) Increase the network depth across stages, the number of blocks per stage.

As the resolution is reduced both depth and width of the stages increases.
    
Each stage is made up of 'd' # of blocks which share the same architecture. Whithin a single stage 
the first block performs downsampling and increase in width (# of channels) while the following ones 
keep both resolution and # of channels unchanged.


--------- STAGES ---------
In defining the number of stages and relative width and depth I rely on the pricniples given by
Radosavovic et Al. (2020). The actual depth is the selected one minus 2 for math reasons.




--------- HEAD ---------

The head will be a Fully connected perceptron working on the pooled feature map of all the channels.
Additionaly I can add shorcuts form the channels at various degree of resolution. The number of layers
(and relative width) depends on the parameter width_shrink. It is a power of 2 and rules the
the successive layer's width. The relation is the following:
    
        width_post_layer = width_pre_layer/width_shrink
        
The number of head Layers is retrived in the following way:
    
    (width_shrink)^n_layers = width_input/embedding_size
    
which leads to 

    n_layers = log(width_input/embedding_size)/log(width_shrink)


---------- DATA AUGMENTATION ----------
Data augmentatio nis performed as follows.
FIrstly time series undergo, separately, to magnitude and temporal warping. Afterwards MSE is calculated between
the original time series and the augmented ones to determine, through a threshold, which are positive instances, leaving
the remaining ones as negatives. It follows homogenous temporal shift.
BE AWARE that the threshold for the MSE is highly sensitive to the overall time series length.

------------------------------------

NOTES:
    1) The architecture follows the one of RegNetX- where - is the number of layers.
    2) The stem will do a first coarse graining of the timeseries.
    3) It is possible to add a droput layer in the single bloks. This allows to  implement it
        for the last block in a stage.
    4) Dropout position is controvelsial, some suggest ot put it just after the linear projection (the conv. layer)
        others after the non linear activation function.
'''


# ---------------------- UTILITY FUNCTIONS ----------------------

def random_sample_without_replacement(arr: torch.Tensor, num_samples: int):
    """
    Randomly samples without replacement from a PyTorch tensor.

    Args:
        arr (torch.Tensor): The input tensor to sample from.
        num_samples (int): The number of items to sample.

    Returns:
        torch.Tensor: A new tensor containing the randomly sampled items.
    
    Raises:
        ValueError: If num_samples is greater than the size of the tensor.
    """
    if num_samples > arr.size()[0]:
        raise ValueError("Number of samples cannot be greater than the tensor size.")

    indices = torch.randperm(arr.size()[0])[:num_samples]
    
    
    return indices



class TimeSeriesDataset(Dataset):
    def __init__(self, full_time_series, window_size_s=100,fs=None ):
        """
        Initializes the dataset with the full time series and window size.
        window size in samples will be a power of two
        
        Args:
            full_time_series (torch.Tensor): The entire time series data
                                             with shape (1,total_samples).
            window_size_s (int): The size of the sliding window in seconds must be power of 2
        """
        self.full_time_series = full_time_series
        window_size_temp = window_size_s*fs # in samples
        
        # Find the window size closest to a power of two
        self.window_size = closest_power_of_2(window_size_temp)
        
        print('---------------- Mini-batch window --------------------')
        print('')
        print('Actual window size (in samples): ',self.window_size )
        print('Actual window size (in seconds): ',self.window_size/fs )
        print('')
        print('------------------------------------------------------')
        
        # Calculate the number of possible windows in the time series.
        # This assumes non-overlapping windows for simplicity.
        # If you want overlapping windows, the calculation would be different.
        self.num_windows = int((self.full_time_series.shape[1] // self.window_size))

    def __len__(self):
        """
        Returns the total number of windows (samples) in the dataset.
        """
        return self.num_windows

    def __getitem__(self, idx):
        """
        Returns a single window from the dataset.
        
        Args:
            idx (int): The index of the window to retrieve.
        
        Returns:
            torch.Tensor: A single time series window with shape (1, window_size).
        """
        start_idx = int(idx * self.window_size)
        end_idx = int(start_idx + self.window_size)
        window = self.full_time_series[0,start_idx:end_idx]
        
        
        
        return window.reshape(1, -1)
def closest_power_of_2(n):
    """
    Finds the closest power of 2 to a given number n,
    which is still less than or equal to n.
    """
    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("Input must be a positive number.")

    # Calculate the logarithm base 2
    exponent = math.log2(n)

    # Round the exponent down to the nearest integer
    # This ensures the result is always <= n
    floored_exponent = math.floor(exponent)

    # Calculate the power of 2
    result = 2 ** floored_exponent
    return result
def analyze_vector(data_vector):
    """
    Analyzes a vector of values to determine the number of unique elements
    and the frequency of each element.

    Args:
        data_vector (list): A list of values.

    Returns:
        tuple: A tuple containing the number of unique elements and a dictionary
               of element frequencies.
    """
    # Determine the number of unique elements by converting to a set and getting its length
    # num_unique = list(dict.fromkeys(data_vector))
    
    # Use Counter to get the frequency of each element
    Counter_obj = Counter(data_vector)
    
    return list(Counter_obj.keys()), list(Counter_obj.values())
def stage_features(depth_max=None,w_in=None,w_m=2):
    '''
    The approach is inspired from the work of Radosavovic (2020)
    
    Hyperparameters are:
        1) Depth_max (<64)
        2) w_a which is taken equal to the stem's output
        3) w_m taken equal to 2 (at each stage the depth is doubled)
    
    '''
    list_depth = range(1,depth_max+1)
    
    list_out = []
    
    for j in list_depth:
        
        uj = w_in*(j+1)
        
        sj = int(math.log(j+1)/math.log(w_m))
        
        wj = w_in*(w_m**sj)
        
        list_out.append(wj)
    
    
    num,freq = analyze_vector(list_out[:-2]  )
    return num,freq 

def calculate_padding(input_size, kernel_size, stride):
    """
    Calculates the required padding for a 1D convolution to ensure the output 
    sequence length is the same as the input length.

    Args:
        input_size (int): The length of the input sequence.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.

    Returns:
        int: The amount of padding required.
    """
    # Check for valid stride
    if stride <= 0:
        raise ValueError("Stride must be a positive integer.")
    
    # Calculate padding using the formula for same-padding
    # This formula ensures that the output size is roughly equal to the input size
    # for a given kernel size and stride.
    output_size = math.ceil(input_size / stride)
    padding = (output_size - 1) * stride + kernel_size - input_size
    
    # For same padding, the required padding must be an even number.
    # The output size will be input_size / stride, but since padding must be an integer,
    # we need to consider how the padding is split.
    if padding % 2 != 0 and padding != 1:
        # If padding is odd, it cannot be perfectly split, so we might need to adjust.
        # This function returns the total padding needed.
        # In practice, PyTorch handles asymmetric padding.
        raise ValueError("Padding is odd.")

    if padding == 1:
        return padding
    
    else:
        return padding // 2
# --------------------------------- DATA PREPROCESSING ---------------------------------   




def Standardization(data):
    """
    Standardizes a time series (univariate or multivariate) by standardizing each feature (column) separately.

    Standardization (Z-score normalization) transforms the data to have a mean of 0 and a standard deviation of 1.
    The formula for standardization is: z = (x - mu) / sigma, where mu is the mean and sigma is the standard deviation.

    Args:
        data (np.ndarray): A NumPy array representing the time series.
                          It can be 1D (univariate) or 2D (multivariate) of shape (timesteps, features).

    Returns:
        np.ndarray: A NumPy array of the same shape as `data`, but with each feature standardized.
                    Returns None if the input data is not a 1D or 2D array.
    """
    

    

   
    mean = np.mean(data)
    std_dev = np.std(data)
    
    
    return (data-mean)/std_dev

        # Handle the case where the standard deviation is zero
      
def Get_IFR(data, fs, Cumulative, t_vec, step_s, bin_size, Isolate_NB, T_max):
    """
    Calculates the Instantaneous Firing Rate (IFR) of the neuronal data.

    Args:
        data (list): A list of spike timings for each channel.
        fs (int): Sampling frequency in Hz.
        Cumulative (numpy.ndarray): The cumulative global activity.
        t_vec (numpy.ndarray): The time vector for the cumulative activity.
        step_s (float): Step size for the cumulative activity calculation.
        bin_size (float): Bin size in seconds.
        Isolate_NB (bool): If True, isolates IFR for neurobursts.
        T_max (int): Total recording time in samples.

    Returns:
        tuple: A tuple containing:
            IFR (list or numpy.ndarray): The calculated IFR.
            bin_size (float): The bin size in samples.
            window_size (int): The window size for NB analysis in samples,
                               or None if Isolate_NB is False.
    """
    bin_size_samples = int(bin_size * fs)  # [samples]

    if Isolate_NB:
        # Construct the window that will be centered at the NB's peak
        pre_w = 1.5  # Pre samples [s]
        post_w = 3.5 # post samples [s]

        # Scale to samples
        pre_w_samples = int(pre_w * fs)
        post_w_samples = int(post_w * fs)

        # Define the window size for calculating the IFR
        window_size = pre_w_samples + post_w_samples  # [samples]
        num_bins = window_size // bin_size_samples

        # Isolate the NB timings (peaks location).
        mean_IFR = np.mean(Cumulative)
        std_IFR = np.std(Cumulative)
        
        # The 'distance' argument for find_peaks is in samples, not seconds.
        # It should be based on the sampling of 'Cumulative', which is 'step_s'.
        # The MATLAB code uses 3 * fs / step_s, where fs is the original sampling rate.
        # This seems to be a scaling factor. We'll replicate it.
        min_peak_distance_samples = int(3 * fs / step_s)
        
        # 'height' in scipy.signal.find_peaks is the equivalent of MinPeakHeight
        idx, _ = find_peaks(Cumulative.flatten(), height=mean_IFR + std_IFR, distance=min_peak_distance_samples)
        NB_T = t_vec[idx]

        IFR = [None] * len(NB_T)
        num_channels = len(data)

        for nb_idx, nb_time in enumerate(NB_T):
            binned_NB = np.zeros((num_channels, num_bins))

            lower_bound = nb_time - pre_w_samples
            upper_bound = nb_time + post_w_samples

            # Extract per each channel the spikes within the NB window
            for ch_idx in range(num_channels):
                data_ = data[ch_idx]

                # Bins for this specific NB
                for bin_idx in range(num_bins - 1):
                    # Define the start and stop timings
                    start_idx = int(lower_bound + bin_idx * bin_size_samples)
                    stop_idx = int(lower_bound + (bin_idx + 1) * bin_size_samples)

                    # Extract spikes within the window (inclusive of the boundaries)
                    within_range = (data_ >= start_idx) & (data_ < stop_idx)

                    # Count and store the number of spikes
                    number_spks = np.sum(within_range)
                    binned_NB[ch_idx, bin_idx] = number_spks

            IFR[nb_idx] = binned_NB

    else:
        window_size = None
        num_channels = int(len(data))
        num_bins = int(T_max // bin_size_samples)
        IFR = np.zeros((num_bins, num_channels))

        for ch_idx in range(num_channels):
            data_ = data[ch_idx]

            for bin_idx in range(num_bins):
                start_idx = bin_idx * bin_size_samples
                stop_idx = (bin_idx + 1) * bin_size_samples
                
                within_range = (data_ >= start_idx) & (data_ < stop_idx)
                
                number_spks = np.sum(within_range)
                
                IFR[bin_idx, ch_idx] = number_spks

    return IFR, bin_size_samples, window_size


def Rect_window(fs, w_size_s, overlap_s, x, T_max):
    """
    Calculates cumulative activity using a sliding rectangular window.

    Args:
        fs (int): Sampling frequency in Hz.
        w_size_s (float): Window size in seconds.
        overlap_s (float): Overlap between windows in seconds.
        x (list): A list of spike timings for each channel.
        T_max (int): Total recording time in samples.

    Returns:
        tuple: A tuple containing:
            Cumulative (numpy.ndarray): The cumulative activity. NOT NORMALIZED
            t_vec (numpy.ndarray): The time vector for the cumulative activity.
            step_size (int): The step size between windows in samples.
    """
    w_size = int(w_size_s * fs)
    overlap = int(overlap_s * fs)
    
    step_size = w_size - overlap
    
    # In MATLAB, '0:step_size:T_max' is inclusive, so we need to adjust np.arange.
    t_vec = np.arange(0, T_max, step_size)
    Cumulative = np.zeros(len(t_vec))
    
    start_index = 0
    idx = 0
    while start_index + w_size <= T_max and idx < len(t_vec):
        temp_cum = 0
        for ch in x:
            data_ = ch
            
            # The MATLAB code uses start_index + w_size-1, which is correct for 1-based indexing.
            # For Python, we use the end index exclusively.
            within_range = (data_ >= start_index) & (data_ < start_index + w_size)
            
            count = np.sum(within_range)
            
            temp_cum += count
            
        # Cumulative[idx] = temp_cum / w_size
        Cumulative[idx] = temp_cum
        
        start_index += step_size
        idx += 1
        
    return Cumulative, t_vec, step_size
    
def get_PCA(NB_IFR_smoothed_concatenated, IFR_smoothed, Isolate_NB,Visible):
    """
    Performs Principal Component Analysis (PCA) on the IFR data.

    Args:
        NB_IFR_smoothed_concatenated: The concatenated smoothed IFR data.
                                      If Isolate_NB is True, this is used for PCA.
        IFR_smoothed: The smoothed IFR data, which can be a list of arrays (Isolate_NB=True)
                      or a 2D array (Isolate_NB=False).
        Isolate_NB: A boolean indicating whether to perform PCA on concatenated
                    neuroburst (NB) data or the total signal.
    
    Returns:
        Variance_explained: The percentage of variance explained by each PC.
        Projected_trajectories: The data projected onto the new PCA space.
        Coefficients: The principal component coefficients (eigenvectors).
        NB_IFR_PCA_mean: The mean PCA trajectory (if Isolate_NB is True).
    """

    if Isolate_NB:
        # In scikit-learn's PCA, the input data should have shape (n_samples, n_features).
        # MATLAB's pca assumes rows are observations and columns are variables.
        # So we transpose the concatenated data.
        NB_IFR_smoothed_concatenated_PCA = NB_IFR_smoothed_concatenated.T

        # Perform PCA
        pca = PCA()
        pca.fit(NB_IFR_smoothed_concatenated_PCA)
        Coefficients = pca.components_.T
        Variance_explained = pca.explained_variance_ratio_

        # Obtain the mean traces
        num_nb = len(IFR_smoothed)
        num_channels = IFR_smoothed[0].shape[0] if num_nb > 0 else 0
        samples_per_window = IFR_smoothed[0].shape[1] if num_nb > 0 else 0
        
        NB_IFR_PCA_mean_ = np.zeros((num_channels, samples_per_window, num_nb))

        for k in range(num_nb):
            nb_ifr_pca_data = IFR_smoothed[k]
            # Project the data
            proj = nb_ifr_pca_data.T @ Coefficients
            NB_IFR_PCA_mean_[:, :, k] = proj.T
        
        # Calculate the mean across the third dimension (k)
        NB_IFR_PCA_mean = np.mean(NB_IFR_PCA_mean_, axis=2).T
        
        # Project the concatenated data
        Projected_trajectories = NB_IFR_smoothed_concatenated_PCA @ Coefficients

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract the first three components
        x = Projected_trajectories[:, 0]
        y = Projected_trajectories[:, 1]
        z = Projected_trajectories[:, 2]

        x_m = NB_IFR_PCA_mean[:, 0]
        y_m = NB_IFR_PCA_mean[:, 1]
        z_m = NB_IFR_PCA_mean[:, 2]
        
        # Use plot3 function to plot the lines
        ax.plot(x, y, z)
        ax.plot(x_m, y_m, z_m, linewidth=4.5, color='red')
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_title('Concatenated NBs')
        
        # To keep the axes scaled appropriately and prevent distortion
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
        
        ax.grid(True)
        plt.show()

    else:
        
        
        # In this case, Isolate_NB is false and IFR_smoothed is a 2D array.
        NB_IFR_PCA_mean = None
        
        # Perform PCA
        pca = PCA()
        pca.fit(IFR_smoothed)
        Coefficients = pca.components_.T
        Variance_explained = pca.explained_variance_ratio_
        
        # Project the data
        Projected_trajectories = pca.transform(IFR_smoothed)


        if Visible == True:
            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract the first three components
            x = Projected_trajectories[:, 0]
            y = Projected_trajectories[:, 1]
            z = Projected_trajectories[:, 2]
            
            # Use plot3 function to plot the lines
            ax.plot(x, y, z)
            
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            ax.set_zlabel('PC 3')
            ax.set_title('Culture Dynamics')
            
            ax.set_box_aspect([1, 1, 1])
            
            ax.grid(True)
            plt.show()

    return Variance_explained, Projected_trajectories, Coefficients, NB_IFR_PCA_mean




def Smoothed_IFR(IFR, bin_size, window_size, fs, Isolate_NB, Gaussian_window, Visible):
    """
    The function takes the raw instantaneous firing rates of the NB-centered
    windows and returns the concatenated and smoothed NB's IFR.
    
    Args:
        IFR: The input IFR data. Its format depends on Isolate_NB.
        bin_size: The size of the time bins.
        window_size: The size of the analysis window.
        fs: The sampling frequency.
        Isolate_NB: If True, IFR is a list of arrays (cell array in MATLAB).
                    If False, IFR is a 2D NumPy array.
        Gaussian_window: The size of the Gaussian smoothing window [s].
        Visible: A boolean to control whether to display plots.
    
    Returns:
        IFR_smoothed: The smoothed IFR data.
        IFR_smoothed_concatenated: The concatenated smoothed IFR data.
    """
    Gaussian_window_samples = Gaussian_window*fs
    if Isolate_NB:
        # MATLAB uses 1-based indexing for size, Python uses 0-based
        num_nb = len(IFR)
        num_channels = IFR[0].shape[0] if num_nb > 0 else 0
        
        # Calculate samples_per_window
        # Assuming Samples_per_window is a global variable in the MATLAB code,
        # we'll calculate it here from the input IFR data.
        if num_nb > 0:
            samples_per_window = IFR[0].shape[1]
        else:
            samples_per_window = 0

        # Create time vector for plotting
        t_vec_nb = np.arange(0, samples_per_window) * bin_size

        if Visible:
            plt.figure()
            for j in range(num_nb):
                ifr_data = IFR[j]
                for i in range(num_channels):
                    channel = ifr_data[i, :]
                    plt.plot(t_vec_nb, channel)
            plt.title('Raw IFR')
            plt.xlabel('Time [s]')
            plt.ylabel('Spikes')
            plt.show()

        IFR_smoothed = [None] * num_nb
        IFR_smoothed_concatenated = np.zeros((num_channels, num_nb * samples_per_window))
        
        # MATLAB's smoothdata('gaussian') is equivalent to a Gaussian filter.
        # We'll use scipy.ndimage.gaussian_filter1d for this.

        for j in range(num_nb):
            ifr_data = IFR[j]
            smoothed_channels = []
            for i in range(num_channels):
                channel = ifr_data[i, :]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
                smoothed_channels.append(smoothed_channel)
                
                # Concatenate the smoothed data
                # MATLAB's n*Samples_per_window + 1 : (n+1)* Samples_per_window
                # is equivalent to n*samples_per_window : (n+1)* samples_per_window in Python
                IFR_smoothed_concatenated[i, j * samples_per_window : (j + 1) * samples_per_window] = smoothed_channel

            IFR_smoothed[j] = np.array(smoothed_channels)

        if Visible:
            plt.figure()
            plt.subplot(2, 1, 1)
            for j in range(num_nb):
                ifr_data = IFR_smoothed[j]
                for i in range(num_channels):
                    channel = ifr_data[i, :]
                    plt.plot(t_vec_nb, channel)
            plt.title(f'Smoothed IFR')
            plt.xlabel('Time [s]')
            plt.ylabel('Spikes')

            plt.subplot(2, 1, 2)
            # The original code plots all channels; Ch = 3 is not used.
            # We'll follow the original code and plot all.
            t_vec_conc = np.arange(IFR_smoothed_concatenated.shape[1])
            plt.plot(t_vec_conc, IFR_smoothed_concatenated.T)
            plt.title(f'Concatenated smoothed NB IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()

    else:
        # IFR is a 2D array: samples x channels
        num_samples, num_channels = IFR.shape
        IFR_smoothed = np.zeros_like(IFR)
        IFR_smoothed_concatenated = []

        if Visible:
            plt.figure()
            plt.subplot(2, 1, 1)
            for i in range(num_channels):
                plt.plot(IFR[:, i])
            plt.title(f'Raw IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            
            plt.subplot(2, 1, 2)
            for i in range(num_channels):
                channel = IFR[:, i]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
                IFR_smoothed[:, i] = smoothed_channel
                plt.plot(smoothed_channel)
            plt.title(f'Smoothed IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()
        else:
            for i in range(num_channels):
                channel = IFR[:, i]
                IFR_smoothed[:, i] = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window_samples)
            
    return IFR_smoothed, IFR_smoothed_concatenated

        

def get_Smoothed_Cumulative(Cumulative,fs_downsampled,Gaussian_window):
    # Gaussian window is the std of the gaussian window. it is defined in s
    # and MUST be grater than the sampling step of fs_downsampled
   
    
    # Gaussian window is defined in s, thus devide by 1000 because fs_downsampled is in Hz
    Gaussian_window_samples = np.ceil(Gaussian_window*fs_downsampled) 
    
    
    if  Gaussian_window_samples == 1:
        
        raise ValueError("Single sample window width.")
    
    # Check consistency
    if Gaussian_window_samples <=5: # five samples are not a lot
    
    
        print('Smoothing with a narrow gaussian window...')
        
        
    smoothed_cumulative = gaussian_filter1d(Cumulative.astype(float), sigma=Gaussian_window_samples)
    
    
    return smoothed_cumulative
    
        

def calculate_mean_burst_duration(time_series_data, fs,scal_factor = 0.5, Visible=False):
    """
    Calculates the mean duration of bursts in a time series of network data and can plot the results.

    Args:
        time_series_data (list or np.array): The network data time series.
        fs = [Hz]
        baseline (float): The threshold value that defines a burst.
        plot (bool): If True, a plot of the data with burst start/end points is created.

    Returns:
        float: The mean duration of all detected bursts. Returns 0 if no bursts are found. the unit of time is s
    """
    burst_durations = []
    in_burst = False
    current_burst_duration = 0
    baseline = np.mean(time_series_data)*scal_factor
    time_step = 1/fs
    burst_start_indices = []
    burst_end_indices = []

    for i, data_point in enumerate(time_series_data):
        if data_point > baseline:
            if not in_burst:
                in_burst = True
                current_burst_duration = time_step
                burst_start_indices.append(i)
            else:
                current_burst_duration += time_step
        else:
            if in_burst:
                burst_durations.append(current_burst_duration)
                in_burst = False
                current_burst_duration = 0
                burst_end_indices.append(i - 1)

    if in_burst:
        burst_durations.append(current_burst_duration)
        burst_end_indices.append(len(time_series_data) - 1)

    if Visible:
        time_points = [i /fs for i in range(len(time_series_data))]
        plt.figure(figsize=(12, 6))
        plt.plot(time_points, time_series_data, label='Global activity')
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline})')

        start_time_points = [time_points[i] for i in burst_start_indices]
        start_values = [time_series_data[i] for i in burst_start_indices]
        end_time_points = [time_points[i] for i in burst_end_indices]
        end_values = [time_series_data[i] for i in burst_end_indices]

        plt.scatter(start_time_points, start_values, color='g', marker='o', s=100, label='Burst Start')
        plt.scatter(end_time_points, end_values, color='b', marker='x', s=100, label='Burst End')

        plt.title(f'Global activity. MBD: {np.mean(burst_durations)}')
        plt.xlabel(f'Time [s])')
        plt.ylabel(f'Global activity')
        plt.legend()
        plt.grid(True)

    
    if not burst_durations:
        return 0
    return np.mean(burst_durations)  

def Neuronal_traces(Char_folder=None,Char_base=None,Type ='Cumulative',t_rec = 600, fs = 10000, w_size = 0.02, overlap = 0.06, 
                    bin_size_s = 0.05, Isolate_NB = False, Gaussian_window = 0.04,
                     Visible = False,NB_statistics = False,Normalization_type = 'Peak amplitude'):
    
    
    # Raster_array = nx2, 1st column the channel's idx, 2nd column the timing of spike
    # Type = PCA or Cumulative. PCA = Usual neuronal dynamics, Cumulative= Cumulative IFR on all the electrodes.
    # t_rec = 600  # [s] Recording time
    # fs = 10000

    # Visible = True

    # # Calculate the GA
    # w_size = 0.12  # [s] 12
    # overlap = 0.06  # [s]

    # # Bin size for the IFR
    # bin_size_s = 0.05  # [s] 0.005 = 5 [ms]

    # # Whether to Isolate NB or keep the total signal
    # Isolate_NB = False

    # # Window size for smoothing
    # Gaussian_window = 2  # [s]

    # # Extract data
    # data = [None] * len(Strings)
    
    # Extract data
    os.chdir(Char_folder)

    base = Char_base
    
    Strings = ['012', '013', '021', '022', '023', '024',
               '031', '032', '033', '034', '042', '043']
    
    # Extract data
    data = [None] * len(Strings)
    
    T_max = t_rec * fs

    for i, s in enumerate(Strings):
        file_name = f'{base}{s}.mat'
        # Use loadmat to read .mat files
        mat_data = loadmat(file_name)
        x = mat_data['peak_train'].toarray()
        lendata = len(x)
        # find non-zero elements
        spk_timing = np.nonzero(x)[0]
        data[i] = spk_timing
    
    # Extract peak trains
    halfsize = lendata // 2
    
    if Visible:
        plt.figure()
        for i in range(len(Strings)):
            data_timings = data[i]
            data_plot = np.ones(len(data_timings)) * (i + 1)
            plt.scatter(data_timings / fs, data_plot, s=15, marker='.')
        plt.xlabel('Time [s]')
        plt.ylabel('Electrodes')
        plt.title('Spike Timings')
        plt.grid(True)
        plt.show()
    
    # Calculate NBs
    # You would need to define Rect_window in Python
    if Type == 'PCA':
        overlap = 0
        [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        
        # The new sampling frequency is downsampled by a factor determined by w_size [s]
        fs_downsampled = 1/w_size
        
        
        smoothed_cumulative =  get_Smoothed_Cumulative(Cumulative,fs_downsampled,Gaussian_window) 
        
        
        if Normalization_type == 'Standardization':
            print('Cumulative traces are STANDARDIZED')
            smoothed_cumulative = Standardization(smoothed_cumulative)
        
            if Visible == True:
                
                plt.figure()
                plt.plot(t_vec/fs,smoothed_cumulative,color = 'r')
                # plt.plot(t_vec,Cumulative,color = 'b')
                plt.xlabel('Time [s]')
                plt.ylabel('Standardized and Smoothed IFR')
                plt.show()
                
                
        elif Normalization_type == 'Peak amplitude':
            print('Cumulative traces are NORMALIZED')
            
            peak_amplitude = np.max(smoothed_cumulative)
            
            smoothed_cumulative = smoothed_cumulative/peak_amplitude
            if Visible == True:
                
                plt.figure()
                plt.plot(t_vec/fs,smoothed_cumulative,color = 'r')
                # plt.plot(t_vec,Cumulative,color = 'b')
                plt.xlabel('Time [s]')
                plt.ylabel('Normalized and Smoothed IFR')
                plt.show()
            
        
        
        return smoothed_cumulative,fs_downsampled
 
    
# --------------------------------- DATA AUGMENTATION ---------------------------------
'''

Data augmentation (DA) techniques for time series rests on traditional methdos or deep learning approaches.
Here I'll use traditional methdos which have as their basis the deformation, shortening, enlargment and magnitude/ temporal
warping (mosly used).

ASSUMPTIONS:
    The main assumption is that to destroy the global activity profile the 'size' of the changes mus be in the same time scale
    of the network busts. For Positives instead a magnitude less, which will just introduce subtle/noisy differences as can be 
    observed due to the biological variability inherent to the in-vitro models.


Be aware that using the wrong techniques to build positive or negative instances may lead to NEGATIVE TRAINING

Paper: Data Augmentation techniques in time series domain: A survey and  taxonomy

Data augmentation techniques can be used also to generate randomized surrogates which will be then used as negatives 
of the original data set in the contrastive learning framework.

Methods for positives:
    flip, shift, magnitude and/or temporal warping or a combination 
    
Methods for negatives:
    Heavy Jittering, Permutation, time slicing window,magnitude and/or temporal warping  or a combination 


Pathological traces can be used as negatives.    

Warping techniques consists in define a set of knots u, scale the data set value at their positions, scale in magnitude
or shift in time the knots and interpolate with a cubic spline.


NOTES:
    1) Magnitude warping is a poverful technique to introduce oscillatory activity within the single burst profile


WHen converting to tensor the output of positives and negatives functions the tensor's sizes are:
    0 : number of bathces
    1 : input data size (1)
    2 : length of the signal sequence
 
The input tensor size required from the 1DCNN cell is (N,C_in,L) 
where:
    N = Batch size
    C_in = Number of channels
    L = length of the signal sequence.


'''
def magnitude_warp(time_series,n_versions=10, intra_knot_dist=0.2, sigma_scale=0.2,fs= None):
    """
    Performs magnitude warping on a univariate time series.

    Args:
        time_series (np.ndarray): The input time series. 1 x sequence length
        n_knots (int): The number of knots to use for the warping curve.
        sigma (float): The standard deviation of the Gaussian distribution
                       to generate scaling values for the knots.
        intra_knot_dist = temporal distance between knots [s]
        sigma_scale = std of scaling factor by which the knot's magnitude is changed

    Returns:
        np.ndarray: The magnitude-warped time series.
    """
    time_series = time_series.numpy()
    # Create the time base for the series
    T = time_series.shape[1]
    x = np.linspace(0, T - 1, T)
    
    
    # Define the numebr of knots. 
    n_knots_ps = int((1)/intra_knot_dist) # Numebr of knots per second
    n_knots = int(n_knots_ps*T/fs)
    
    # Define the sigma
    sigma = sigma_scale

    # Define the knots for the scaling curve
    # The knots are placed at regular intervals across the time series.
    knots = np.linspace(0, T - 1, n_knots)

    Wraped_ts = []
    for it in range(n_versions):
        # Generate the scaling values for each knot using a Gaussian distribution.
        # The mean is 1 to maintain the original magnitude on average.
        scaling_values = np.random.normal(loc=1.0, scale=sigma, size=n_knots)
    
        # Create a cubic spline interpolation function for the scaling curve.
        # This ensures a smooth transition between scaling values.
        spline = CubicSpline(knots, scaling_values)
    
        # Get the scaling curve values for each point in the time series.
        scaling_curve = spline(x)
    
        # Apply the scaling to the original time series by element-wise multiplication.
        warped_series = time_series * scaling_curve
        
        Wraped_ts.append(warped_series)
    
    Wraped_ts = np.vstack(Wraped_ts)

    return torch.from_numpy(Wraped_ts)



def time_warp(time_series,n_versions=10, intra_knot_dist=0.2, sigma_scale=15,fs=None):
    """
    Performs time warping on a univariate time series.

    Args:
        time_series (np.ndarray): The input time series.
        n_knots (int): The number of knots to use for the warping curve.
        sigma (float): The standard deviation of the Gaussian distribution
                       to generate time offsets for the knots in seconds
                       
        intra_knot_dist = temporal distance between knots [s]
    Returns:
        np.ndarray: The time-warped time series.
    """
    
    time_series = time_series.numpy()
    
    # Create the time base for the series
    T = time_series.shape[1]
    x = np.linspace(0, T - 1, T)
    
    # Define the numebr of knots. 
    n_knots_ps = int((1)/intra_knot_dist) # Numebr of knots per second
    n_knots = int(n_knots_ps*T/fs)

    # Define the sigma
    sigma = sigma_scale*fs # TEMPORAL SHIFT

    # Define the knots for the warping curve
    knots = np.linspace(0, T - 1, n_knots)
    Wraped_ts = []
    for it in range(n_versions):
        # Generate random offsets for the knots using a Gaussian distribution.
        # These offsets define the "stretch" or "squeeze" at each knot.
        offsets = np.random.normal(loc=0.0, scale=sigma, size=n_knots)
    
        # The y-values for the spline are the knot locations plus the random offsets.
        spline_y = knots + offsets
    
        # Create a cubic spline interpolation function for the warping curve.
        spline = CubicSpline(knots, spline_y)
    
        # Get the warped time indices. These can be non-integer.
        warped_indices = spline(x)
    
        # Ensure warped indices are within the valid range [0, T-1].
        warped_indices[warped_indices < 0] = 0
        warped_indices[warped_indices > T - 1] = T - 1
    
        # Use linear interpolation to resample the original time series at the new
        # warped indices. This creates a new series with the stretched/compressed data.
        interp_func = interp1d(x, time_series)
        warped_series = interp_func(warped_indices)
        Wraped_ts.append(warped_series)
        
    Wraped_ts = np.vstack(Wraped_ts)
    return torch.from_numpy(Wraped_ts)


#
def Shift(time_series, n_versions, shift_magnitude_s, fs):
    """
    Generates n randomly shifted versions of a univariate time series.
    
    Args:
        time_series (torch.Tensor): The original time series with shape (samples,).
        n_versions (int): The number of shifted versions to generate.
        shift_magnitude_s (float): The maximum absolute value of the random shift (in seconds).
        fs (int): The sampling frequency in Hz.
        
    Returns:
        tuple: A tuple containing:
            - shifted_series (list): A list of torch.Tensors, where each tensor is a
                                     randomly shifted version of the original time series.
            - shifts (list): A list of the random integer shifts applied to each version.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 1:
        raise TypeError("Input 'time_series' must be a 1D PyTorch tensor.")
    
    max_shift_samples = int(shift_magnitude_s * fs)
    if n_versions <= 0 or max_shift_samples < 0:
        raise ValueError("'n_versions' must be > 0 and 'max_shift_samples' must be >= 0.")
    
    shifted_series = []
    shifts = []

    for _ in range(n_versions):
        shift_value = torch.randint(low=-max_shift_samples, high=max_shift_samples + 1, size=(1,)).item()
        shifted_version = torch.roll(time_series, shifts=shift_value, dims=0)
        
        shifted_series.append(shifted_version)
        shifts.append(shift_value)

    return shifted_series
def log_uniform(low, high, size=1):
    """
    Generates n random samples from a log-uniform distribution.

    Args:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        size (int): The number of samples to generate.

    Returns:
        np.ndarray: An array of log-uniformly distributed random numbers.
    """
    # Step 1: Log-transform the boundaries
    log_low = np.log(low)
    log_high = np.log(high)

    # Step 2: Sample uniformly in the log-space
    uniform_samples = np.random.uniform(log_low, log_high, size=size)

    # Step 3: Inverse transform to get the final samples
    log_uniform_samples = np.exp(uniform_samples)

    return log_uniform_samples


def Data_Augmentation(data,n_versions_insatances,n_param_vector,intra_knot_dist_range= [0.35,0.1],sigma_scale_range= [0.4,0.01],
                      MSE_threshold= 0.01,fs = None, shift_magnitude_s=30,choose_subset=False,Visible = False):
    '''
    
    This data augmentation algorithm first generates a certain number of surrogates and then splits negatives and 
    positives instances w.r.t. the mean squared error.
    Afterworks each insance in the Positives and Negatives will be shifted temporally.
    shift_magnitude_s (float): Max shift magnitude in seconds for the 'Shift' method.
    sigma_scale_range = Temporal shift in s and magnitude scaling for time and magnitude warping resp.
    
    data size = (1,window size)
    '''

  
    # First phase is to generate time and magnitude wraped samples.
    # Sample set of parameters:
        
    intra_knot_dist = np.random.uniform(intra_knot_dist_range[0], intra_knot_dist_range[1],n_param_vector)
    sigma_scale = log_uniform(sigma_scale_range[0],sigma_scale_range[1],n_param_vector)
    
    out_temp = []
    
   
    for it in range(n_param_vector):
        
        out_temp.append(magnitude_warp(data,n_versions=n_versions_insatances, intra_knot_dist=intra_knot_dist[it], sigma_scale=sigma_scale[it],fs= fs))
        out_temp.append(time_warp(data,n_versions=n_versions_insatances, intra_knot_dist=intra_knot_dist[it], sigma_scale=sigma_scale[it],fs= fs))
    
    
    
    out_temp = torch.vstack(out_temp)
    
    
    
    # -------- MSE calculation --------
    # Calculate element-wise squared error
    mse_per_row_ = []    
    for m in range(out_temp.size()[0]):
        
        mse_per_row_.append(F.mse_loss(out_temp[m,:].reshape(1,-1), data))
    
    # Compute the mean for each row
    mse_per_row = torch.stack(mse_per_row_)
    
    
    # mse_per_row = mse_per_row.unsqueeze(1) # unsqueeze to make it 50x1

    # --- Step 1: Find the indices of values less than 0.1 ---
    # This creates a boolean tensor (mask) where `True` means the condition is met
    threshold_mask = mse_per_row < MSE_threshold
    
    # Use torch.nonzero() to get the explicit indices if you need them
    indices_low_mse = torch.nonzero(threshold_mask)
    
    
    # --- Step 2: Divide the out_temp tensor into two groups ---
    # Group 1: Rows with MSE < 0.1
    Positives = out_temp[threshold_mask.squeeze()]
    
    # Group 2: Rows with MSE >= 0.1
    # The '~' operator inverts the boolean mask
    Negatives = out_temp[~threshold_mask.squeeze()]
    
    
    
    if Visible == True:
        
        
        t_vec = np.linspace(0,len(Positives[0,:]),len(Positives[0,:]))/fs
        
        
        
        plt.figure()
        plt.plot(t_vec,np.squeeze(data.numpy()),label='Original Series',linewidth=3)
        for k in range(len(Positives[:,0])):
            plt.plot(t_vec,Positives[k,:].numpy(), '--',alpha = 0.3 )
            
            
        plt.xlabel('Time [s]')
        plt.ylabel('Cumulative IFR')
        plt.title(f'Positive instances')
        plt.legend()
        plt.grid(True)
        plt.show()
            
            
        plt.figure()
        plt.plot(t_vec,np.squeeze(data.numpy()),label='Original Series',linewidth=3)
        for k in range(len(Negatives[:,0])):
            plt.plot(t_vec,Negatives[k,:].numpy(), '--',alpha = 0.3 )
            
            
        plt.xlabel('Time [s]')
        plt.ylabel('Cumulative IFR')
        plt.title(f'Negative instances')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
        
        
        
    Positives_final = []
    Negatives_final = []
    
    for k in range(len(Positives[:,0])):
        Positives_final.append(torch.vstack(Shift(Positives[k,:], n_versions_insatances, shift_magnitude_s, fs)))
        
        
    Positives_final = torch.vstack(Positives_final)
            
    # Generate the labels
    Pos_Lables = torch.zeros(Positives_final.shape[0], dtype=torch.int8)
    
    
    lbl = 1
    Neg_Label = []
    for k in range(len(Negatives[:,0])):
        Negatives_final.append(torch.vstack(Shift(Negatives[k,:], n_versions_insatances, shift_magnitude_s, fs)))
        Neg_Label.append(torch.tensor([lbl]).repeat(n_versions_insatances))
        
        lbl=lbl+1
        
        
    Negatives_final = torch.vstack(Negatives_final)
    Neg_Labels = torch.squeeze(torch.vstack(Neg_Label)).flatten()
    
    if choose_subset:
        
        
        # Handle when choose_subset is grater than len(positive/negatives_final)
        if choose_subset > len(Positives_final):
            choose_subset_ = len(Positives_final)
            
        else: choose_subset_ = choose_subset
        idx_pos =  random_sample_without_replacement(Positives_final, num_samples= choose_subset_)
        
        
        if choose_subset > len(Negatives_final):
            choose_subset_ = len(Negatives_final)
            
        else: choose_subset_ = choose_subset
        idx_neg=  random_sample_without_replacement(Negatives_final, num_samples= choose_subset_)
        
        
        
    
        return Positives_final[idx_pos,:],Negatives_final[idx_neg,:],Pos_Lables[idx_pos],Neg_Labels[idx_neg]
    
    
    else:
        return Positives_final,Negatives_final,Pos_Lables,Neg_Labels

# --------------------------------- 1D CNN ---------------------------------
class ResNet_Block(nn.Module):
    
    '''
    We also have the option to halve the output height and width while increasing the number of output channels. In this case we use 1x1
    convolutions via use_1x1conv=True. This comes in handy at the beginning of each ResNet block to reduce the spatial dimensionality via strides=2 or more.
    
    torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    '''
    
    
    def __init__(self,input_size,in_channels_res, out_channels_res, kernel_size_res, use_1x1conv=False, stride_res=1,use_dropout=False,dropout_pers=0.2):
        super().__init__()
        
        padding_amount =  calculate_padding(input_size, kernel_size_res, stride_res)
        self.c1 = nn.Conv1d(in_channels_res, out_channels_res, kernel_size_res, stride=stride_res,padding=padding_amount)
        
        padding_amount =  calculate_padding(input_size//stride_res, kernel_size_res, 1)
        self.c2 = nn.Conv1d(out_channels_res, out_channels_res, kernel_size_res, stride=1,padding =padding_amount )
        if use_1x1conv:
            # In this case strides MUST be grater than one
            self.c3 = nn.Conv1d(in_channels_res,out_channels_res, kernel_size=1,
                                        stride=stride_res)
        else:
            self.c3 = None
                
        self.bn1 = nn.BatchNorm1d(out_channels_res)
        self.bn2 = nn.BatchNorm1d(out_channels_res)     
        self.relu = nn.ReLU()
                  
        if use_dropout:
            
            self.dropout = nn.Dropout(p = dropout_pers)
            
        else:
            self.dropout = None
        
        
        
    def forward(self,X):
        
        Y =  self.relu(self.bn1(self.c1(X)))
        Y = self.bn2(self.c2(Y))
        if self.c3:
            X = self.c3(X)
        Y += X
        if self.dropout:
            Y = self.dropout(Y)

        return  self.relu(Y)
    

class ResNeXt_Block(nn.Module):
    
    '''
    ResNeXtBlock class takes as argument groups (g), with bot_channels intermediate (bottleneck) channels WHICH IS SET TO 1 (read above). 
    Lastly, when we need to reduce the height and width of the representation, we set use_1x1conv=True, strides=2.
    Remember that in and out channels MUST be divisible by the number fo groups.
    
    torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    '''
    
    def __init__(self,input_size, in_channels_res, out_channels_res, kernel_size_res, groups, use_1x1conv=False, stride_res=1,use_dropout=False,dropout_pers=0.2):
        super().__init__()
                
        
        self.c1 = nn.Conv1d(in_channels_res, out_channels_res, 1, stride=1)
        padding_amount =  calculate_padding(input_size, kernel_size_res, stride_res)
        self.c2 = nn.Conv1d(out_channels_res, out_channels_res, kernel_size_res, stride=stride_res,padding=padding_amount,
                                    groups=out_channels_res//groups)
        
        
        self.c3 = nn.Conv1d(out_channels_res, out_channels_res, 1, stride=1)
        
        self.bn1 = nn.BatchNorm1d(out_channels_res)
        self.bn2 = nn.BatchNorm1d(out_channels_res)
        self.bn3 = nn.BatchNorm1d(out_channels_res)
        if use_1x1conv:
            self.c4 = nn.Conv1d(in_channels_res,out_channels_res, kernel_size=1,
                                        stride=stride_res)
            self.bn4 = nn.BatchNorm1d(out_channels_res)
        else:
            self.c4 = None
    
    
        self.relu = nn.ReLU()
        
        if use_dropout:
            
            self.dropout = nn.Dropout(p = dropout_pers)
            
        else:
            self.dropout = None
            
        
        
        
    def forward(self,X):
        
        Y =  self.relu(self.bn1(self.c1(X)))
        Y =  self.relu(self.bn2(self.c2(Y)))
        Y = self.bn3(self.c3(Y))
        if self.c4:
            X = self.bn4(self.c4(X))
        Y += X
        if self.dropout:
            Y = self.dropout(Y)
            
        return  self.relu(Y)
        
        
class Head_Block(nn.Module):

    def __init__(self,in_channels_head,out_channels_head,use_dropout,dropout_pers):
        super().__init__()
        
        self.linear = nn.Linear(in_channels_head, out_channels_head)
        
        if use_dropout:
            
            self.dropout = nn.Dropout(p = dropout_pers)
            
        else:
            self.dropout = None
            
        self.relu = nn.ReLU()
        
        
        
    def forward(self,X):
        
        # 1. Pass the input through the linear layer
        Y = self.linear(X)
        
        # 2. Apply dropout if it exists
        if self.dropout:
            Y = self.dropout(Y)
        
        # 3. Apply the ReLU activation function
        Y = self.relu(Y)
            
            
        return Y
            
        
class Stem_Block(nn.Module):
    
    def __init__(self,input_size, out_channels_stem, kernel_size_stem, stride_stem):
        super().__init__()
        
        padding_amount =  calculate_padding(input_size, kernel_size_stem, stride_stem)
        
        self.c1 = nn.Conv1d(1, out_channels_stem, kernel_size_stem, stride=stride_stem,padding=padding_amount)
        self.bn1 = nn.BatchNorm1d(out_channels_stem)
        self.relu = nn.ReLU()  # Corrected spelling here
        
    def forward(self, X):
        Y = self.bn1(self.c1(X))
        Y = self.relu(Y)
        
        return Y


        



class OneD_CNN(nn.Module):
    
    '''
1D concolutional deep neuronal network. It can be used as an alternative to the GRU network or in combination.
It could be applied on single intances of isolated NB or on the entire sequence. The stride and kernel size must be 
tuned in base of the number of samples and the sampling frequency.
Params:
    
    in_channels (int) – Number of channels in the input image

    out_channels (int) – Number of channels produced by the convolution

    kernel_size (int or tuple) – Size of the convolving kernel

    stride (int or tuple, optional) – Stride of the convolution. Default: 1

    padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
    
    '''
    
    def __init__(self,input_fs=None, input_size = None,last_dropout=False, head_dropout=True, downsampling_rate=2, groups=16,
              dropout_pers=0.2, Block_Type='ResNet_Block', width_shrink=5,
              Network_depth=int(2**3), Stage_kernel=3, embedding_size=16,
              Stem_augmentation=16, Stem_kernel=5, Stem_stride=4,Verbose=False,weight_multiplier = 2,device = None):
        
        
        super().__init__()  
        
        self.Verbose = Verbose
        
        self.input_size = input_size
        
        
        # Input's sampling frequency in [s]
        self.input_fs = input_fs
        
        # Device for GPU acceleration
        self.device = device
    
        # ----- STAGES -----
        # Whether to use or not a dropout layer between stages
        self.last_dropout = last_dropout
        
        # Downsampling factor between successive stages
        self.downsampling_rate = downsampling_rate
        
        # Number of groups in ResNeXt block. Remember that it must be so that both 
        # in and out channels per stage are divisible by it
        self.groups = groups
        
        # Persentage of dropout nodes
        self.dropout_pers = dropout_pers
        
        # Type of block's architecutre: 'ResNet_Block' or 'ResNeXt_Block'
        self.Block_Type = Block_Type
        
        # Layer scaling factor in the head section
        self.width_shrink = width_shrink
        
        # Maximum depth of the network
        self.Network_depth = Network_depth
        
        # Window size of the bloks in the stages
        self.Stage_kernel = Stage_kernel
        
        # Weight multiplier (wm) for the quantized linear function
        self.w_m = weight_multiplier
        
        
        # ----- HEAD -----
        # Output embedding vector size
        self.embedding_size = embedding_size
        
        # Whether to use or not dropout layers in the head
        self.head_dropout = head_dropout
        
        
        # ----- STEM -----
        # Define the size of the first set of filters at the Stem section
        self.Stem_augmentation = Stem_augmentation
        
        # Define the kernel/window size of the stem
        self.Stem_kernel = Stem_kernel
        
        # Set the stride to take at the stem
        self.Stem_stride = Stem_stride
  
        
        
        # -------------------- BUILDING THE ARCHITECTURE --------------------
        
        # Initialize an empty sequential container to build the network
        self.net = nn.Sequential()

        self.build_stem()
        
        self.final_width = self.build_stages()
        
        self.build_head()
        
        
        
    
    def forward(self, X,fs=None,State='Embedding',n_repl=10,n_repl_params=20,choose_subset=False) :
        '''
        
        The input tensor size required from the 1DCNN cell is (N,C_in,L) 
        where:
            N = Batch size
            C_in = Number of channels (1 in our case)
            L = length of the signal sequence.
        
        '''
        
        # n_repl = number of replica for the augmented data, given the generation type we have
                # a TOTAL number of replica equal to
        # n_repl_params= Number of iteration to produce augmentation params
            
        # State = Training, Validation, Embedding
       
        
        if State == 'Training':
            
           
            # ------- Augment the data -------
            
            Positives,Negatives,Pos_Labels,Neg_Labels= Data_Augmentation(X,n_repl,n_repl_params,fs=self.input_fs,choose_subset = choose_subset )
            
            
            
            # --- Positives ---
            Pos_out_ =  Positives.to(torch.float32)
            # Add a dimension at index 1 for pytorch compliance
            Pos_out = torch.unsqueeze(Pos_out_, dim=1)
            
            
            # Load the tensors on the device
            Pos_out = Pos_out.to(self.device)
            
          
            
            # Train
            final_embedding_pos = self.net(Pos_out)
            
            
            
            
            # --- Negatives ---
            # We do NOT need to get the same number of istances as the positives.
            
            Neg_out_ =  Negatives.to(torch.float32)
            Neg_out = torch.unsqueeze(Neg_out_, dim=1)
            Neg_out =  Neg_out.to(self.device)
            
           
            # Train
            final_embedding_neg = self.net(Neg_out)
            
            
            
            # --- CONCATENATION STEP ---
            # Concatenate the final embeddings along the batch dimension (dim=0)
            final_embeddings = torch.cat((final_embedding_pos, final_embedding_neg), dim=0)
        
            # Concatenate the labels in the same order
            final_labels = torch.cat((Pos_Labels, Neg_Labels), dim=0)
            
            
            
            return final_embeddings,final_labels
        
        
        
        elif State == 'Validation':
        
            # In validation settings the Positive instances are not much variated from the reference 
            
           
            # ------- Augment the data -------
            
            # --- Positives ---
            Positives,Negatives,Pos_Labels,Neg_Labels= Data_Augmentation(X,n_repl,n_repl_params,fs=self.input_fs)
            

            
            # Train
            
            # --- Positives ---
            Pos_out_ =  Positives.to(torch.float32)
            # Add a dimension at index 1 for pytorch compliance
            Pos_out = torch.unsqueeze(Pos_out_, dim=1)
            # Load the tensors on the device
            Pos_out = Pos_out.to(self.device)

            # Train
            final_embedding_pos = self.net(Pos_out)
            
            
            
           
            
            
            
            
            
            return final_embedding_pos,Pos_Labels        
            
            
            
            
        elif State == 'Embedding':
            
            X_ = torch.unsqueeze(X, dim=1)
            # Load the tensors on the device
            X_  = X_ .to(self.device)
            
            # Embedd
            Embedding= self.net(X_)
            
            
            
            return Embedding
        
    
    
    def stage(self, depth, in_channels_stage ,out_channels_stage):
        
        
        blk = []
        
        if self.Block_Type == 'ResNet_Block':
            for i in range(depth):
                if i == 0:
                    # First does downsampling and channel enlargement.
                    blk.append(ResNet_Block(
                        self.input_size,
                        in_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        use_1x1conv=True, 
                        stride_res=self.downsampling_rate,
                        use_dropout=False,
                        dropout_pers=None
                    ))
                    # Update the input size
                    self.input_size = self.input_size//self.downsampling_rate
                    

                elif self.last_dropout == True and i == depth-1:
                    # Last has added a dropout layer for regularization
                    
                    blk.append(ResNet_Block(
                        self.input_size,
                        out_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        use_1x1conv=False, 
                        stride_res=1,
                        use_dropout=True,
                        dropout_pers=self.dropout_pers
                    ))
                    
                else:
                    
                    blk.append(ResNet_Block(
                        self.input_size,
                        out_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        use_1x1conv=False, 
                        stride_res=1,
                        use_dropout=False,
                        dropout_pers=None
                    ))
                    
                    
        elif self.Block_Type == 'ResNeXt_Block':
            
            for i in range(depth):
                
                if i == 0:
                    # First does downsampling and channel enlargement.
                    
                    blk.append(ResNeXt_Block(
                        self.input_size,
                        in_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        self.groups, 
                        use_1x1conv=True, 
                        stride_res=self.downsampling_rate,
                        use_dropout=False,
                        dropout_pers=None
                    ))
                    
                    #Update the input size
                    self.input_size = self.input_size//self.downsampling_rate
                    

                elif self.last_dropout == True and i == depth-1:
                    # Last has added a dropout layer for regularization
                    blk.append(ResNeXt_Block(
                        self.input_size,
                        out_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        self.groups, 
                        use_1x1conv=False, 
                        stride_res=1,
                        use_dropout=True,
                        dropout_pers=self.dropout_pers
                    ))
                    
                else:
                    
                    blk.append(ResNeXt_Block(
                        self.input_size,
                        out_channels_stage, 
                        out_channels_stage, 
                        self.Stage_kernel, 
                        self.groups, 
                        use_1x1conv=False, 
                        stride_res=1,
                        use_dropout=False,
                        dropout_pers=None
                    ))
                
        
        return nn.Sequential(*blk)
    


    
    def head(self):
        
        
        
        head_layers = nn.Sequential()
        
        # Add the adaptive average pooling layer
        head_layers.add_module('avg_pool', nn.AdaptiveAvgPool1d(1))
        
        # This custom lambda layer will flatten the tensor after pooling
        head_layers.add_module('flatten', nn.Flatten())
        
        # Calculate the number of linear layers
        num_layers = int(math.log(self.final_width / self.embedding_size) / math.log(self.width_shrink))
       
        
        current_in_size = self.final_width
        
        for i in range(num_layers):
            current_out_size = current_in_size // self.width_shrink
            head_layers.add_module(f'head_block_{i+1}', Head_Block(current_in_size, current_out_size, self.head_dropout, self.dropout_pers))
            current_in_size = current_out_size
        
        # Add a final linear layer if the last output size doesn't match the embedding size
        if current_in_size != self.embedding_size:
            head_layers.add_module('final_linear', nn.Linear(current_in_size, self.embedding_size))
        
        
        return head_layers
 
    
 
    def build_stages(self):
        # It inherents the net module
         
          Stage_width,Stage_depth = stage_features(depth_max=self.Network_depth,w_in=self.Stem_augmentation,w_m=self.w_m)
         
          
              
         
          for i in range(len(Stage_depth)):
              
              # TODO: I must define the input and output channels in the iterative way
              if i == 0:
                 
                  self.net.add_module(f'stage{i+1}',self.stage( Stage_depth[i], self.Stem_augmentation ,Stage_width[i]))
                 
              else:
                 
                  self.net.add_module(f'stage{i+1}',self.stage( Stage_depth[i], Stage_width[i-1] ,Stage_width[i]))
                 
              
          return  Stage_width[-1] # The last number of channels
             
    def build_stem(self):
        
        stem = Stem_Block(self.input_size,self.Stem_augmentation, self.Stem_kernel, self.Stem_stride)
        self.net.add_module('stem', stem)
        
        # Update input size
        self.input_size = self.input_size//self.Stem_stride
        
        
        
        
        
    
    def build_head(self):
        # It inherents thefinal width
        
        
        self.net.add_module(r'head',self.head())
    

# --------------------  WITH DATA FLOW  --------------------

# class ResNet_Block(nn.Module):
    
#     '''
#     We also have the option to halve the output height and width while increasing the number of output channels. In this case we use 1x1
#     convolutions via use_1x1conv=True. This comes in handy at the beginning of each ResNet block to reduce the spatial dimensionality via strides=2 or more.
    
#     torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
#     '''
    
    
#     def __init__(self,input_size,in_channels_res, out_channels_res, kernel_size_res, use_1x1conv=False, stride_res=1,use_dropout=False,dropout_pers=0.2):
#         super().__init__()
        
#         padding_amount =  calculate_padding(input_size, kernel_size_res, stride_res)
#         self.c1 = nn.Conv1d(in_channels_res, out_channels_res, kernel_size_res, stride=stride_res,padding=padding_amount)
        
#         padding_amount =  calculate_padding(input_size//stride_res, kernel_size_res, 1)
#         self.c2 = nn.Conv1d(out_channels_res, out_channels_res, kernel_size_res, stride=1,padding =padding_amount )
#         if use_1x1conv:
#             # In this case strides MUST be grater than one
#             self.c3 = nn.Conv1d(in_channels_res,out_channels_res, kernel_size=1,
#                                     stride=stride_res)
#         else:
#             self.c3 = None
                 
#         self.bn1 = nn.BatchNorm1d(out_channels_res)
#         self.bn2 = nn.BatchNorm1d(out_channels_res)      
#         self.relu = nn.ReLU()
            
#         if use_dropout:
            
#             self.dropout = nn.Dropout(p = dropout_pers)
            
#         else:
#             self.dropout = None
        
        
#     def forward(self,X):
        
#         print(f"ResNet_Block input shape: {X.shape}")
        
#         Y =  self.relu(self.bn1(self.c1(X)))
#         print(f"ResNet_Block after c1, bn1, relu shape: {Y.shape}")
        
#         Y = self.bn2(self.c2(Y))
#         print(f"ResNet_Block after c2, bn2 shape: {Y.shape}")
        
#         if self.c3:
#             X = self.c3(X)
#             print(f"ResNet_Block after c3 (1x1 conv) shape: {X.shape}")
            
#         Y += X
#         print(f"ResNet_Block after addition (residual connection) shape: {Y.shape}")
        
#         if self.dropout:
#             Y = self.dropout(Y)
#             print(f"ResNet_Block after dropout shape: {Y.shape}")

#         return  self.relu(Y)
    

# class ResNeXt_Block(nn.Module):
    
#     '''
#     ResNeXtBlock class takes as argument groups (g), with bot_channels intermediate (bottleneck) channels WHICH IS SET TO 1 (read above). 
#     Lastly, when we need to reduce the height and width of the representation, we set use_1x1conv=True, strides=2.
#     Remember that in and out channels MUST be divisible by the number fo groups.
    
#     torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
#     '''
    
#     def __init__(self,input_size, in_channels_res, out_channels_res, kernel_size_res, groups, use_1x1conv=False, stride_res=1,use_dropout=False,dropout_pers=0.2):
#         super().__init__()
                
        
#         self.c1 = nn.Conv1d(in_channels_res, out_channels_res, 1, stride=1)
#         padding_amount =  calculate_padding(input_size, kernel_size_res, stride_res)
#         self.c2 = nn.Conv1d(out_channels_res, out_channels_res, kernel_size_res, stride=stride_res,padding=padding_amount,
#                                  groups=out_channels_res//groups)
        
        
#         self.c3 = nn.Conv1d(out_channels_res, out_channels_res, 1, stride=1)
        
#         self.bn1 = nn.BatchNorm1d(out_channels_res)
#         self.bn2 = nn.BatchNorm1d(out_channels_res)
#         self.bn3 = nn.BatchNorm1d(out_channels_res)
#         if use_1x1conv:
#             self.c4 = nn.Conv1d(in_channels_res,out_channels_res, kernel_size=1,
#                                      stride=stride_res)
#             self.bn4 = nn.BatchNorm1d(out_channels_res)
#         else:
#             self.c4 = None
    
        
#         self.relu = nn.ReLU()
        
#         if use_dropout:
            
#             self.dropout = nn.Dropout(p = dropout_pers)
            
#         else:
#             self.dropout = None
            
        
        
#     def forward(self,X):
        
#         print(f"ResNeXt_Block input shape: {X.shape}")
        
#         Y =  self.relu(self.bn1(self.c1(X)))
#         print(f"ResNeXt_Block after c1, bn1, relu shape: {Y.shape}")
        
#         Y =  self.relu(self.bn2(self.c2(Y)))
#         print(f"ResNeXt_Block after c2, bn2, relu shape: {Y.shape}")
        
#         Y = self.bn3(self.c3(Y))
#         print(f"ResNeXt_Block after c3, bn3 shape: {Y.shape}")
        
#         if self.c4:
#             X = self.bn4(self.c4(X))
#             print(f"ResNeXt_Block after c4, bn4 (1x1 conv) shape: {X.shape}")
            
#         Y += X
#         print(f"ResNeXt_Block after addition (residual connection) shape: {Y.shape}")
        
#         if self.dropout:
#             Y = self.dropout(Y)
#             print(f"ResNeXt_Block after dropout shape: {Y.shape}")
            
#         return  self.relu(Y)
        
        
# class Head_Block(nn.Module):

#     def __init__(self,in_channels_head,out_channels_head,use_dropout,dropout_pers):
#         super().__init__()
        
#         self.linear = nn.Linear(in_channels_head, out_channels_head)
        
#         if use_dropout:
            
#             self.dropout = nn.Dropout(p = dropout_pers)
            
#         else:
#             self.dropout = None
            
#         self.relu = nn.ReLU()
        
        
#     def forward(self,X):
#         print(f"Head_Block input shape: {X.shape}")
        
#         # 1. Pass the input through the linear layer
#         Y = self.linear(X)
#         print(f"Head_Block after linear layer shape: {Y.shape}")
        
#         # 2. Apply dropout if it exists
#         if self.dropout:
#             Y = self.dropout(Y)
#             print(f"Head_Block after dropout shape: {Y.shape}")
        
#         # 3. Apply the ReLU activation function
#         Y = self.relu(Y)
            
            
#         return Y
            
        
# class Stem_Block(nn.Module):
    
#     def __init__(self,input_size, out_channels_stem, kernel_size_stem, stride_stem):
#         super().__init__()
        
#         padding_amount =  calculate_padding(input_size, kernel_size_stem, stride_stem)
        
#         self.c1 = nn.Conv1d(1, out_channels_stem, kernel_size_stem, stride=stride_stem,padding=padding_amount)
#         self.bn1 = nn.BatchNorm1d(out_channels_stem)
#         self.relu = nn.ReLU()  # Corrected spelling here
        
#     def forward(self, X):
#         print(f"Stem_Block input shape: {X.shape}")
        
#         Y = self.bn1(self.c1(X))
#         print(f"Stem_Block after conv1d and batchnorm shape: {Y.shape}")
        
#         Y = self.relu(Y)
#         print(f"Stem_Block after relu shape: {Y.shape}")
        
#         return Y


    

# class OneD_CNN(nn.Module):
    
#     '''
# 1D concolutional deep neuronal network. It can be used as an alternative to the GRU network or in combination.
# It could be applied on single intances of isolated NB or on the entire sequence. The stride and kernel size must be 
# tuned in base of the number of samples and the sampling frequency.
# Params:
    
#     in_channels (int) – Number of channels in the input image

#     out_channels (int) – Number of channels produced by the convolution

#     kernel_size (int or tuple) – Size of the convolving kernel

#     stride (int or tuple, optional) – Stride of the convolution. Default: 1

#     padding (int, tuple or str, optional) – Padding added to both sides of the input. Default: 0
    
#     '''
    
#     def __init__(self,input_fs=None, input_size = None,last_dropout=True, head_dropout=True, downsampling_rate=2, groups=8,
#               dropout_pers=0.2, Block_Type='ResNet_Block', width_shrink=4,
#               Network_depth=64, Stage_kernel=3, embedding_size=16,
#               Stem_augmentation=32, Stem_kernel=None, Stem_stride=None,Verbose=True,weight_multiplier=2,device = None):
        
        
#         super().__init__()  
        
#         self.Verbose = Verbose
        
#         self.input_size = input_size
#         print('-------------------------------------')
#         print('')
#         print(f'Current input size: {self.input_size}')
#         print('')
#         print('-------------------------------------')
        
#         # Input's sampling frequency in [s]
#         self.input_fs = input_fs
        
#         # Device for GPU acceleration
#         self.device = device
    
#         # ----- STAGES -----
#         # Whether to use or not a dropout layer between stages
#         self.last_dropout = last_dropout
        
#         # Downsampling factor between successive stages
#         self.downsampling_rate = downsampling_rate
        
#         # Number of groups in ResNeXt block. Remember that it must be so that both 
#         # in and out channels per stage are divisible by it
#         self.groups = groups
        
#         # Persentage of dropout nodes
#         self.dropout_pers = dropout_pers
        
#         # Type of block's architecutre: 'ResNet_Block' or 'ResNeXt_Block'
#         self.Block_Type = Block_Type
        
#         # Layer scaling factor in the head section
#         self.width_shrink = width_shrink
        
#         # Maximum depth of the network
#         self.Network_depth = Network_depth
        
#         # Window size of the bloks in the stages
#         self.Stage_kernel = Stage_kernel

        # # Weight multiplier (wm) for the quantized linear function
        # self.w_m = weight_multiplier
        
        
        
#         # ----- HEAD -----
#         # Output embedding vector size
#         self.embedding_size = embedding_size
        
#         # Whether to use or not dropout layers in the head
#         self.head_dropout = head_dropout
        
        
#         # ----- STEM -----
#         # Define the size of the first set of filters at the Stem section
#         self.Stem_augmentation = Stem_augmentation
        
#         # Define the kernel/window size of the stem
#         self.Stem_kernel = Stem_kernel
        
#         # Set the stride to take at the stem
#         self.Stem_stride = Stem_stride
    
        
        
#         # -------------------- BUILDING THE ARCHITECTURE --------------------
        
#         # Initialize an empty sequential container to build the network
#         self.net = nn.Sequential()

#         self.build_stem()
        
#         self.final_width = self.build_stages()
        
#         self.build_head()
        
        
    
#     def forward(self, X,fs=None,State='Embedding',n_repl=10,n_repl_params=5) :
#         '''
        
#         The input tensor size required from the 1DCNN cell is (N,C_in,L) 
#         where:
#              N = Batch size
#              C_in = Number of channels
#              L = length of the signal sequence.
        
#         '''
        
#         # n_repl = number of replica for the augmented data, given the generation type we have
#                  # a TOTAL number of replica equal to
#         # n_repl_params= Number of iteration to produce augmentation params
              
#         # State = Training, Validation, Embedding
        
        
#         print(f"OneD_CNN forward pass started. Initial input shape: {X.shape}")

#         if State == 'Training':
            
            
#             # ------- Augment the data -------
            
#             Positives,Negatives,Pos_Labels,Neg_Labels= Data_Augmentation(X,n_repl,n_repl_params,fs=self.input_fs)
            
            
            
            
#             # --- Positives ---
#             Pos_out_ =  Positives.to(torch.float32)
#             # Add a dimension at index 1 for pytorch compliance
#             Pos_out = torch.unsqueeze(Pos_out_, dim=1)
            
            
#             # Load the tensors on the device
#             Pos_out = Pos_out.to(self.device)
            
            
#             # Train
#             final_embedding_pos = self.net(Pos_out)
            
            
            
            
#             # --- Negatives ---
#             # We do NOT need to get the same number of istances as the positives.
            
#             Neg_out_ =  Negatives.to(torch.float32)
#             Neg_out = torch.unsqueeze(Neg_out_, dim=1)
#             Neg_out =  Neg_out.to(self.device)
            
            
#             # Train
#             final_embedding_neg = self.net(Neg_out)
            
            
            
#             # --- CONCATENATION STEP ---
#             # Concatenate the final embeddings along the batch dimension (dim=0)
#             final_embeddings = torch.cat((final_embedding_pos, final_embedding_neg), dim=0)
            
#             # Concatenate the labels in the same order
#             final_labels = torch.cat((Pos_Labels, Neg_Labels), dim=0)
            
            
            
#             return final_embeddings,final_labels
        
        
        
#         elif State == 'Validation':
        
#             # In validation settings the Positive instances are not much variated from the reference 
            
            
#             # ------- Augment the data -------
            
#             # --- Positives ---
#             Positives,Negatives,Pos_Labels,Neg_Labels= Data_Augmentation(X,n_repl,n_repl_params,fs=self.input_fs)
            

            
#             # Train
            
#             # --- Positives ---
#             Pos_out_ =  Positives.to(torch.float32)
#             # Add a dimension at index 1 for pytorch compliance
#             Pos_out = torch.unsqueeze(Pos_out_, dim=1)
#             # Load the tensors on the device
#             Pos_out = Pos_out.to(self.device)

#             # Train
#             final_embedding_pos = self.net(Pos_out)
            
            
            
            
            
            
            
            
#             return final_embedding_pos,Pos_Labels     
            
            
            
#         elif State == 'Embedding':
            
#             print(f"Processing in Embedding state...")
#             X_ = torch.unsqueeze(X, dim=1)
#             print(f"Input after unsqueeze: {X_.shape}")
            
#             # Load the tensors on the device
#             X_  = X_ .to(self.device)
            
#             # Embedd
#             Embedding= self.net(X_)
            
#             print(f"Final embedding shape: {Embedding.shape}")
            
#             return Embedding
        
    
    
#     def stage(self, depth, in_channels_stage ,out_channels_stage):
        
#         print(f"Parameters: depth={depth}, in_channels_stage={in_channels_stage}, out_channels_stage={out_channels_stage}")
        
#         blk = []
        
#         if self.Block_Type == 'ResNet_Block':
#             print(f"Building stage with {self.Block_Type}...")
#             for i in range(depth):
#                 print(f"--- Block {i+1}/{depth} ---")
#                 if i == 0:
#                     # First does downsampling and channel enlargement.
#                     print("Condition: First block (downsampling and channel enlargement)")
#                     blk.append(ResNet_Block(
#                         self.input_size,
#                         in_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         use_1x1conv=True, 
#                         stride_res=self.downsampling_rate,
#                         use_dropout=False,
#                         dropout_pers=None
#                     ))
#                     print(f"Created ResNet_Block with in_channels={in_channels_stage}, out_channels={out_channels_stage}, stride_res={self.downsampling_rate}")
#                     # Update the input size
#                     self.input_size = self.input_size//self.downsampling_rate
#                     print('-------------------------------------')
#                     print('')
#                     print(f'Current input size: {self.input_size}')
#                     print('')
#                     print('-------------------------------------')

#                 elif self.last_dropout == True and i == depth-1:
#                     # Last has added a dropout layer for regularization
#                     print("Condition: Last block with dropout")
#                     blk.append(ResNet_Block(
#                         self.input_size,
#                         out_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         use_1x1conv=False, 
#                         stride_res=1,
#                         use_dropout=True,
#                         dropout_pers=self.dropout_pers
#                     ))
#                     print(f"Created ResNet_Block with dropout, in_channels={out_channels_stage}, out_channels={out_channels_stage}, dropout_pers={self.dropout_pers}")
                    
#                 else:
#                     print("Condition: Intermediate block")
#                     blk.append(ResNet_Block(
#                         self.input_size,
#                         out_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         use_1x1conv=False, 
#                         stride_res=1,
#                         use_dropout=False,
#                         dropout_pers=None
#                     ))
#                     print(f"Created ResNet_Block, in_channels={out_channels_stage}, out_channels={out_channels_stage}")
                    
#         elif self.Block_Type == 'ResNeXt_Block':
#             print(f"Building stage with {self.Block_Type}...")
#             for i in range(depth):
#                 print(f"--- Block {i+1}/{depth} ---")
#                 if i == 0:
#                     # First does downsampling and channel enlargement.
#                     print("Condition: First block (downsampling and channel enlargement)")
#                     blk.append(ResNeXt_Block(
#                         self.input_size,
#                         in_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         self.groups, 
#                         use_1x1conv=True, 
#                         stride_res=self.downsampling_rate,
#                         use_dropout=False,
#                         dropout_pers=None
#                     ))
#                     print(f"Created ResNeXt_Block with in_channels={in_channels_stage}, out_channels={out_channels_stage}, groups={self.groups}, stride_res={self.downsampling_rate}")
#                     #Update the input size
#                     self.input_size = self.input_size//self.downsampling_rate
#                     print('-------------------------------------')
#                     print('')
#                     print(f'Current input size: {self.input_size}')
#                     print('')
#                     print('-------------------------------------')

#                 elif self.last_dropout == True and i == depth-1:
#                     # Last has added a dropout layer for regularization
#                     print("Condition: Last block with dropout")
#                     blk.append(ResNeXt_Block(
#                         self.input_size,
#                         out_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         self.groups, 
#                         use_1x1conv=False, 
#                         stride_res=1,
#                         use_dropout=True,
#                         dropout_pers=self.dropout_pers
#                     ))
#                     print(f"Created ResNeXt_Block with dropout, in_channels={out_channels_stage}, out_channels={out_channels_stage}, groups={self.groups}, dropout_pers={self.dropout_pers}")
                    
#                 else:
#                     print("Condition: Intermediate block")
#                     blk.append(ResNeXt_Block(
#                         self.input_size,
#                         out_channels_stage, 
#                         out_channels_stage, 
#                         self.Stage_kernel, 
#                         self.groups, 
#                         use_1x1conv=False, 
#                         stride_res=1,
#                         use_dropout=False,
#                         dropout_pers=None
#                     ))
#                     print(f"Created ResNeXt_Block, in_channels={out_channels_stage}, out_channels={out_channels_stage}, groups={self.groups}")
                
        
#         return nn.Sequential(*blk)
    

    
#     def head(self):
        
#         print('--------------------------------------------------')
#         print('')
#         print(f'         BUILDING HEAD          ')
#         print(f"Initial values: final stage width={self.final_width}, embedding size={self.embedding_size}, width shrink={self.width_shrink}")
        
#         head_layers = nn.Sequential()
        
#         # Add the adaptive average pooling layer
#         head_layers.add_module('avg_pool', nn.AdaptiveAvgPool1d(1))
        
#         # This custom lambda layer will flatten the tensor after pooling
#         head_layers.add_module('flatten', nn.Flatten())
        
#         # Calculate the number of linear layers
#         num_layers = int(math.log(self.final_width / self.embedding_size) / math.log(self.width_shrink))
#         # print(f'Layer calculation: log({self.final_width}/{self.embedding_size}) / log({self.width_shrink})')
#         print(f"Calculated number of layers: {num_layers}")
        
#         current_in_size = self.final_width
        
#         for i in range(num_layers):
#             current_out_size = current_in_size // self.width_shrink
#             head_layers.add_module(f'head_block_{i+1}', Head_Block(current_in_size, current_out_size, self.head_dropout, self.dropout_pers))
#             current_in_size = current_out_size
        
#         # Add a final linear layer if the last output size doesn't match the embedding size
#         if current_in_size != self.embedding_size:
#             head_layers.add_module('final_linear', nn.Linear(current_in_size, self.embedding_size))
        
#         print("--- Head built successfully ---")
#         return head_layers
 
    
#     def build_stages(self):
#         # It inherents the net module
          
#         Stage_width,Stage_depth = stage_features(depth_max=self.Network_depth,w_in=self.Stem_augmentation,w_m=self.w_m)
          
#         if self.Verbose == True:
#             print('')
#             print('--------------- INFO STAGES --------------')
            
#             print('Stages depth: ',Stage_depth)
#             print('Stage width: ',Stage_width)
#             print('Block type: ',self.Block_Type)
#             print('')
        
#         for i in range(len(Stage_depth)):
#             print('--------------------------------------------------')
#             print('')
#             print(f'       BUILDING STAGE {i}       ')
            
#             # TODO: I must define the input and output channels in the iterative way
#             if i == 0:
                
#                 self.net.add_module(f'stage{i+1}',self.stage( Stage_depth[i], self.Stem_augmentation ,Stage_width[i]))
                
#             else:
                
#                 self.net.add_module(f'stage{i+1}',self.stage( Stage_depth[i], Stage_width[i-1] ,Stage_width[i]))
                
#             print('')
#             print('--------------------------------------------------')
          
#         print('Stages built succesfully')
#         return  Stage_width[-1] # The last number of channels
              
#     def build_stem(self):
        
#         stem = Stem_Block(self.input_size,self.Stem_augmentation, self.Stem_kernel, self.Stem_stride)
#         self.net.add_module('stem', stem)
        
#         # Update input size
#         self.input_size = self.input_size//self.Stem_stride
#         print('-------------------------------------')
#         print('')
#         print(f'Current input size: {self.input_size}')
#         print('')
#         print('-------------------------------------')
        
        
#     def build_head(self):
#         # It inherents thefinal width
        
        
#         self.net.add_module(r'head',self.head())


# ------------------------- TRAINING AND OPTIMIZATION -------------------------




def train_one_epoch(model,dataloader,loss_fn,optimizer_fn,fs,device,miner_hard,Margin_loss):
    '''
    The output loss value (broadcasted to the bayesian optimizer) is normalized by Margin_loss value
    given to the loss function.
    '''
    
    running_loss = 0.
    last_loss = 0.
    
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
        print('  batch {} norm loss: {}'.format(i + 1, avg_loss/Margin_loss))
       
    return avg_loss/Margin_loss     
     
     



def get_newspace(res_gp,pers):
    
    '''
    This function returns the narrowed search space for the Bayesian Optimization 
    algorithm given the 'pers' persentage of best points from the previous Optimization 
    Process.
  
    
    pers = between 0 and 1. Persentage of best points chosen.
    
    Parameters' name:
        
        Depth: d
        Width multiplier: wm
        Block type: blk
        Width shrink: ws
        Embedding size: es
    
    The cited HP are ordered in the same way in 'Params'
    
    
    '''
    # --- Best points ---
    n_bp = np.int16(len(res_gp.func_vals[:]) * pers)
    
    
    
    # --- Objective Function Evaluations ---
    # Draw out and sort them.
    
    # np.argsort() does not sort the array itself. Instead, it returns an array of integer indices that, 
    # if used to index the original array, would produce a sorted version of the array.
    Sorting_idx = np.argsort(res_gp.func_vals[:])
    
    # Sort function values
    Sorted_funcval = res_gp.func_vals[Sorting_idx]
    
    
    # --- Extract space values ---
    # Define storing array sizes
    n_params = np.int16(len(res_gp.x_iters[0]))
    
    
    Space_temp = np.zeros((n_bp,n_params))
    
    i = 0
    for sort_idx in Sorting_idx[0:n_bp]:
        
        
        Space_temp[i,:] = res_gp.x_iters[sort_idx]
        i = i+1
        
        
        
    
    # ------ Redefine the searching space ------
    # Find extremities
    lower_bounds = np.min(Space_temp,axis=0)
    upper_bounds = np.max(Space_temp,axis=0)
    
    
    # Define the new space
    
    # Set HPs' range
    d = Real(lower_bounds[0],upper_bounds[0],'log-uniform',name = 'Depth')
    wm = Real(lower_bounds[1],upper_bounds[1],'log-uniform',name = 'Width multiplier')
    blk  = Real(lower_bounds[2],upper_bounds[2],name = 'Block type')
    ws =  Real(lower_bounds[3],upper_bounds[3],name = 'Width shrink')
    es =  Real(lower_bounds[3],upper_bounds[3],name = 'Embedding size')
    




    space = [d,
             wm,
             blk,
             ws,
             es
        ]
    
    return space


def OneD_CNN_Arch_Wrapper(Params):
    
 
    
    '''
    Used for architecture optimization
    
    Parameters' name:
        
        Depth: d
        Width multiplier: wm
        Block type: blk
        Width shrink: ws
        Embedding size: es
    
    The cited HP are ordered in the same way in 'Params'
    '''
    
    # ---------- Parameters ----------
    d   = Params[0]
    wm  = Params[1]
    blk = Params[2]
    ws  = Params[3]
    es  = Params[4]
    
    # Retrieve parameters' values
    Block_array = ['ResNet_Block','ResNeXt_Block']
    depth = int(2**d)
    
    
    # Fixed values
    Margin_loss = 0.4
    Margin_miner = Margin_loss/2
    
    
    
    
    # ---------- Print Section ----------
    print('')
    print('')
    print("---------------- Hyperparameter Values ---------------")
    print(f"Depth (d): {d} -> Converted Depth: {depth}")
    print(f"Width Multiplier (wm): {wm}")
    print(f"Block Type (blk): {Block_array[blk]}")
    print(f"Width Shrink (ws): {ws}")
    print(f"Embedding Size (es): {es}")
    print('')
    print('')
    print('------------------------------------------------------')

        
    
    

    # --------------------- DATA PREPARATION ---------------------

    # ------------- Free cuda -------------
    torch.cuda.empty_cache()


    Dataset_training_window_size_s = 200


    print('')
    print('')
    print('-------------------- Loading data --------------------')
    # --------------------- PREPARE THE DATASET ---------------------
    Char_folder_array = [r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11',r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well17']
    Char_base_array = ['ptrain_Control00_Well11_','ptrain_Control00_Well17_']

    data_array = []
    for j in range(2):
        
        
        smoothed_cumulative,fs_downsampled = Neuronal_traces(Visible=False,Char_folder=Char_folder_array[j],Char_base=Char_base_array[j],w_size=0.02,Gaussian_window=0.04)
        

        
        data_array.append( torch.unsqueeze(torch.from_numpy(smoothed_cumulative),0).float() )

    data = torch.cat((data_array[0], data_array[1]), dim=1)


    #%
    print('')
    print('')
    
    #%
    # ---------------- TRAINING DATA LOADER ----------------

    Dataset_training = TimeSeriesDataset(data,fs=fs_downsampled,window_size_s=Dataset_training_window_size_s)
    # Create the DataLoader
    Dataloader_training = DataLoader(Dataset_training, 
                            batch_size=1, 
                            sampler=RandomSampler(Dataset_training,replacement=True, num_samples=120),
                            shuffle=False, 
                            drop_last=True) # drop_last is important to ensure all batches have the same size


    # Set the length of trianing data to correctly initialize the network
    window_size_temp = Dataset_training_window_size_s*fs_downsampled # in samples

    # Find the window size closest to a power of two
    Training_data_length = closest_power_of_2(window_size_temp)





    # --- SET DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
           
            
    print('')
    print('')
    print('-------------------- NETWORK --------------------')
    network_temp = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                       head_dropout = True, downsampling_rate=2, groups=16,dropout_pers=0.2, Block_Type = Block_array[blk],
                       width_shrink = ws, Network_depth = depth, Stage_kernel = 3, embedding_size = es, 
                       Stem_augmentation = 16, Stem_kernel = 5, Stem_stride = 4,weight_multiplier = wm, device = device)
    
    
    print(network_temp)
    del network_temp
    print('')
    print('')
    print('-----------------------------------------------------')


    # ... 10 epoch
    EPOCHS = 3
    
    temp_loss = 0
    
    for epoch in range(EPOCHS):
        print('')
        print('')
        print('------------------------------------------------------')
        print(f'----------- Starting instance: {epoch+1} ------------')
        
        
        # --------------------- INITIALIZE THE NETWORK ---------------------
        
        # Reset the parameters. Re-state the network:
        network = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                           head_dropout = True, downsampling_rate=2, groups=16,dropout_pers=0.2, Block_Type = Block_array[blk],
                           width_shrink = ws, Network_depth = depth, Stage_kernel = 3, embedding_size = es, 
                           Stem_augmentation = 16, Stem_kernel = 5, Stem_stride = 4,weight_multiplier = wm, device = device)

             

        # ------- Load the network on the device -------
        network = network.to(device)
                


        # --- Training algorithms ---

        reducer = reducers.AvgNonZeroReducer()
        loss_fn = losses.TripletMarginLoss(margin=Margin_loss,
                                           swap=True,
                                           distance=CosineSimilarity(), #  is preferred in embedding contexts.
                                           reducer = reducer,
                                           embedding_regularizer = None) # See in the obsidian page 'Deep learning Training Regularizers' the reason


        # --- Miner ---
        miner_hard = miners.TripletMarginMiner(margin=Margin_miner, type_of_triplets="hard",distance=CosineSimilarity()) 
                                        
        # --- Optimizer ---
        optimizer_fn = torch.optim.AdamW(network.parameters())
        
        
        network.train(True)
        # --------------------- TRAINING ---------------------
        training_loss = train_one_epoch(network,Dataloader_training,loss_fn,optimizer_fn,fs_downsampled,device,miner_hard,Margin_loss)
        
        
        temp_loss = temp_loss+training_loss
        print('')
        print('')
        print('------------------------------------------------------')

        
        
    average_loss = temp_loss/(EPOCHS)
    # The loss has already been normalized in the one epoch training loop
    average_loss_norm = average_loss
    
    return average_loss_norm





























def OneD_CNN_Train_Wrapper(Params):
    
 
    
    '''
    Used for training optimization
    
    Parameter names:
        Margin: mrg 
        Learing rate: lr 
        Beta 1: b1
        Beta 2: b2
        Weight decay: wd 
    
    The cited HP are ordered in the same way in 'Params'
    '''
    
    
    # ---------------- CHANGE OPTIMIZED PARAMETERS ----------------
    
    # Optimized parameters from previous phase
    d   = 3
    wm  = 2
    blk = int(0)
    ws  = 5
    es  = 16
    
    # Retrieve parameters' values
    Block_array = ['ResNet_Block','ResNeXt_Block']
    depth = int(2**d)
    
    # ---------- Print Section ----------
    print('')
    print('')
    print("---------------- Hyperparameter Architecture Values ---------------")
    print(f"Depth (d): {d} -> Converted Depth: {depth}")
    print(f"Width Multiplier (wm): {wm}")
    print(f"Block Type (blk): {Block_array[blk]}")
    print(f"Width Shrink (ws): {ws}")
    print(f"Embedding Size (es): {es}")
    print('')
    print('')
    print('------------------------------------------------------')

    
    
    
    
    
    
    
    # ---------- Parameters ----------
    mrg = Params[0]
    lr  = Params[1]
    b1  = Params[2]
    b2  = Params[3]
    wd  = Params[4]
    
   
    
    # Fixed values
    Margin_loss = mrg
    Margin_miner = Margin_loss/2
    
    
    print('')
    print('')
    print("---------------- Hyperparameter Optimizer Values ---------------")
    print(f"Margin (mrg): {mrg}")
    print(f"Learning Rate (lr): {lr}")
    print(f"Beta 1 (b1): {b1}")
    print(f"Beta 2 (b2): {b2}")
    print(f"Weight Decay (wd): {wd}")
    print('')
    print('')
    print('------------------------------------------------------')
        
    
        
    
    

    # --------------------- DATA PREPARATION ---------------------

    # ------------- Free cuda -------------
    torch.cuda.empty_cache()


    Dataset_training_window_size_s = 200


    print('')
    print('')
    print('-------------------- Loading data --------------------')
    # --------------------- PREPARE THE DATASET ---------------------
    Char_folder_array = [r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11',r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well17']
    Char_base_array = ['ptrain_Control00_Well11_','ptrain_Control00_Well17_']

    data_array = []
    for j in range(2):
        
        
        smoothed_cumulative,fs_downsampled= Neuronal_traces(Visible=False,Char_folder=Char_folder_array[j],Char_base=Char_base_array[j],w_size=0.02,Gaussian_window=0.04)
        

        
        data_array.append( torch.unsqueeze(torch.from_numpy(smoothed_cumulative),0).float() )

    data = torch.cat((data_array[0], data_array[1]), dim=1)


    #%
    print('')
    print('')
    
    #%
    # ---------------- TRAINING DATA LOADER ----------------

    Dataset_training = TimeSeriesDataset(data,fs=fs_downsampled,window_size_s=Dataset_training_window_size_s)
    # Create the DataLoader
    Dataloader_training = DataLoader(Dataset_training, 
                            batch_size=1, 
                            sampler=RandomSampler(Dataset_training,replacement=True, num_samples=120),
                            shuffle=False, 
                            drop_last=True) # drop_last is important to ensure all batches have the same size


    # Set the length of trianing data to correctly initialize the network
    window_size_temp = Dataset_training_window_size_s*fs_downsampled # in samples

    # Find the window size closest to a power of two
    Training_data_length = closest_power_of_2(window_size_temp)





    # --- SET DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
           
            
    print('')
    print('')
    print('-------------------- NETWORK --------------------')
    network_temp = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                       head_dropout = True, downsampling_rate=2, groups=16,dropout_pers=0.2, Block_Type = Block_array[blk],
                       width_shrink = ws, Network_depth = depth, Stage_kernel = 3, embedding_size = es, 
                       Stem_augmentation = 16, Stem_kernel = 5, Stem_stride = 4,weight_multiplier = wm, device = device)
    
    
    print(network_temp)
    del network_temp
    print('')
    print('')
    print('-----------------------------------------------------')


    # ... 5 epoch
    EPOCHS = 4
    
    temp_loss = 0
    
    for epoch in range(EPOCHS):
        print('')
        print('')
        print('------------------------------------------------------')
        print(f'----------- Starting instance: {epoch+1} ------------')
        
        
        # --------------------- INITIALIZE THE NETWORK ---------------------
        
        # Reset the parameters. Re-state the network:
        network = OneD_CNN(input_fs=fs_downsampled, input_size=Training_data_length, last_dropout = False,
                           head_dropout = True, downsampling_rate=2, groups=16,dropout_pers=0.2, Block_Type = Block_array[blk],
                           width_shrink = ws, Network_depth = depth, Stage_kernel = 3, embedding_size = es, 
                           Stem_augmentation = 16, Stem_kernel = 5, Stem_stride = 4,weight_multiplier = wm, device = device)

             

        # ------- Load the network on the device -------
        network = network.to(device)
                


        # --- Training algorithms ---

        reducer = reducers.AvgNonZeroReducer()
        loss_fn = losses.TripletMarginLoss(margin=Margin_loss,
                                           swap=True,
                                           distance=CosineSimilarity(), #  is preferred in embedding contexts.
                                           reducer = reducer,
                                           embedding_regularizer = None) # See in the obsidian page 'Deep learning Training Regularizers' the reason


        # --- Miner ---
        miner_hard = miners.TripletMarginMiner(margin=Margin_miner, type_of_triplets="hard",distance=CosineSimilarity()) 
                                        
        # --- Optimizer ---
        optimizer_fn = torch.optim.AdamW(network.parameters(),lr = lr, betas=[b1,b2],weight_decay=wd)
        
        
        network.train(True)
        # --------------------- TRAINING ---------------------
        training_loss = train_one_epoch(network,Dataloader_training,loss_fn,optimizer_fn,fs_downsampled,device,miner_hard,Margin_loss)
        
        
        temp_loss = temp_loss+training_loss
        print('')
        print('')
        print('------------------------------------------------------')

        
        
    average_loss = temp_loss/(EPOCHS)
    # The loss has already been normalized in the one epoch training loop
    average_loss_norm = average_loss
    
    return average_loss_norm
        


































# ------------------------- EMBEDDING QUALITY METRICS -------------------------

def Embedding_Scores(Embeddings,True_Labels,Visible = True):
    
    # ------------ K-MEANS ------------
    
    '''
    In practice, the k-means algorithm is very fast (one of the fastest clustering algorithms available), but it falls in local minima. 
    That’s why it can be useful to restart it several times (ruled by n_init term)
    
    Notes:
        1) It expects embeddings to be already scaled
    
    '''
    n_clusters = len(np.unique(True_Labels))    
    
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0)
    estimator = kmeans.fit(Embeddings)
    
    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results = [m(True_Labels, estimator.labels_) for m in clustering_metrics]
    
    
    
    if Visible == True:
        reduced_data = PCA(n_components=2).fit_transform(Embeddings)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
        kmeans.fit(reduced_data)
        
                # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )
        
        # plt.scatter(
        #         reduced_data[:, 0], reduced_data[:, 1], c=True_Labels, s=10, cmap=plt.cm.Paired
        #             )

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=10)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        plt.title(
            "K-means clustering on the embeddings (PCA-reduced data)\n"
            "Centroids are marked with white cross"
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
        
    return results,reduced_data

