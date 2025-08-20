# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:09:51 2025

@author: Admin
"""
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import os
from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F


def magnitude_warp(time_series,n_versions=10, intra_knot_dist=0.2, sigma_scale=0.2,fs= None):
    """
    Performs magnitude warping on a univariate time series.

    Args:
        time_series (np.ndarray): The input time series.
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
    T = len(time_series)
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


import numpy as np
from scipy.interpolate import CubicSpline, interp1d

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
    T = len(time_series)
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



#---------------------------------------- NEURONAL DYNAMICS ----------------------------------------
def Standardization(data):
    """
    Standardizes a multivariate time series by standardizing each feature (column) separately.

    Standardization (Z-score normalization) transforms the data to have a mean of 0 and a standard deviation of 1.
    The formula for standardization is: z = (x - mu) / sigma, where mu is the mean and sigma is the standard deviation.

    Args:
        data (np.ndarray): A 2D NumPy array of shape (timesteps, features).

    Returns:
        np.ndarray: A 2D NumPy array of the same shape as `data`, but with each feature standardized.
                    Returns None if the input data is not a 2D array.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        print("Error: Input data must be a 2D NumPy array.")
        return None

    # Get the number of timesteps and features
    timesteps, features = data.shape

    # Initialize an array to store the standardized data
    standardized_data = np.zeros_like(data)

    # Standardize each feature (column) separately
    for feature_index in range(features):
        feature_data = data[:, feature_index]
        mean = np.mean(feature_data)
        std_dev = np.std(feature_data)

        # Handle the case where the standard deviation is zero to avoid division by zero
        if std_dev == 0:
            print(f"Warning: Standard deviation for feature {feature_index} is zero. This feature will be all zeros.")
            standardized_data[:, feature_index] = 0
        else:
            standardized_data[:, feature_index] = (feature_data - mean) / std_dev

    return standardized_data

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

# The Smoothed_IFR function from the previous response is included here for completeness.
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
        Gaussian_window: The size of the Gaussian smoothing window.
        Visible: A boolean to control whether to display plots.
    
    Returns:
        IFR_smoothed: The smoothed IFR data.
        IFR_smoothed_concatenated: The concatenated smoothed IFR data.
    """

    from scipy.ndimage import gaussian_filter1d

    if Isolate_NB:
        num_nb = len(IFR)
        num_channels = IFR[0].shape[0] if num_nb > 0 else 0
        
        if num_nb > 0:
            samples_per_window = IFR[0].shape[1]
        else:
            samples_per_window = 0

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
        
        for j in range(num_nb):
            ifr_data = IFR[j]
            smoothed_channels = []
            for i in range(num_channels):
                channel = ifr_data[i, :]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
                smoothed_channels.append(smoothed_channel)
                
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
            plt.title('Smoothed IFR')
            plt.xlabel('Time [s]')
            plt.ylabel('Spikes')

            plt.subplot(2, 1, 2)
            t_vec_conc = np.arange(IFR_smoothed_concatenated.shape[1])
            plt.plot(t_vec_conc, IFR_smoothed_concatenated.T)
            plt.title('Concatenated smoothed NB IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()

    else:
        num_samples, num_channels = IFR.shape
        IFR_smoothed = np.zeros_like(IFR)
        IFR_smoothed_concatenated = []

        if Visible:
            plt.figure()
            plt.subplot(2, 1, 1)
            for i in range(num_channels):
                plt.plot(IFR[:, i])
            plt.title('Raw IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            
            plt.subplot(2, 1, 2)
            for i in range(num_channels):
                channel = IFR[:, i]
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
                IFR_smoothed[:, i] = smoothed_channel
                plt.plot(smoothed_channel)
            plt.title('Smoothed IFR')
            plt.xlabel('Samples')
            plt.ylabel('Spikes')
            plt.tight_layout()
            plt.show()
        else:
            for i in range(num_channels):
                channel = IFR[:, i]
                IFR_smoothed[:, i] = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
            
    return IFR_smoothed, IFR_smoothed_concatenated



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
    
        
def calculate_mean_burst_duration(time_series_data, time_step,fs):
    """
    Calculates the mean duration of bursts in a time series of network data.

    Args:
        time_series_data (list or np.array): The network data time series.
        time_step (float): The time interval between data points (e.g., 1 second).
        baseline (float): The threshold value that defines a burst.

    Returns:
        float: The mean duration of all detected bursts. Returns 0 if no bursts are found.
    """
    burst_durations = []
    in_burst = False
    current_burst_duration = 0
    baseline = np.mean(time_series_data)
    for data_point in time_series_data:
        if data_point > baseline:
            # We are currently in a burst
            if not in_burst:
                in_burst = True
                current_burst_duration = time_step
            else:
                current_burst_duration += time_step
        else:
            # We are not in a burst or just ended one
            if in_burst:
                burst_durations.append(current_burst_duration)
                in_burst = False
                current_burst_duration = 0

    # Handles the case where the time series ends while still in a burst
    if in_burst:
        burst_durations.append(current_burst_duration)

    if not burst_durations:
        return 0

    return np.mean(burst_durations)    

def Neuronal_traces(Char_folder=None,Char_base=None,Type ='Cumulative',t_rec = 600, fs = 10000, w_size = 0.02, overlap = 0.06, 
                    bin_size_s = 0.05, Isolate_NB = False, Gaussian_window = 0.05,
                     Visible = False,NB_statistics = True):
    
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
        [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        
        [IFR, bin_size,window_size] = Get_IFR(data,fs,Cumulative,t_vec,step_s,bin_size_s,Isolate_NB,T_max);
            
   
        fs_downsampled = 1/bin_size_s
        
    
   
        [IFR_smoothed, IFR_smoothed_concatenated] = Smoothed_IFR(IFR, bin_size,window_size,fs_downsampled,Isolate_NB,Gaussian_window,Visible);
    
    
    
        [Variance_explained, Projected_trajectories,Coefficients,NB_IFR_PCA_mean] = get_PCA(IFR_smoothed_concatenated,IFR_smoothed,Isolate_NB,Visible);
    
        
    
        return Projected_trajectories,Variance_explained,fs_downsampled
    
    
    elif Type == 'Cumulative':
        overlap = 0
        [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        
        # The new sampling frequency is downsampled by a factor determined by w_size [s]
        fs_downsampled = 1/w_size
        
        
        smoothed_cumulative =  get_Smoothed_Cumulative(Cumulative,fs_downsampled,Gaussian_window)
        
        
        if Visible == True:
            
            plt.figure()
            plt.plot(t_vec*fs_downsampled,smoothed_cumulative)
            plt.xlabel('Time [s]')
            plt.ylabel('IFR')
            plt.show()
            
            
            
        # if NB_statistics:
            
            
            
            
            
        
        
        return smoothed_cumulative,fs_downsampled
#%%

%matplotlib
    
Char_folder = r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11'
Char_base = 'ptrain_Control00_Well11_'
smoothed_cumulative,fs_downsampled = Neuronal_traces(Visible=True,Char_folder=Char_folder,Char_base=Char_base)
#%%


def calculate_mean_burst_duration(time_series_data, fs,scal_factor = 0.5, Visible=True):
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




burst_stat = calculate_mean_burst_duration(smoothed_cumulative,fs_downsampled, Visible=True)

#%%

# %matplotlib
# --- Example Usage ---
n_vers = 10
# Create a sample time series as a PyTorch tensor

# --- Magnitude Warping Example ---
print("Applying Magnitude Warping...")
magnitude_warped_series = magnitude_warp(torch.from_numpy(smoothed_cumulative),n_versions=n_vers, intra_knot_dist=0.25, sigma_scale=0.05,fs= fs_downsampled)




mse_mag = mean_squared_error(smoothed_cumulative, magnitude_warped_series[2,:].numpy())

# Plot the original vs. magnitude-warped signal
plt.figure(figsize=(10, 6))
plt.plot(smoothed_cumulative, label='Original Series', alpha=1)

for it in range(n_vers):
    plt.plot(magnitude_warped_series[it,:].numpy(), '--',alpha = 0.6 ,label='Magnitude Warped Series')  
plt.title(f'Magnitude Warping. MSE: {mse_mag:.4f}')
plt.legend()
plt.grid(True)
plt.show()


#%%
# --- Time Warping Example ---
print("\nApplying Time Warping...")
n_vers = 10
# Create a sample time series as a PyTorch tensor


# --- Magnitude Warping Example ---

time_warped_series = time_warp(torch.from_numpy(smoothed_cumulative),n_versions=10, intra_knot_dist=0.3, sigma_scale=0.2,fs=fs_downsampled)




# mse_mag = mean_squared_error(smoothed_cumulative, time_warped_series[:,:].numpy())

# Plot the original vs. magnitude-warped signal
plt.figure(figsize=(10, 6))
plt.plot(smoothed_cumulative, label='Original Series', alpha=1)

for it in range(n_vers):
    plt.plot(time_warped_series[it,:].numpy(), '--',alpha = 0.6 ,label='Magnitude Warped Series')  
plt.title(f'Magnitude Warping. MSE: {mse_mag:.4f}')
plt.legend()
plt.grid(True)
plt.show()

#%%
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


def Data_Augmentation(data,n_versions_insatances,n_param_vector,intra_knot_dist_range,sigma_scale_range,fs):
    '''
    
    This data augmentation algorithm first generates a certain number of surrogates and then splits negatives and 
    positives instances w.r.t. the mean squared error.
    Afterworks each insance in the Positives and Negatives will be shifted temporally.
    

    
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
    element_wise_mse = F.mse_loss(out_temp, data, reduction='none')
    
    # Compute the mean for each row
    mse_per_row = torch.mean(element_wise_mse, dim=1)
    
    
    # mse_per_row = mse_per_row.unsqueeze(1) # unsqueeze to make it 50x1

    # --- Step 1: Find the indices of values less than 0.1 ---
    # This creates a boolean tensor (mask) where `True` means the condition is met
    threshold_mask = mse_per_row < 0.1
    
    # Use torch.nonzero() to get the explicit indices if you need them
    indices_low_mse = torch.nonzero(threshold_mask)
    
    
    # --- Step 2: Divide the out_temp tensor into two groups ---
    # Group 1: Rows with MSE < 0.1
    Positives = out_temp[threshold_mask.squeeze()]
    
    # Group 2: Rows with MSE >= 0.1
    # The '~' operator inverts the boolean mask
    Negatives = out_temp[~threshold_mask.squeeze()]
            
    
    c = 0
    
    
    
    
    
    
intra_knot_dist_range = [0.15,0.3]
sigma_scale_range = [0.25,0.001]
Data_Augmentation(torch.from_numpy(smoothed_cumulative),5,5,intra_knot_dist_range,sigma_scale_range,fs_downsampled)









