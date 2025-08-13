

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
        num_channels = len(data)
        num_bins = T_max // bin_size_samples
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
            Cumulative (numpy.ndarray): The cumulative activity.
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
            
        Cumulative[idx] = temp_cum / w_size
        
        start_index += step_size
        idx += 1
        
    return Cumulative, t_vec, step_size
    
def get_PCA(NB_IFR_smoothed_concatenated, IFR_smoothed, Isolate_NB):
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
        Gaussian_window: The size of the Gaussian smoothing window.
        Visible: A boolean to control whether to display plots.
    
    Returns:
        IFR_smoothed: The smoothed IFR data.
        IFR_smoothed_concatenated: The concatenated smoothed IFR data.
    """

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
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
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
                smoothed_channel = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
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
                IFR_smoothed[:, i] = gaussian_filter1d(channel.astype(float), sigma=Gaussian_window)
            
    return IFR_smoothed, IFR_smoothed_concatenated



def Neuronal_traces(t_rec = 600, fs = 10000, w_size = 0.12, overlap = 0.06, 
                    bin_size_s = 0.05, Isolate_NB = False, Gaussian_window = 2,
                    Char_folder = None, Char_base= None, Visible = False):
    
    
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
    # Gaussian_window = 2  # [samples]

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
        # [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
        # The following is a placeholder for the Rect_window function call
        # This part would need to be implemented in Python based on the MATLAB function's logic
        
        # Assume Cumulative, t_vec, and step_s are computed here
        # For example:
        # Cumulative, t_vec, step_s = Rect_window(fs, w_size, overlap, data, T_max)
    
        # # Let's assume we have Cumulative, t_vec, and step_s for the next part
        # # Example mock data for plotting:
        # t_vec = np.linspace(0, t_rec, int(T_max))
        # Cumulative = np.random.rand(len(t_vec)) * 100
    
        # Mean_IFR = np.mean(Cumulative)
        # STD_IFR = np.std(Cumulative)
        # plot_MFR = np.ones(len(t_vec)) * Mean_IFR
        # plot_MFR_STD_plus = np.ones(len(t_vec)) * (Mean_IFR + STD_IFR)
        # plot_MFR_STD_minus = np.ones(len(t_vec)) * (Mean_IFR - STD_IFR) # The MATLAB code had a mistake here
        
        # # findpeaks
        # # The 'MinPeakDistance' argument in MATLAB is different in Python's find_peaks
        # # In Python, distance is in samples, not seconds.
        # # The MATLAB code has 3 * fs / step_s, which should be adjusted for Python
        # # Assuming step_s is the sampling rate of Cumulative, not fs
        
        # # Let's assume a step_s value
        # step_s = 1000 # Example step_s value
        # idx, _ = find_peaks(Cumulative, height=Mean_IFR + STD_IFR, distance=int(3 * fs / step_s))
        # NB_T = t_vec[idx]
        
        # plt.figure()
        # plt.plot(Cumulative)
        # plt.plot(idx, Cumulative[idx], 'x')
        # plt.title('findpeaks')
        # plt.show()
    
        # plt.figure()
        # plt.title('Global Activity')
        # plt.plot(t_vec / fs, Cumulative, label='Cumulative IFR')
        # plt.plot(t_vec / fs, plot_MFR, linestyle='-.', color='r', linewidth=1.5, label='Mean IFR')
        # plt.plot(t_vec / fs, plot_MFR_STD_plus, linestyle='--', color='g', linewidth=1.5, label='Mean + STD')
        # plt.xlabel('Time [s]')
        # plt.ylabel('Instantaneous firing rate [spk/s]')
        # plt.legend()
        # plt.show()
    
    # Calculate NBs
    # You would need to define Rect_window in Python
    [Cumulative, t_vec, step_s] = Rect_window(fs, w_size, overlap, data, T_max)
    
    [IFR, bin_size,window_size] = Get_IFR(data,fs,Cumulative,t_vec,step_s,bin_size_s,Isolate_NB,T_max);
    
    
    [IFR_smoothed, IFR_smoothed_concatenated] = Smoothed_IFR(IFR, bin_size,window_size,fs,Isolate_NB,Gaussian_window,Visible);
    
    
    [Variance_explained, Projected_trajectories,Coefficients,NB_IFR_PCA_mean] = get_PCA(IFR_smoothed_concatenated,IFR_smoothed,Isolate_NB);
    fs_downsampled = 1/bin_size_s
    
    return Projected_trajectories,Variance_explained,fs_downsampled



    
Char_folder = r'C:\Users\Admin\Desktop\Leonardo\Neuronal Dynamic\Nuova cartella\ptrain_Control00_Well11'
Char_base = 'ptrain_Control00_Well11_'
Projected_trajectories,Variance_explained,fs_downsampled = Neuronal_traces(Visible=True,Char_folder=Char_folder,Char_base=Char_base)

#%%


# ------- DATA AUGMENTATION -------

'''
Data augmentation (DA) techniques for time series rests on traditional methdos or deep learning approaches.
Here I'll use traditional methdos which have as their basis the deformation, shortening, enlargment or modification
of the data samples of the dataset. 


Be aware that using the wrong techniques to build positive or negative instances may lead to NEGATIVE TRAINING

Paper: Data Augmentation techniques in time series domain: A survey and  taxonomy

Data augmentation techniques can be used also to generate randomized surrogates which will be then used as negatives 
of the original data set in the contrastive learning framework.

Methods for positives:
    flip, shift or a combination 
    
Methods for negatives:
    Heavy Jittering, Permutation, time slicing window or a combination 


Pathological traces can be used as negatives.    

Warping techniques consists in define a set of knots u, scale the data set value at their positions, scale in magnitude
or shift in time the knots and interpolate with a cubic spline.


WHen converting to tensor the output of positives and negatives functions the tensor's sizes are:
    0 : number of bathces
    1 : data size
    2 : number of feature
 
The input tensor size required from the GRU cell is (L,N,Hin) or (N,L,Hin) if batch first is True,
where:
    L = sequence length
    N = batch size
    Hin = input size

'''
Positive_DA_methods = ['Flip','Shift']
Negative_DA_methods = ['Jittering','Permutation','Time Slicing Window']
# ----------------------------------- TORCH IMPLEMENTATION -----------------------------------

import torch
import math
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


def Shift(time_series, n_versions, shift_magnitude_s, fs):
    """
    Generates n randomly shifted versions of a multivariate time series.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        n_versions (int): The number of shifted versions to generate.
        shift_magnitude (float): The maximum absolute value of the random shift (in ms).
        fs (int): The sampling frequency in Hz.

    Returns:
        tuple: A tuple containing:
            - shifted_series (list): A list of torch.Tensors, where each tensor is a
                                     randomly shifted version of the original time series.
            - shifts (list): A list of the random integer shifts applied to each version.
    """
    
   
    max_shift_samples = int(shift_magnitude_s * fs)
    
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    if n_versions <= 0 or max_shift_samples < 0:
        raise ValueError("'n_versions' must be > 0 and 'max_shift_samples' must be >= 0.")
    
    shifted_series = []
    shifts = []

    for _ in range(n_versions):
        # Generate a random integer shift between -max_shift and +max_shift
        # Using torch.randint for random integer generation
        shift_value = torch.randint(low=-max_shift_samples, high=max_shift_samples + 1, size=(1,)).item()
        
        # Use torch.roll to apply a circular shift to the samples (axis=0 or dim=0)
        shifted_version = torch.roll(time_series, shifts=shift_value, dims=0)
        
        shifted_series.append(shifted_version)
        shifts.append(shift_value)

    return shifted_series, shifts
    
def Homogeneous_scaling(time_series, n_versions, n_max, n_min):
    """
    Generates n versions of a multivariate time series, each with a homogeneous scaling.

    The scaling factor for each version is randomly chosen from the specified range.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        n_versions (int): The number of scaled versions to generate.
        n_std_max (float): The maximum number of standard deviations to fix the max scaling factor.
        n_std_min (float): The minimum number of standard deviations to fix the min scaling factor.
    
    Returns:
        tuple: A tuple containing:
            - scaled_series (list): A list of torch.Tensors, where each tensor is a
                                     randomly scaled version of the original time series.
            - scaling_factors (list): A list of the random scaling factors applied to each version.
    
    Raises:
        TypeError: If the input 'time_series' is not a PyTorch tensor.
        ValueError: If the number of versions is not positive.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    if n_versions <= 0:
        raise ValueError("'n_versions' must be a positive integer.")
    
    scaled_series = []
    scaling_factors = []

    for _ in range(n_versions):
        # Generate a random scaling factor as a float within the specified range
        # Using torch.rand to generate a uniform random number between [0, 1)
        # and then scaling it to the desired range
        min_factor = n_min
        max_factor = n_max
        
        scaling_factor = torch.rand(1).item() * (max_factor - min_factor) + min_factor
        
        # Apply the scaling to the entire time series
        scaled_version = time_series * scaling_factor
        
        scaled_series.append(scaled_version)
        scaling_factors.append(scaling_factor)
    
    return scaled_series, scaling_factors

def Positives(data, shift_magnitude_s=30, n_versions=10, n_max=2, n_min=0.5, fs=None, Generation_method='Combination'):
    """
    Generates "positive" versions of a time series using various data augmentation techniques.
    
    Args:
        data (torch.Tensor): The original time series.
        shift_magnitude (int): Max shift magnitude in ms for the 'Shift' method.
        n_versions (int): Number of versions to generate for each method.
        n_std_max (float): Max scaling factor for the 'Scaling' method.
        n_std_min (float): Min scaling factor for the 'Scaling' method.
        fs (int): Sampling frequency in Hz.
        Generation_method (str): The augmentation method to use. 'Combination', 'Shift', or 'Scaling'.
        
    Returns:
        list: A list of augmented time series (torch.Tensor).
    """
    if Generation_method == 'Combination':
        out_ = []
        
        # Apply both techniques
        data_shifted, shifts = Shift(data, n_versions, shift_magnitude_s, fs)
        
        for data_shift in data_shifted:
            data_scaled, scales = Homogeneous_scaling(data_shift, n_versions, n_max, n_min)
            out_.append(data_scaled)
            
        # Flatten the list of lists
        out = [item for sublist in out_ for item in sublist]
    
    elif Generation_method == 'Shift':
        out, shifts = Shift(data, n_versions, shift_magnitude_s, fs)
    
    elif Generation_method == 'Scaling':
        out, scales = Homogeneous_scaling(data, n_versions, n_max, n_min)
    
    else:
        raise ValueError("Generation_method must be 'Combination', 'Shift', or 'Scaling'.")

    return out

def Permutation(time_series, n_versions, window_size_s, fs):
    """
    Divides a multivariate time series into windows, shuffles the windows,
    and reconstructs the time series. This process is repeated n times.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        n_versions (int): The number of permuted versions to generate.
        window_size_ms (int): The window size in milliseconds.
        fs (int): The sampling frequency in Hz.

    Returns:
        list: A list of torch.Tensors, where each tensor is a new,
              randomly permuted version of the time series.
    
    Raises:
        ValueError: If window_size is not a positive integer.
        TypeError: If time_series is not a 2D PyTorch tensor.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    
    num_samples, num_features = time_series.shape
    
    # Convert window size from ms to samples
    window_size = int(window_size_s * fs )
    
    if window_size <= 0:
        raise ValueError("'window_size_ms' must be a positive integer.")
    if num_samples < window_size:
        raise ValueError(f"Number of samples ({num_samples}) is less than window size ({window_size}).")
    
    # Truncate the time series so its length is a multiple of the window size
    truncated_length = (num_samples // window_size) * window_size
    if truncated_length < num_samples:
        print(f"Warning: Time series length is not a multiple of window_size. "
              f"Truncating from {num_samples} to {truncated_length} samples.")
    
    truncated_ts = time_series[:truncated_length, :]

    # Reshape the time series into windows (3D tensor)
    num_windows = truncated_length // window_size
    windows = truncated_ts.reshape(num_windows, window_size, num_features)

    permuted_versions = []
    for _ in range(n_versions):
        # Generate a random permutation of window indices
        # torch.randperm(n) returns a random permutation of integers from 0 to n-1
        permuted_indices = torch.randperm(num_windows)
        
        # Reorder the windows using the permuted indices
        shuffled_windows = windows[permuted_indices]
        
        # Reshape the windows back into a single time series
        reconstructed_ts = shuffled_windows.reshape(truncated_length, num_features)
        
        permuted_versions.append(reconstructed_ts)

    return permuted_versions

def Jittering(time_series, pers_std):
    """
    Adds homogeneous Gaussian noise to a multivariate time series.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        pers_std (float): Percentage of the data's standard deviation that is used
                          as the standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: A new tensor representing the jittered time series.
    
    Raises:
        TypeError: If the input 'time_series' is not a PyTorch tensor.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    
    # Calculate the standard deviation of the data per channel (feature)
    # torch.std() with dim=0 computes the std for each column
    stds = torch.std(time_series, dim=0, keepdim=True)
    
    # Define the noise's magnitude based on the percentage of the data's std
    std_factor_array = pers_std * stds
    
    # Generate a random tensor from a Gaussian distribution with mean=0 and a given std
    # The `std_factor_array` is broadcasted to the full size of the noise tensor
    # torch.randn generates numbers from a standard normal distribution (mean=0, std=1)
    # We then scale and shift it to our desired distribution.
    noise = torch.randn(time_series.shape) * std_factor_array
    
    # Add the noise to the original time series
    jittered_series = time_series + noise
    
    return jittered_series

def Negatives(data, window_size_s=5, n_versions=5 ,n_max = 4,n_min = 0.1, fs=None, Generation_method='Combination'):
    """
    Generates "negative" versions of a time series using various data augmentation techniques.
    
    Args:
        data (torch.Tensor): The original time series.
        window_size_ms (int): Window size in ms for the 'Permutation' method.
        n_versions (int): Number of versions to generate for each method.
        n_std_max (float): Max scaling factor for the 'Scaling' method.
        n_std_min (float): Min scaling factor for the 'Scaling' method.
        fs (int): Sampling frequency in Hz.
        Generation_method (str): The augmentation method to use. 'Combination', 'Permutation', or 'Scaling'.
        
    Returns:
        list: A list of augmented time series (torch.Tensor).
    """
    if Generation_method == 'Permutation':
        out = Permutation(data, n_versions, window_size_s, fs)
    
    elif Generation_method == 'Combination':
        out_ = []
        
        # Apply both techniques
        data_permuted = Permutation(data, n_versions, window_size_s, fs)
        
        for data_perm in data_permuted:
            # Note: The original numpy code had an issue here, using a variable 'data_shift'
            # which wasn't defined in this loop. I've corrected it to use 'data_perm'.
            data_scaled, scales = Homogeneous_scaling(data_perm, n_versions, n_max, n_min)
            out_.append(data_scaled)
        
        # Flatten the list of lists
        out = [item for sublist in out_ for item in sublist]
    
    elif Generation_method == 'Scaling':
        out, scales = Homogeneous_scaling(data, n_versions, n_max, n_min)
    
    else:
        raise ValueError("Generation_method must be 'Combination', 'Permutation', or 'Scaling'.")
    
    return out


Projected_trajectories_ = Standardization(Projected_trajectories[:,0:3])      
Projected_trajectories_torch = torch.from_numpy(Projected_trajectories_)
        
out =  Positives(Projected_trajectories_torch[:,0:3],shift_magnitude_s=30, n_versions=10, n_max=2, n_min=0.5, fs=fs_downsampled, Generation_method='Combination')       
        
        
        
        


#%%
iterat = 4
plt.figure()
support = np.linspace(0,Projected_trajectories_torch.shape[0],Projected_trajectories_torch.shape[0])
plt.plot(support,Projected_trajectories_torch[:,0],'k')    
plt.plot(support,out[iterat][:,0],'b')       
plt.show()   
    
plt.figure()
support = np.linspace(0,Projected_trajectories_torch.shape[0],Projected_trajectories_torch.shape[0])
plt.plot(support,Projected_trajectories_torch[:,1],'k')    
plt.plot(support,out[iterat][:,1],'b')       
plt.show()   
       
   
plt.figure()
support = np.linspace(0,Projected_trajectories_torch.shape[0],Projected_trajectories_torch.shape[0])
plt.plot(support,Projected_trajectories_torch[:,2],'k')    
plt.plot(support,out[iterat][:,2],'b')       
plt.show()   
      
    
    
    
#%%
# ----------------------------------- TORCH IMPLEMENTATION -----------------------------------
def Permutation(time_series, n_versions, window_size_ms, fs):
    """
    Divides a multivariate time series into windows, shuffles the windows,
    and reconstructs the time series. This process is repeated n times.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        n_versions (int): The number of permuted versions to generate.
        window_size_ms (int): The window size in milliseconds.
        fs (int): The sampling frequency in Hz.

    Returns:
        list: A list of torch.Tensors, where each tensor is a new,
              randomly permuted version of the time series.
    
    Raises:
        ValueError: If window_size is not a positive integer.
        TypeError: If time_series is not a 2D PyTorch tensor.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    
    num_samples, num_features = time_series.shape
    
    # Convert window size from ms to samples
    window_size = int(window_size_ms * fs / 1000)
    
    if window_size <= 0:
        raise ValueError("'window_size_ms' must be a positive integer.")
    if num_samples < window_size:
        raise ValueError(f"Number of samples ({num_samples}) is less than window size ({window_size}).")
    
    # Truncate the time series so its length is a multiple of the window size
    truncated_length = (num_samples // window_size) * window_size
    if truncated_length < num_samples:
        print(f"Warning: Time series length is not a multiple of window_size. "
              f"Truncating from {num_samples} to {truncated_length} samples.")
    
    truncated_ts = time_series[:truncated_length, :]

    # Reshape the time series into windows (3D tensor)
    num_windows = truncated_length // window_size
    windows = truncated_ts.reshape(num_windows, window_size, num_features)

    permuted_versions = []
    for _ in range(n_versions):
        # Generate a random permutation of window indices
        # torch.randperm(n) returns a random permutation of integers from 0 to n-1
        permuted_indices = torch.randperm(num_windows)
        
        # Reorder the windows using the permuted indices
        shuffled_windows = windows[permuted_indices]
        
        # Reshape the windows back into a single time series
        reconstructed_ts = shuffled_windows.reshape(truncated_length, num_features)
        
        permuted_versions.append(reconstructed_ts)

    return permuted_versions

def Jittering(time_series, pers_std):
    """
    Adds homogeneous Gaussian noise to a multivariate time series.

    Args:
        time_series (torch.Tensor): The original time series with shape (samples, features).
        pers_std (float): Percentage of the data's standard deviation that is used
                          as the standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: A new tensor representing the jittered time series.
    
    Raises:
        TypeError: If the input 'time_series' is not a PyTorch tensor.
    """
    if not isinstance(time_series, torch.Tensor) or time_series.ndim != 2:
        raise TypeError("Input 'time_series' must be a 2D PyTorch tensor.")
    
    # Calculate the standard deviation of the data per channel (feature)
    # torch.std() with dim=0 computes the std for each column
    stds = torch.std(time_series, dim=0, keepdim=True)
    
    # Define the noise's magnitude based on the percentage of the data's std
    std_factor_array = pers_std * stds
    
    # Generate a random tensor from a Gaussian distribution with mean=0 and a given std
    # The `std_factor_array` is broadcasted to the full size of the noise tensor
    # torch.randn generates numbers from a standard normal distribution (mean=0, std=1)
    # We then scale and shift it to our desired distribution.
    noise = torch.randn(time_series.shape) * std_factor_array
    
    # Add the noise to the original time series
    jittered_series = time_series + noise
    
    return jittered_series

def Negatives(data, window_size_ms=10, n_versions=5,n_max = 2,n_min = 0.5, fs=10000, Generation_method='Combination'):
    """
    Generates "negative" versions of a time series using various data augmentation techniques.
    
    Args:
        data (torch.Tensor): The original time series.
        window_size_ms (int): Window size in ms for the 'Permutation' method.
        n_versions (int): Number of versions to generate for each method.
        n_std_max (float): Max scaling factor for the 'Scaling' method.
        n_std_min (float): Min scaling factor for the 'Scaling' method.
        fs (int): Sampling frequency in Hz.
        Generation_method (str): The augmentation method to use. 'Combination', 'Permutation', or 'Scaling'.
        
    Returns:
        list: A list of augmented time series (torch.Tensor).
    """
    if Generation_method == 'Permutation':
        out = Permutation(data, n_versions, window_size_ms, fs)
    
    elif Generation_method == 'Combination':
        out_ = []
        
        # Apply both techniques
        data_permuted = Permutation(data, n_versions, window_size_ms, fs)
        
        for data_perm in data_permuted:
            # Note: The original numpy code had an issue here, using a variable 'data_shift'
            # which wasn't defined in this loop. I've corrected it to use 'data_perm'.
            data_scaled, scales = Homogeneous_scaling(data_perm, n_versions, n_max, n_min)
            out_.append(data_scaled)
        
        # Flatten the list of lists
        out = [item for sublist in out_ for item in sublist]
    
    elif Generation_method == 'Scaling':
        out, scales = Homogeneous_scaling(data, n_versions, n_max, n_min)
    
    else:
        raise ValueError("Generation_method must be 'Combination', 'Permutation', or 'Scaling'.")
    
    return out






# ----------------------------------- NUMPY IMPLEMENTATION -----------------------------------
# def Permutation(time_series, n_versions, window_size_ms,fs):
#     """
#     Divides a multivariate time series into windows, shuffles the windows,
#     and reconstructs the time series. This process is repeated n times.

#     Args:
#         time_series (np.ndarray): The original time series with shape (samples, features).
#         n_versions (int): The number of permuted versions to generate.
#         window_size (int): The number of samples in each window.

#     Returns:
#         list: A list of np.ndarrays, where each array is a new,
#               randomly permuted version of the time series.

#     Raises:
#         ValueError: If window_size is not a positive integer.
#         TypeError: If time_series is not a 2D NumPy array.
#     """
#     if not isinstance(time_series, np.ndarray) or time_series.ndim != 2:
#         raise TypeError("Input 'time_series' must be a 2D NumPy array.")
    

#     num_samples, num_features = time_series.shape
    
    
#     # COnvert window size in samples
#     fs_ms = fs/1000
#     window_size = int(window_size_ms*fs_ms)
    
    
#     # Truncate the time series so its length is a multiple of the window size
#     truncated_length = (num_samples // window_size) * window_size
#     if truncated_length < num_samples:
#         print(f"Warning: Time series length is not a multiple of window_size. "
#               f"Truncating from {num_samples} to {truncated_length} samples.")
              
#     truncated_ts = time_series[:truncated_length, :]

#     # Reshape the time series into windows (3D array)
#     num_windows = truncated_length // window_size
#     windows = truncated_ts.reshape(num_windows, window_size, num_features)

#     permuted_versions = []
#     for _ in range(n_versions):
#         # Generate a random permutation of window indices
#         permuted_indices = np.random.permutation(num_windows)
        
#         # Reorder the windows using the permuted indices
#         shuffled_windows = windows[permuted_indices]
        
#         # Reshape the windows back into a single time series
#         reconstructed_ts = shuffled_windows.reshape(truncated_length, num_features)
        
#         permuted_versions.append(reconstructed_ts)

#     return permuted_versions        
          
# def Jittering(time_series, pers_std):
#     """
#     Adds homogeneous Gaussian noise to a multivariate time series.

#     Args:
#         time_series (np.ndarray): The original time series with shape (samples, features).
#         pers_std (float): persantage of the data std that is used as The standard deviation (sigma) of the Gaussian noise.
#                                  The mean (mu) is always zero.

#     Returns:
#         np.ndarray: A new array representing the jittered time series.
    
#     Raises:
#         TypeError: If the input 'time_series' is not a NumPy array.
#     """
#     if not isinstance(time_series, np.ndarray) or time_series.ndim != 2:
#         raise TypeError("Input 'time_series' must be a 2D NumPy array.")
    
#     # Define the noise's magnitude
#     # Calculate the std of the data per channel
#     stds = np.std(time_series, axis=0)
     
#     std_factor_array = pers_std*stds
    
    
    
    
#     # Generate a random array from a Gaussian distribution with mean=0 and given sigma
#     noise = np.random.normal(loc=0.0, scale=std_factor_array, size=time_series.shape)
    
#     # Add the noise to the original time series
#     jittered_series = time_series + noise
    
#     return jittered_series    
    



# def Negatives(data, window_size_ms = 10,n_versions = 5,n_std_max = 0.1,n_std_min = 0.5,fs = 10000, Generation_method= 'Combination'):
    
#     if  Generation_method == 'Permutation':
        
#         out = Permutation(data, n_versions, window_size_ms,fs)
    
    
    
#     elif Generation_method == 'Combination':
        
#         out_ = []
        
#         # Apply both techniques
#         Data_permutated = Permutation(data, n_versions, window_size_ms,fs)
        
#         for data_shift in Data_permutated:
            
#             Data_scaled,scales = Homogeneous_scaling(data_shift, n_versions, n_std_max,n_std_min)
            
            
#             out_.append(Data_scaled)
        
        
#         out = [inner_list for row in out_ for inner_list in row]
        
        
#         return out



n_versions = 10
window_size_ms = 10 # [ms]
fs = 10000
pers_std = 0.1
# Permutated_data = Permutation(Projected_trajectories[:,0:3], n_versions, window_size_ms,fs)
Negative_data = Negatives(Projected_trajectories[:,0:3],window_size_ms=10, n_versions=5,n_max = 2,n_min = 0.5, fs=10000, Generation_method='Combination')


 #%%
iterat = 2
plt.figure()
support = np.linspace(0,Projected_trajectories.shape[0],Projected_trajectories.shape[0])
plt.plot(support,Projected_trajectories[:,0],'k')    
plt.plot(support,Negative_data[iterat][:,0],'b')       
plt.show()   
    
plt.figure()
support = np.linspace(0,Projected_trajectories.shape[0],Projected_trajectories.shape[0])
plt.plot(support,Projected_trajectories[:,1],'k')    
plt.plot(support,Negative_data[iterat][:,1],'b')       
plt.show()   
       
   
plt.figure()
support = np.linspace(0,Projected_trajectories.shape[0],Projected_trajectories.shape[0])
plt.plot(support,Projected_trajectories[:,2],'k')    
plt.plot(support,Negative_data[iterat][:,2],'b')       
plt.show()   




