'''
This package computes the Structure function as implemented by Sergison+19 and Venuti+21, in order to apply it to GaiaDR3 data
'''

#################################
#Libraries

import numpy             as     np
import pandas            as     pd


def structure_function_slow(mag, time, num_bins = 100, epsilon = 0) :
    """
    Calculate the structure function of a magnitude time series using a slow method.

    Parameters:
    - mag, time (numpy.ndarray): magnitudes and times of the light curve
    - interp : if False, no interpolation is made and a discrete SF is returned
    - num_bins : number of bins for time_bins
    - epsilon : tolerence threshold (min nb of pairs required for a bin) 

    Returns:
    - sf (numpy.ndarray): The calculated structure function
    - time_bins (numpy.ndarray): corresponding log-spaced time bins.
    """
    
    tmin, tmax = min(np.diff(time)), max(time)-min(time)
    
    # Create logarithmically spaced time bins
    log_bins  = np.logspace(np.log10(tmin), np.log10(tmax), num=num_bins+1, endpoint=True, base=10.0)
    time_bins = (log_bins[:-1] + log_bins[1:]) / 2.0

    sf       = np.zeros(len(time_bins))
    it_list  = []

    #Compute all time lags, corresponding delta_mags
    delta_t_  = []
    delta_mag = []
    
    for i in range(len(time)):
        for j in range(i+1, len(time)):
            dt = abs(time[i] - time[j])
            delta_t_.append(dt)
            delta_mag.append((mag[i] - mag[j])**2)
            
    delta_t_  = np.array(delta_t_)
    delta_mag = np.array(delta_mag)
    
    sort      = np.argsort(delta_t_)
    delta_t_, delta_mag = delta_t_[sort], delta_mag[sort]
    
    #Loop to adjust the time bins to have at least epsilon pairs per time bin :
    for i, time_bin in enumerate(time_bins) :
        
        bin_diff_d = abs(time_bin - log_bins[i])
        bin_diff_u = abs(time_bin - log_bins[i+1])

        lags_ind   = np.where((delta_t_>time_bin-bin_diff_d) & \
                              (delta_t_<time_bin+bin_diff_u))[0]
        
        while len(lags_ind) < epsilon :
        
            if len(lags_ind) == 0 :
                break
                
            bin_diff_d += 0.05*bin_diff_d
            bin_diff_u += 0.05*bin_diff_d
        
            lags_ind    = np.where((delta_t_>time_bin-bin_diff_d) & \
                                  (delta_t_<time_bin+bin_diff_u))[0]
            #print(len(lags_ind))  
        if len(lags_ind)>0 :
            sf[i] = 1/len(lags_ind) * sum(delta_mag[lags_ind])

        else :
            sf[i] = np.nan
            
        it_list.append(len(lags_ind))

    #Remove nan
    sel                     = ~np.isnan(sf)
    sf_sel, time_lag_sel    = sf[sel], np.array(time_bins)[sel]
    it_list                 = np.array(it_list)[sel]
    
    return sf_sel, time_lag_sel, np.array(it_list)
    
def structure_function_roll(mag, time, num_bins = 100, epsilon = 1) :
    """
    Calculate the structure function of a magnitude time series. 
    The structure function at each time is evaluated on a 'rolling' fashion

    Parameters:
    - mag, time (numpy.ndarray): magnitudes and times of the light curve
    - interp : if False, no interpolation is made and a discrete SF is returned
    - num_bins : number of bins for time_bins
    - epsilon : tolerence threshold

    Returns:
    - sf (numpy.ndarray): The calculated structure function
    - time_bins (numpy.ndarray): corresponding log-spaced time bins.
    """
    
    tmin, tmax = min(np.diff(time)), max(time)-min(time)
    
def find_timescale(sf, tau):
    """
    Find the timescale corresponding to the first peak or plateau in the structure function.

    Parameters:
    - sf, tau (numpy.ndarray): The Structure function.

    Returns:
    - timescale (int): The timescale corresponding to the first peak or plateau.
    """
    
    slope     = np.gradient(sf)
    signs     = np.sign(slope)
    
    peaks = []
    for i, sign in enumerate(signs) :  #find a better way to compute this
        if i>=2 :
            if (sign == -1 or sign == 0) and signs[i-1] == 1:# and signs[i-2] == 1 :
                peaks.append(i)
                
    tau_high = np.array(tau)[peaks]   
    ind_tau  = np.where(tau_high > 0.7)[0][0]

    return tau_high[ind_tau]

