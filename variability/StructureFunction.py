'''
This package computes the Structure function as implemented by Sergison+19 and Venuti+21, in order to apply it to GaiaDR3 data
'''

#################################
#Libraries

import numpy             as     np
import pandas            as     pd

from scipy.signal import find_peaks


###################################
# Structure function

def structure_function_slow(mag, time, num_bins = 100, epsilon = 0) :
    """
    Calculates the structure function of a magnitude time series using a slow method.

    Args:
        - mag, time (numpy.ndarray): magnitudes and times of the light curve
        - num_bins (int):            number of bins for time_bins
        - epsilon(int):              tolerence threshold (min nb of pairs required for a bin) 

    Returns:
        - sf (numpy.ndarray):        structure function
        - time_lag (numpy.ndarray):  corresponding log-spaced time bins.
        - it_list(ndarray)           number of pairs used to compute sf at each time lag
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

        lags_ind   = np.where((delta_t_>time_bin-bin_diff_d) & (delta_t_<time_bin+bin_diff_u))[0]
        
        while len(lags_ind) < epsilon :
        
            if len(lags_ind) == 0 :
                break 
                
            bin_diff_d += 0.05*bin_diff_d
            bin_diff_u += 0.05*bin_diff_d
        
            lags_ind    = np.where((delta_t_>time_bin-bin_diff_d) & (delta_t_<time_bin+bin_diff_u))[0]
             
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
    

    
################################################
#Derive timescale    
    
def find_timescale(sf, t_log, err = 0.01, thresh = 80):

    """
    Finds the characteristic timescale of a time-series from the Structure Function.

    Args:
        - sf, t_log (ndarray): Structure function and log time bins (structure function must be in mag**2)
        - err (float):         Error threshold. Defaults to 0.01. Used to define the noise-dominated regime of the sf
        - thresh (int):        Thresh_th percentile is used for a first approximation of tau_peak (timescale where most variability occurs)           

    Returns:
        - ts_sf (float):       The characteristic timescale. For a periodic time series, should be 1/2 the period
        - dict_fit (dict):     Fitting parameters for each regime of the SF
        - flag(bool):          True is no ts was retrieved with the code (in which case ts_sf will be NaN)
    """
    
    #Divide the SF into 3 regimes
    taus = [min(t_log),\
            t_log[np.where(sf >= 2 * err**2)[0][0]],\
            t_log[np.where(sf >= np.percentile(sf, thresh))[0][0]],\
            max(t_log)]

    sfs, ts = zip(*[(sf[(t_log >= taus[i]) & (t_log <= taus[i+1])],\
              t_log[(t_log >= taus[i]) & (t_log <= taus[i+1])]) for i in range(len(taus)-1)])


    #Fit each regime with a line in log-space
    dict_fit = {}
    
    for i, (ts_i, sf_i) in enumerate(zip(ts, sfs)):
    
        if len(ts_i) > 3 : #Considering it needs at least 3 points to fit a line
            
            xp   = np.log10(np.linspace(0.9*min(ts[i]), 2*max(ts[i]), 600))
            x, y = np.log10(ts[i]), np.log10(sfs[i])
            f    = np.poly1d(np.polyfit(x, y, 1))
            yp   = 10**f(xp)
            xp   = 10**xp

            dict_fit[f't_{i}'], dict_fit[f'sf_{i}'], dict_fit[f'f_{i}'] = xp, yp, f
            
        elif len(ts_i) > 0 and 0 < i < 3:
            
            ts_i = np.concatenate([ts_i, ts[i+1]])
            sf_i = np.concatenate([sf_i, sfs[i+1]])
            
            xp   = np.log10(np.linspace(0.9*min(ts[i]), 2*max(ts[i]), 600))
            x, y = np.log10(ts[i]), np.log10(sfs[i])
            f    = np.poly1d(np.polyfit(x, y, 1))
            yp   = 10**f(xp)
            xp   = 10**xp

            dict_fit[f't_{i}'], dict_fit[f'sf_{i}'], dict_fit[f'f_{i}'] = xp, yp, f
    
    f_1, f_2 = dict_fit['f_1'], dict_fit['f_2']
    a1, b1   = f_1[1], f_1[0]
    a2, b2   = f_2[1], f_2[0]

    # Find the crossing point between 2nd and 3rd regime in log scale
    log_x12 = (b2 - b1) / (a1 - a2)

    # Transform log_x12 back to linear scale
    tau_eq  = 10**log_x12
    
    
    #Find the first peak after the crossing point (which should correspond to tau_peak)
    peaks   = find_peaks(sf)[0]
    t_peaks = t_log[peaks]
    sel     = np.where((t_peaks>=tau_eq))[0]
    peaks   = t_peaks[sel]
    
    if len(peaks) > 0 :
        ts_sf   = peaks[0]
    else :
        print('no ts found')
        ts_sf = np.nan
    
    return ts_sf, dict_fit


