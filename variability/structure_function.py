"""
Module for computing structure functions.

@juliaroquette

__Last update__: 24/07/2025

"""
import numpy as np
import pandas as pd
from iminuit import Minuit
from variability.lightcurve import LightCurve

class StructureFunction(LightCurve):
    """
    Class to compute structure functions from time series data.
    It provides methods to compute the structure function, fit it,
    and adaptively bin the results.
    """
    
    def __init__(self, **kwargs):
        if 'lc' in kwargs:
            # can take a LightCurve object as input
            super().__init__(kwargs['lc'].time, kwargs['lc'].mag, kwargs['lc'].err)
            self.lc = kwargs['lc']
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            # otherwise, can take time, mag and err arrays as input and define a LightCurve object
            super().__init__(kwargs['time'], kwargs['mag'], kwargs['err'], kwargs.get('mask', None))
            self.lc = LightCurve(self.time, self.mag, self.err)
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")
        
    def get_sf(self):
        """
        Uses get_sf_time_lags(time, mag, err) to return the
        structure function time lags, values and errors.
        - Note that  get_sf_time_lags(time, mag, err) expects
        time, mag and err to be sorted in ascending time order.
        The outputed data is also sorted as a function of ascending time_lags.
        """
        self.time_lag, self.sf_vals, self.sf_err = get_sf_time_lags(self.lc.time, self.lc.mag, self.lc.err)
    
    def bin_sf(self, **kwargs):
        """
        Uses adaptative_binning to bin the structure function data.
        The output is stored in the class attributes sf_binned, time_bins, sf_bin_err, pair_counts and irregular_bin.
        """
        num_bins=100, log=True, hybrid=False,
        min_bin_num=5, max_expand_factor=3.0        
        num_bins = kwargs.get('num_bins', 1.5*len(self.time))
        self.sf_binned, self.time_bins, self.sf_bin_err, self.pair_counts, self.irregular_bin = adaptative_binning(
            self.time_lag, self.sf_vals, self.sf_err,
            num_bins=100, log=True, hybrid=False,
            min_bin_num=5, max_expand_factor=3.0
        )



def get_sf_time_lags(time, mag, err):
    """
    Computes structure function and time lags.
    Assumes time is sorted in ascending order.
    Computes SF = (mj - mi)^2 and lag = tj - ti for j > i.
    This SF definition corresponds to the classical first order
    SF as in Hughes, et al. 1992, ApJ, 396, 469
    Estimate uncertainty through error propagation.
    Returns sf ordered by time lag.
    
    Parameters:
        time: array of time values (sorted in ascending order)
        mag:  array of magnitude values (same length as time)
        err:  array of photometric uncertainties (same length as time)
    Returns:
        time_lag: array of time lags (tj - ti)
        sf_vals:  array of structure function values ((mj - mi)^2)
        sf_err:   array of structure function errors 
                  (estimated from photometric uncertainties)
    
    TO DO: what if no uncertainties are provided?
    """
    N = len(time)
    # since time is sorted, tj > ti
    i_idx, j_idx = np.triu_indices(N, k=1)  # j > i

    # Time lags: tj - ti ensures positive values since time is sorted
    time_lag = time[j_idx] - time[i_idx]
    sf_vals = (mag[j_idx] - mag[i_idx])**2

    # Error propagation based on photometric uncertainties
    # y = u**2, where u =(x-z) -> e_y**2 = (e_x*(dy/du)*du/dx)**2 + (e_z*(dy/dv)*dv/dz)**2
    # Here, dy/du = 2*(x-z), dy/dv = -2*(x-z)
    # e_y**2 = (2*(x-z)*e_x)**2 + (2*(x-z)*e_z)**2
    # e_y = 2*(x-z)*sqrt(e_x**2 + e_z**2)
    sf_err = 2 * np.abs(mag[j_idx] - mag[i_idx]) * np.sqrt(err[j_idx]**2 + err[i_idx]**2)

    # Sort output by time-lag
    sort_idx = np.argsort(time_lag)

    return time_lag[sort_idx], sf_vals[sort_idx], sf_err[sort_idx]

def adaptative_binning(time_lag, sf_values, sf_err=None,
                       num_bins=100, 
                       log=True,
                       hybrid=False,
                       min_bin_num=5, 
                       max_expand_factor=3.0):
    """
    Suite of binning strategies for structure function data.
    Generally, data can be binned in either log/linear space, while ensuring
    that bins have at least min_bin_num pairs.
    Parameters:
        time_lag:    array of SF time lags
        sf_values:   array of SF values (same length as time_lag)
        sf_err:      array of SF errors (same length as time_lag)
                     If None, error is estimated from sf_values
        num_bins:    desired number of bins (centers)
                     Considers the number of bins in the context
                     of uniformly spaced time lags. The real 
                     number of populated bins will be less or 
                     equal to num_bins.
        scale:       'log' or 'linear'          
        min_bin_num:   limit number of pairs per bin. 
                    If hybrid=False: if the number of pairs in a bin 
                    is less than min_bin_num, the bin is iteratively 
                    expanded around its center by a factor of 1.1 until
                    a maximum expansion factor (set by `max_expand_factor)
                    is reached. 
                    If hybrid=False, if the number of pairs in a bin is less
                    than min_bin_num, the few existent SF pairs are stored in 
                    a list for later processing.
        hybrid:      True/False
        max_expand_factor: maximum factor by which to expand bin width
    Returns:
        sf_binned, time_bins, sf_bin_err, pair_counts, irregular_bin
        where:
        - sf_binned:   binned SF values
        - time_bins:   centers of the bins
        - sf_bin_err:  binned SF errors 
                       (or standard error of the mean if sf_err is None)
        - pair_counts: number of pairs in each bin
        - irregular_bin: flag indicating if the bin is irregular 
                         1.0 if regular, 
                         -1.0 if using individual pairs
                         >1.0 final factor of bin expansion
    Note on binning strategies::
        - If `log` is True, bins are logarithmically spaced. 
            When data is unevenly spaced, and sparse, log binning is preferred.
        
    """
    tmin, tmax = np.min(time_lag), np.max(time_lag)
    # Bin centers
    if log:
        bin_edges = np.logspace(np.log10(tmin), np.log10(tmax), num_bins+1)
    else:
        bin_edges = np.linspace(tmin, tmax, num_bins+1)
    
    sf_binned, time_bins, sf_bin_err, pair_counts, irregular_bin = [], [], [], [], []
    bin_idxs = np.digitize(time_lag, bin_edges, right=True) - 1  # subtract 1 for 0-based bins
    for i in range(len(bin_edges) - 1):

        # Get indices for current bin
        bin_idx = np.where(bin_idxs == i)[0]
        # test if there is at least one pair in the bin
        # otherwise discard it 
        
        if (len(bin_idx) > 0) :

            # get the center of the bin
            bin_left = bin_edges[i]
            bin_right = bin_edges[i + 1]
            if log: #geometric mean
                centre = np.sqrt(bin_left * bin_right)
            else:
                centre = (bin_left + bin_right) / 2.0
            # test if bin has enough pairs    
            if len(bin_idx) >= min_bin_num:
                time_bins.append(centre)
                sf_binned.append(np.mean(sf_values[bin_idx]))
                pair_counts.append(len(bin_idx))
                irregular_bin.append(1.0)
                if sf_err is not None:
                    # error propagation
                    sf_bin_err.append(np.sqrt(np.sum(sf_err[bin_idx]**2)) / (len(bin_idx)))
                else:
                    # standard error of the mean
                    sf_bin_err.append(np.std(sf_values[bin_idx], ddof=1) / np.sqrt(len(bin_idx)))
            elif len(bin_idx) > 0:
                # if hybrid binning is chose, append the few datapoints
                if bool(hybrid):
                    for j in bin_idx:
                        # if no bin is calculated, centre data at its own place
                        time_bins.append(time_lag[j])
                        sf_binned.append(sf_values[j])
                        pair_counts.append(1)
                        irregular_bin.append(-1.0)
                        if sf_err is not None:
                            sf_bin_err.append(sf_err[j])
                        else:
                            # if no uncertainty is provided, use the average of the bin errors calculated so far
                            sf_bin_err.append(np.nanmedian(sf_bin_err))
                else:
                    # if hybrid binning is not chosen, attempt to
                    # expand bin until either it has enough pairs 
                    # or it reaches the maximum expansion factor     
                    expand = 1.0           
                    while len(bin_idx) < min_bin_num and expand <= max_expand_factor:
                        expand *= 1.1
                        if log:    
                            log_center = np.log10(centre)
                            log_halfwidth = (np.log10(bin_right) - np.log10(bin_left)) / 2
                            log_left = log_center - log_halfwidth * expand
                            log_right = log_center + log_halfwidth * expand
                            # makes sure I am not going beyong the limits
                            # of my binning
                            left = max(10 ** log_left, tmin)
                            right = min(10 ** log_right, tmax)
                            
                        else:
                            # linear expansion
                            left = max(centre - (bin_right - bin_left) * expand / 2.0, tmin)
                            right = min(centre + (bin_right - bin_left) * expand / 2.0, tmax)      
                        bin_idx = np.where((time_lag >= left) & (time_lag < right))[0]
                    if len(bin_idx) >= min_bin_num:
                        time_bins.append(centre)
                        sf_binned.append(np.mean(sf_values[bin_idx]))
                        pair_counts.append(len(bin_idx))
                        irregular_bin.append(expand)
                        if sf_err is not None:
                            sf_bin_err.append(np.sqrt(np.sum(sf_err[bin_idx]**2)) /len(bin_idx))
                        else:
                            sf_bin_err.append(np.std(sf_values[bin_idx], ddof=1) / np.sqrt(len(bin_idx)))                  
    return np.array(sf_binned), np.array(time_bins), np.array(sf_bin_err), np.array(pair_counts), np.array(irregular_bin)

def model_function(x, t0, C0, C1, beta):
    """
    Model function for the structure function."""
    return C1 * (1 - np.exp(-(x / t0))**beta) + C0
