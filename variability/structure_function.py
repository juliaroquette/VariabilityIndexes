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
                       log=True,
                       hybrid=False,
                       bin_min_size=5, 
                       max_bin_exp_factor=3.0, 
                       resolution=0.15):
    """
    Suite of adaptative binning methods.
    This bins data in either log/linear space, while ensuring
    that bins have at least bin_min_size SF data points.
    Parameters:
        time_lag:    array of SF time lags
        sf_values:   array of SF values (same length as time_lag)
        sf_err:      array of SF errors (same length as time_lag)
                     If None, error is estimated from sf_values
        scale:       'log' or 'linear'
        hybrid:      True/False
        bin_min_size:   limit number of pairs per bin. 
                    If hybrid=False: if the number of pairs in a bin 
                    is less than bin_min_size, the bin is iteratively 
                    expanded around its center in steps of 20% of the 
                    resolution until a maximum expansion factor 
                    (set by `max_bin_exp_factor) is reached. 
                    If hybrid=False, if the number of pairs in a bin is less
                    than bin_min_size, the few existent SF pairs are stored in 
                    a list for later processing.
        max_bin_exp_factor: maximum factor by which to expand bin width. 
                    It is clipped to 3.0, which is the case when the bin is merged to
                    its two immediate neighbours.
        resolution: Resolution for binning. If log=True, this is in dex.
                    If log=False, this is in linear units.
                    Recommended maximum value for log scale is 0.15 for log binning 
                    (guarantees a 5 days timescale is distinguishable from a 7 days one)
    Returns:
        sf_binned, time_bins, sf_bin_err, pair_counts, irregular_bin
        where:
        - sf_binned:   binned SF values
        - time_bins:   centers of the bins
        - sf_bin_err:  binned SF errors (or standard error of the mean if sf_err is None)
        - pair_counts: number of pairs in each bin
        - irregular_bin: flag indicating if the bin is irregular 
                         0.0 if regular, 
                         -1.0 if using individual pairs
                         >0.0 final expanded bin width used
    """

    # note that we need some limit in this max_bin_exp_factor
    # otherwise the bin expansion can completely mess
    # the resolutions of the binning
    # the limit set here is the case when the bin is merged 
    # with its two immediate neighbours
    max_bin_exp_factor = min(max_bin_exp_factor, 3.)

    # get bins and edges:
    bin_edges, centres  = bin_edges_and_centers(time_lag, resolution, log=log)
        
    sf_binned, time_bins, sf_bin_err, pair_counts, irregular_bin = [], [], [], [], []
    bin_idxs = np.digitize(time_lag, bin_edges, right=True) - 1  # subtract 1 for bin indexes starting at 0
    for i in range(len(bin_edges) - 1):

        # Get indices for current bin
        bin_idx = np.where(bin_idxs == i)[0]
        
        # test if there is at least one pair in the bin
        # otherwise discard it 
        if (len(bin_idx) > 0) :
            if len(bin_idx) >= bin_min_size:
                time_bins.append(centres[i])
                sf_binned.append(np.mean(sf_values[bin_idx]))
                pair_counts.append(len(bin_idx))
                irregular_bin.append(0.0) # no irregular flagging
                if sf_err is not None:
                    # error propagation
                    sf_bin_err.append(np.sqrt(np.sum(sf_err[bin_idx]**2)) / (len(bin_idx)))
                else:
                    # standard error of the mean
                    sf_bin_err.append(np.std(sf_values[bin_idx], ddof=1) / np.sqrt(len(bin_idx)))
            # if minimum number of pairs is not reached
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
                            # if no uncertainty is provided, use the MAD for the full dataset
                            sf_bin_err.append(1.4826*np.median(np.abs(sf_values-np.median(sf_values))))
                else:
                    # if hybrid binning is not chosen, attempt to
                    # expand bin until either it has enough pairs 
                    # or it reaches the max_bin_width allowed
                    new_width = resolution
                    step = 0.2 * resolution # use 20% of the resolution as an increment
                    found = False
                    while (new_width < max_bin_exp_factor * resolution):
                        # add increment of 20% the resolution
                        new_width += step
                        if log:
                            log_center = np.log10(centres[i])                       
                            log_left = log_center - new_width / 2.0
                            log_right = log_center + new_width / 2.0
                            # Update the data in the bin
                            bin_idx = np.where((time_lag > 10**log_left) & (time_lag < 10**log_right))[0]
                        else:
                            # linear expansion
                            left = centres[i] - new_width / 2.0
                            right = centres[i] + new_width / 2.0
                            bin_idx = np.where((time_lag >= left) & (time_lag < right))[0]
                        if (len(bin_idx) >= bin_min_size):
                            found = True
                            break
                    if found:
                        time_bins.append(centres[i])
                        sf_binned.append(np.mean(sf_values[bin_idx]))
                        pair_counts.append(len(bin_idx))
                        irregular_bin.append(new_width)
                        if sf_err is not None:
                            sf_bin_err.append(np.sqrt(np.sum(sf_err[bin_idx]**2)) /len(bin_idx))
                        else:
                            sf_bin_err.append(np.std(sf_values[bin_idx], ddof=1) / np.sqrt(len(bin_idx)))
    return np.array(sf_binned), np.array(time_bins), np.array(sf_bin_err), np.array(pair_counts), np.array(irregular_bin)


def bin_edges_and_centers(time_lag, 
                        resolution,
                        log=True):
    """Given an array of time lags and the 
    choice of a binning resolution, returns the edges 
    and centres of the bins in either linear or log space.
    Assumes that min(time_lag) and max(time_lag) should be
    at the bin centres.

    Args:
        time_lag: Structure Function time lags
        resolution: resolution for binning.
                    If log=True, this is in dex, 
                    where N = 1 + int((log10(max(time_lag))-log10(min(time_lag)))/resolution)
                    If log=False, this is in days, 
                    where N = 1 + int((max(time_lag)-min(time_lag))/resolution)
        log: True for binning in log space,
             False for binning in linear space.

    Returns:
        edges: array of bin edges
        centres: array of bin centres
        Note that len(edges) == len(centres) + 1
    """
    tmin = np.min(time_lag)
    tmax = np.max(time_lag)
    if log:
        log_tmin = np.log10(tmin)
        log_tmax = np.log10(tmax)
        # decides number of bins based on the resolution in dex
        n_bins = 1 + int(round((log_tmax - log_tmin) / resolution))
        # makes sure the edge bins are centred at max/min values
        centres_log = np.linspace(log_tmin, log_tmax, n_bins)
        centres = 10**centres_log
        # set the edges given the resolution and the correct bin centre
        edges = np.empty(n_bins + 1)
        # first edge
        edges[0] = 10**(centres_log[0] - resolution/2)
        # last edge
        edges[-1] = 10**(centres_log[-1] + resolution/2)
        # intermediate edges
        for i in range(1, n_bins):
            edges[i] = 10**((centres_log[i-1] + centres_log[i])/2)
    else:
        # same here but in linear space
        n_bins = 1 + int(round((tmax - tmin) / resolution))
        centres = np.linspace(tmin, tmax, n_bins)
        edges = np.empty(n_bins + 1)
        edges[0] = centres[0] - resolution/2
        edges[-1] = centres[-1] + resolution/2
        for i in range(1, n_bins):
            edges[i] = (centres[i-1] + centres[i]) / 2
    # len(edges) == len(centres) + 1
    return edges, centres         

def model_function(x, t0, C0, C1, beta):
    """
    Model function for the structure function."""
    return C1 * (1 - np.exp(-(x / t0))**beta) + C0
