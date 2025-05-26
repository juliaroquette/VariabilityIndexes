"""
Module for computing variability timescales.

@juliaroquette:


TODO: 
- Modify LSP to use least-squares fitting to get the best period
- Debug SAVGOL
- Can I make an version of savgol where the window runs as a function of phase?

"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.optimize import minimize
from iminuit import Minuit
import warnings
import matplotlib.pyplot as plt


class TimeScale:
    def __init__(self, lc):
        from variability.lightcurve import LightCurve
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")        
        self.lc = lc
        
    def get_LSP_period(self,
                          fmin=1./300., 
                          fmax=1./0.5,
                          osf=10., 
                          periodogram=False, 
                          fap_prob=[0.001, 0.01, 0.1]):
        """
        Simple Period estimation code using Lomb-Scargle. 
        This adopts an heuristic approach for the frequency grid where,
        given the max/min values in the frequency, the grid is oversampled 
        by a default value of 5.
        

        Args:
            astropy.timeseries.LombScargle arguments:
            osf (int, optional): samples_per_peak Defaults to 5.
            fmin (int, optional): minimum frequency for the periodogram
            fmax (int, optional): maximum frequency for the periodogram
                NOTE: default min and max value consider:
                        - fmax is set by a 0.5 days period, which is about 
                        the break-up speed for very young stars. 
                        - fmin is arbitrary set to 250 days.
            periodogram (bool, optional): if True, returns the periodogram, 
                                          otherwise returns the period.

        Returns:
        if periodogram is True:
            frequency float: frequency of the highest peak
            power float: power of the highest peak
            FAP_level float: False alarm probability level for 1%, 10% and 40%
        else:
            frequency of the highest peak: float
            power of the highest peak: float
            FAP_highest_peak: 0-1. float: False Alarm Probability for the highest peak
        """
        # define the base for the Lomb-Scargle
        ls = LombScargle(self.lc.time, self.lc.mag)
        frequency, power = ls.autopower(samples_per_peak=osf,
                                        minimum_frequency=fmin,
                                        maximum_frequency=fmax) 
        # get False alarm probability levels
        self.FAP_level = ls.false_alarm_level(fap_prob, method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
        if bool(periodogram):
            return frequency, power, self.FAP_level
        else:
            highest_peak = power[np.argmax(power)]
            FAP_highest_peak = ls.false_alarm_probability(power.max(),method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
            return frequency[np.argmax(power)], highest_peak, FAP_highest_peak

    def get_structure_function_timescale(self):
        """
        Uses Chloé's implementation of the structure function to get a timescale
        """
        sf = StructureFunction(self.lc.time, self.lc.mag, self.lc.err)
        sf.structure_function_slow()
        print('SF timescale:', sf.find_timescale())
        return sf.find_timescale()
    
    def get_MSE_timescale(self):
        pass
    
    def get_CPD_timescale():
        pass


class StructureFunction:
    '''
    Class for computing the structure function of a light curve.
    Author: @ChloeMas
    Adapted to object-oriented by @juliaroquette
    '''    
    ###################################
    # Structure function
    def __init__(self, time, mag, err, **kwargs):
        self.mag = mag
        self.time = time
        self.err = np.mean(err)        
        self.num_bins = kwargs.get('num_bins', 120)
        self.epsilon = kwargs.get('epsilon', 5)
        # self.thresh = kwargs.get('thresh', 80)

        

    def structure_function_slow(self) :
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
        tmin, tmax = min(np.diff(self.time)), max(self.time) - min(self.time)

        
        # Create logarithmically spaced time bins
        log_bins  = np.logspace(np.log10(tmin), np.log10(tmax),\
            num=self.num_bins + 1, endpoint=True, base=10.0)
        self.time_bins = (log_bins[:-1] + log_bins[1:]) / 2.0
        # Ensure no zero time difference
        p = 10
        while tmin == 0:
            tmin = np.percentile(np.diff(self.time), p)
            p += 2    

    # Create logarithmically spaced time bins
        log_bins = np.logspace(np.log10(tmin), np.log10(tmax),\
            num=self.num_bins + 1)
        self.time_bins = (log_bins[:-1] + log_bins[1:]) / 2.0

        # Compute all time lags and corresponding delta magnitudes
        delta_t = np.abs(self.time[:, None] - self.time)
        delta_mag = (self.mag[:, None] - self.mag)**2
        delta_err = (self.err[:, None] - self.err)**2
        # Only upper triangular part
        delta_t = delta_t[np.triu_indices(len(self.time), 1)]  
        delta_mag = delta_mag[np.triu_indices(len(self.time), 1)]
        delta_err = delta_err[np.triu_indices(len(self.time), 1)]

        # Sort by time lag
        sort_indices = np.argsort(delta_t)
        delta_t, delta_mag = delta_t[sort_indices], delta_mag[sort_indices]
        delta_err = delta_err[sort_indices]

        # Initialize structure function array and pair counts
        self.sf = np.full(len(self.time_bins), np.nan)
        it_list = []

        # Adjust time bins to have at least epsilon pairs per time bin
        for i, time_bin in enumerate(self.time_bins):
            bin_diff = abs(time_bin - log_bins[i:i+2])
            lags_ind = np.where((delta_t > time_bin - bin_diff[0]) & (delta_t < time_bin + bin_diff[1]))[0]

            while len(lags_ind) < self.epsilon and len(lags_ind) > 0:
                bin_diff *= 0.95  # Narrow bin window
                lags_ind  = np.where((delta_t > time_bin - bin_diff[0]) & (delta_t < time_bin + bin_diff[1]))[0]

            if len(lags_ind) > 0:
                self.sf[i] = np.mean(delta_mag[lags_ind]) #- 2*np.mean(delta_err[lags_ind])**2
            
            it_list.append(len(lags_ind))

        # Remove NaN values and return the results
        valid_idx = ~np.isnan(self.sf)
        self.sf, self.time_bins = self.sf[valid_idx], self.time_bins[valid_idx]
        self.it_list = np.array(it_list)[valid_idx]    
    
    
    def model_function(x, t0, C0, C1, beta):
        return C1 * (1 - np.exp(-(x / t0))**beta) + C0

    
    def chi_squared(x, y, t0, C0, C1, beta):
        model = model_function(x, t0, C0, C1, beta)
        return (np.sum((y - model) ** 2/np.std(y)**2))

    def log_chi_squared(x, y, t0, C0, C1, beta):
        model     = model_function(x, t0, C0, C1, beta)
        log_y     = np.log10(y)
        log_model = np.log10(model)
        return np.sum((log_y - log_model) ** 2 / np.std(log_y) ** 2)    
    ################################################
    #Derive timescale    
        
    ########################################
    # Minuit Fit Function in Log Space
    def fit_with_minuit(self, last_params=None):

        if last_params is None:
            last_params = [1.0, 0.1, 0.1, 1.0]  # Initial guesses for parameters

        # Define a wrapped function for Minuit that only takes the parameters to be optimized
        def chi2_for_minuit(t0, C0, C1, beta):
            return log_chi_squared(self.time_bins, sf, t0, C0, C1, beta)

        # Initialize Minuit
        m = Minuit(chi2_for_minuit, t0=last_params[0], C0=last_params[1], C1=last_params[2], beta=last_params[3])
        m.errordef = 1  # For least squares fitting
        #m.limits = [(1e-6, max(time_bins)), (1e-6, np.percentile(sf, 20)), 
        #          (np.percentile(sf, 35), np.percentile(sf, 85)), (0.85, 1.15)]
        m.limits = [(0.07, max(self.time_bins)), (1e-5, 0.07), (1e-3, 5.5), (1, 1)]
                
        m.fixed["beta"] = True
        
        # Perform the minimization
        m.migrad()

        # Extract the fit results
        t0_fit, C0_fit, C1_fit, beta_fit = m.values
        fit_errors = m.errors
        chi2_min   = m.fval  # Get the minimum chi-squared value
        y_fit      = model_function(self.time_bins, t0_fit, C0_fit, C1_fit, beta_fit)

        return y_fit, (t0_fit, C0_fit, C1_fit, beta_fit), fit_errors, chi2_min

    #######################################
    # Main Function to Derive Timescale Using Log Space
    def find_timescale(sf, time_bins, threshold=1e-5, last_params=None, method='curve_fit', plot=False):
        """
        Derive the timescale from the structure function fit using the specified method in log space.
        
        Args:
            sf (ndarray): Structure function values.
            time_bins (ndarray): Corresponding time bins.
            threshold (float, optional): Gradient threshold to determine the timescale.
            last_params (list, optional): Initial parameters for fitting.
            method (str): Fitting method ('curve_fit' or 'minuit').
            plot (bool, optional): Whether to plot the structure function and fit results.

        Returns:
            ts (float): Derived timescale.
            y_fit (ndarray): Fitted structure function values.
            params (list): Best-fit parameters [t0, C0, C1, beta].
            fit_errors (list or None): Uncertainties for the fit parameters if available.
        """
        if method == 'curve_fit':
            y_fit, params, fit_errors = fit_with_curve_fit(sf, time_bins, last_params)
        elif method == 'minuit':
            y_fit, params, fit_errors, chi2 = fit_with_minuit(sf, time_bins, last_params)
        elif method == 'minimize_chi2':
            y_fit, params, fit_errors, chi2 = fit_with_minimize_chi2(sf, time_bins, last_params)
        else:
            raise ValueError("Method must be either 'curve_fit' or 'minuit'.")

        t0_fit, C0_fit, C1_fit, beta = params
        ts = min(t0_fit, max(time_bins))

        # Plot the structure function and fit results if requested
        if plot:
            plt.figure(figsize=(8, 5))
            plt.loglog(time_bins, sf, 'o', label="Structure Function")
            plt.loglog(time_bins, y_fit, '-', label="Fitted Model")
            plt.axvline(ts, color="red", linestyle="--", label=f"Timescale (t0): {ts:.2f}")
            plt.xlabel("Time Bins (log scale)")
            plt.ylabel("Structure Function")
            plt.legend()
            # plt.show()

        return ts, y_fit, params, fit_errors  # Return timescale, fit, params, and uncertainties

        