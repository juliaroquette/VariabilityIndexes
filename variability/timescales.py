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
from scipy.spatial.distance import pdist
from iminuit import Minuit
import warnings
import matplotlib.pyplot as plt
from variability.lightcurve import LightCurve

class TimeScale(LightCurve):
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
        # deal with which timescale method to use
        # if method is SF or LSP, it will only return the timescale in the given method. Otherwise, it will attempt to get a timescale using LSP first, and then SF if no periodic timescale was found.
        if 'method' not in kwargs:
            warnings.warn("No 'method' specified. Using default 'auto' method.", UserWarning)
            kwargs['method'] = 'auto'
        elif kwargs['method'] not in ['LSP', 'SF', 'auto']:
            raise ValueError("Method must be 'LSP', 'SF' or 'auto'")
        method = kwargs['method'] 
        #
        fap = kwargs.get('fap_prob', 0.001)
        definition = kwargs.get('definition', 'auto')
        # if methods set to lomb scargle or to auto, first try to get 
        # timescales using the Lomb-Scargle periodogram
        if (method == 'LSP') or (method == 'auto'):
            best_freq, best_power, FAP_highest_peak = self.get_LSP_period(fmin=kwargs.get('fmin', 
                                                self.get_min_freq(definition=definition)), 
                                fmax=kwargs.get('fmax', 
                                                self.get_max_freq(definition=definition)), 
                                osf=kwargs.get('osf', 
                                                self.get_osf(definition=definition)), 
                                periodogram=kwargs.get('periodogram', False), 
                                fap_prob=fap, definition=definition)
    
            if (FAP_highest_peak < fap) or (method =="LSP"):
                self.ts = 1 / best_freq
                self.fap = FAP_highest_peak
                self.power = best_power
                self.method = 'LSP'
            else:
                self.ts = np.nan
                self.fap = np.nan
                self.power = np.nan
                self.method = ''
        # if method is set to SF, or if no timescale was obtained from LS
        # then proceed to get timescale from SF
        if kwargs['method'] == 'SF' or kwargs['method'] == 'auto' and not hasattr(self, 'ts'):
            self.ts, self.C0, self.C1 = self.get_structure_function_timescale()
            self.method = 'SF'

            
    def get_min_freq(self, definition='Gaia'):
        """
        Returns the minimum frequency based on different definitions.
        """
        if definition == 'Gaia':
            return 0.5 / (max(self.time) - min(self.time))
        elif definition == 'Chloe':
            return 1./ ((max(self.time) - min(self.time)))
        elif definition == 'auto':
            """
            Guarantees at least one full period cycle is covered
            """
            return 2 / (max(self.time) - min(self.time)/2)
        else:
            raise ValueError("Definition must be 'Gaia', 'auto' or 'Chloe'")
    
    def get_max_freq(self, definition='Gaia'):
        """
        Returns the maximum frequency based on different definitions.
        """
        if definition == 'Gaia':
            return 2.8
        elif definition == 'Chloe':
            return 1 / (np.median(np.diff(self.time)) * 2)
        elif definition == 'auto':
            return 1./ 0.5 / (np.median(np.diff(self.time)))
            
        else:
            raise ValueError("Definition must be 'Gaia', 'auto' or 'Chloe'")

    def get_osf(self, definition='Gaia'):
        """
        This factor is called samples_per_peak in Astropy's LombScargle
        and it defaults to 5. 
        Once fmin and fmax are defined, this is used to set the step in frequency for the .autopower method. 
        In Gaia, this is called `stepFrequencyScaleFactor` and had 
        recommended values of 5 in DR3 and 10 in DR4.
        """
        if definition == 'Gaia':
            return 10
        elif definition == 'Chloe':
            # Chloe defined that dynamically where she actually fiexes the df
            return 5
        elif definition == 'auto':
            return 5
        else:
            raise ValueError("Definition must be 'Gaia', 'auto', or 'Chloe'")        
        #   ts, y_fit, params, fit_errors = SF.find_timescale(np.array(sf), np.array(t_log), threshold=5e-5, plot=plot, method = 'minuit', last_params=[15, 0.02, 0.08, 1])

        # step_size = 1 / (scale_factor * (last_time - first_time))
        # scale_factor = 10.
                
    def get_LSP_period(self, 
                          fmin, 
                          fmax,
                          osf,
                          periodogram,
                          fap_prob,
                          definition='auto'):
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
        ls = LombScargle(self.time, self.mag, self.err)
        if definition != 'Chloe':
            frequency = np.arange(fmin, fmax, step=0.0002)
            power = ls.power(frequency, method='slow')
        else:
            frequency, power = ls.autopower(samples_per_peak=osf,
                                            minimum_frequency=fmin,
                                            maximum_frequency=fmax, 
                                            method='slow') 

        self.FAP_probs = ls.false_alarm_probability(power,method='baluev', 
                                            minimum_frequency=fmin, 
                                            maximum_frequency=fmax, 
                                            samples_per_peak=osf)
        
        if bool(periodogram):
            return frequency, power, self.FAP_probs
        else:
            freq_highest_peak = frequency[np.argmax(power)]
            power_highest_peak = power.max()
            # print('Highest peak:', pow)
            # print('Frequency of highest peak:', freq_highest_peak)
            FAP_highest_peak = ls.false_alarm_probability(power_highest_peak,method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
            # print(FAP_highest_peak)
            return freq_highest_peak, power_highest_peak, FAP_highest_peak

    def get_structure_function_timescale(self):
        """
        Uses Chloé's implementation of the structure function to get a timescale
        """
        sf = StructureFunction(lc=self.lc)
        sf.structure_function_slow()
        sf.find_timescale()
        # print('SF timescale:', sf.ts)
        return sf.ts, sf.C0, sf.C1
    
    def get_MSE_timescale(self):
        pass
    
    def get_CPD_timescale():
        pass


class StructureFunction(LightCurve):
    '''
    Class for computing the structure function of a light curve.
    Author: @ChloeMas
    Adapted to object-oriented by @juliaroquette
    '''    
    ###################################
    # Structure function
    def __init__(self, **kwargs):
        if 'lc' in kwargs:
            super().__init__(kwargs['lc'].time, kwargs['lc'].mag, kwargs['lc'].err)
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            super().__init__(kwargs['time'], kwargs['mag'], kwargs['err'], kwargs.get('mask', None))
            sorted_indices = np.argsort(self.time)
            self.time = self.time[sorted_indices]
            self.mag = self.mag[sorted_indices]
            self.err = self.err[sorted_indices]                
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")        

        # Sort all arrays by time
        self.num_bins = kwargs.get('num_bins', int(1.5*len(self.time)))
        self.epsilon = kwargs.get('epsilon', 3)
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
            print('Warning: tmin is zero, adjusting time bins')
            tmin = np.percentile(np.diff(self.time), p)
            p += 2    

        # Compute all time lags and corresponding delta magnitudes
        delta_t = np.abs(self.time[:, None] - self.time)
        delta_mag = (self.mag[:, None] - self.mag)**2
        # delta_err = (self.err[:, None] - self.err)**2
        # Only upper triangular part
        delta_t = delta_t[np.triu_indices(len(self.time), 1)]  
        delta_mag = delta_mag[np.triu_indices(len(self.time), 1)]
        # delta_err = delta_err[np.triu_indices(len(self.time), 1)]

        # Sort by time lag
        sort_indices = np.argsort(delta_t)
        delta_t, delta_mag = delta_t[sort_indices], delta_mag[sort_indices]
        # delta_err = delta_err[sort_indices]

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
    
    @staticmethod
    def model_function(x, t0, C0, C1, beta):
        return C1 * (1 - np.exp(-(x / t0))**beta) + C0

    @staticmethod
    def chi_squared(x, y, t0, C0, C1, beta):
        model = StructureFunction.model_function(x, t0, C0, C1, beta)
        return (np.sum((y - model) ** 2/np.std(y)**2))

    @staticmethod
    def log_chi_squared(x, y, t0, C0, C1, beta):
        model     = StructureFunction.model_function(x, t0, C0, C1, beta)
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
        @staticmethod
        def chi2_for_minuit(t0, C0, C1, beta):
            return self.log_chi_squared(self.time_bins, self.sf, t0, C0, C1, beta)

        # Initialize Minuit
        m = Minuit(chi2_for_minuit, t0=last_params[0], C0=last_params[1], C1=last_params[2], beta=last_params[3])
        m.errordef = 1  # For least squares fitting
        m.limits = [(0.07, max(self.time_bins)), (1e-5, 0.07), (1e-3, 5.5), (1, 1)]
                
        m.fixed["beta"] = True
        
        # Perform the minimization
        m.migrad()

        # Extract the fit results
        t0_fit, C0_fit, C1_fit, beta_fit = m.values
        fit_errors = m.errors
        chi2_min   = m.fval  # Get the minimum chi-squared value
        y_fit      = StructureFunction.model_function(self.time_bins, t0_fit, C0_fit, C1_fit, beta_fit)

        return y_fit, (t0_fit, C0_fit, C1_fit, beta_fit), fit_errors, chi2_min

    #######################################
    # Main Function to Derive Timescale Using Log Space
    def find_timescale(self, threshold=1e-5, last_params=None, plot=False):
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


        self.y_fit, self.params, self.fit_errors, self.chi2 = self.fit_with_minuit(last_params)


        t0_fit, C0_fit, C1_fit, beta = self.params
        self.C0 = C0_fit
        self.C1 = C1_fit
        self.ts = min(t0_fit, max(self.time_bins))


    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.loglog(self.time_bins, self.sf, 'o', label="Structure Function")
        plt.loglog(self.time_bins, self.y_fit, '-', label="Fitted Model")
        plt.axvline(self.ts, color="red", linestyle="--", label=f"Timescale (t0): {self.ts:.2f}")
        plt.xlabel("Time Bins (log scale)")
        plt.ylabel("Structure Function")
        plt.legend()
        # plt.show()

class StructureFunction_(LightCurve):
    '''
    Class for computing the structure function of a light curve.
    Adapted to object-oriented by @juliaroquette
    '''    
    ###################################
    # Structure function
    def __init__(self, **kwargs):
        if 'lc' in kwargs:
            super().__init__(kwargs['lc'].time, kwargs['lc'].mag, kwargs['lc'].err)
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            super().__init__(kwargs['time'], kwargs['mag'], kwargs['err'], kwargs.get('mask', None))
            sorted_indices = np.argsort(self.time)
            self.time = self.time[sorted_indices]
            self.mag = self.mag[sorted_indices]
            self.err = self.err[sorted_indices]                
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")        

        # Sort all arrays by time
        self.num_bins = kwargs.get('num_bins', int(1.5*len(self.time)))
        self.epsilon = kwargs.get('epsilon', 3)
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
        
        tmin, tmax = np.percentile(np.diff(self.time), 1), max(self.time) - min(self.time)
        # Create logarithmically spaced time bins
        log_bins  = np.logspace(np.log10(tmin), np.log10(tmax),\
            num=self.num_bins + 1, endpoint=True, base=10.0)
        #using geometric mean for center of log bins
        self.time_bins = np.sqrt(log_bins[:-1] * log_bins[1:])

        # Compute all time lags and corresponding delta magnitudes
        delta_t = pdist(self.time.reshape(-1, 1), metric='cityblock')  # |ti - tj|
        # Pairwise squared mag differences:
        def squared_diff(u, v): return (u - v)**2
        delta_mag = pdist(self.mag.reshape(-1, 1), metric=squared_diff)


        # Sort by time lag
        delta_mag = delta_mag[np.argsort(delta_t)]
        delta_t = np.sort(delta_t)

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
    
    @staticmethod
    def model_function(x, t0, C0, C1, beta):
        return C1 * (1 - np.exp(-(x / t0))**beta) + C0

    @staticmethod
    def chi_squared(x, y, t0, C0, C1, beta):
        model = StructureFunction.model_function(x, t0, C0, C1, beta)
        return (np.sum((y - model) ** 2/np.std(y)**2))

    @staticmethod
    def log_chi_squared(x, y, t0, C0, C1, beta):
        model     = StructureFunction.model_function(x, t0, C0, C1, beta)
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
        @staticmethod
        def chi2_for_minuit(t0, C0, C1, beta):
            return self.log_chi_squared(self.time_bins, self.sf, t0, C0, C1, beta)

        # Initialize Minuit
        m = Minuit(chi2_for_minuit, t0=last_params[0], C0=last_params[1], C1=last_params[2], beta=last_params[3])
        m.errordef = 1  # For least squares fitting
        m.limits = [(0.07, max(self.time_bins)), (1e-5, 0.07), (1e-3, 5.5), (1, 1)]
                
        m.fixed["beta"] = True
        
        # Perform the minimization
        m.migrad()

        # Extract the fit results
        t0_fit, C0_fit, C1_fit, beta_fit = m.values
        fit_errors = m.errors
        chi2_min   = m.fval  # Get the minimum chi-squared value
        y_fit      = StructureFunction.model_function(self.time_bins, t0_fit, C0_fit, C1_fit, beta_fit)

        return y_fit, (t0_fit, C0_fit, C1_fit, beta_fit), fit_errors, chi2_min

    #######################################
    # Main Function to Derive Timescale Using Log Space
    def find_timescale(self, threshold=1e-5, last_params=None, plot=False):
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


        self.y_fit, self.params, self.fit_errors, self.chi2 = self.fit_with_minuit(last_params)


        t0_fit, C0_fit, C1_fit, beta = self.params
        self.C0 = C0_fit
        self.C1 = C1_fit
        self.ts = min(t0_fit, max(self.time_bins))


    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.loglog(self.time_bins, self.sf, 'o', label="Structure Function")
        plt.loglog(self.time_bins, self.y_fit, '-', label="Fitted Model")
        plt.axvline(self.ts, color="red", linestyle="--", label=f"Timescale (t0): {self.ts:.2f}")
        plt.xlabel("Time Bins (log scale)")
        plt.ylabel("Structure Function")
        plt.legend()
        # plt.show()
