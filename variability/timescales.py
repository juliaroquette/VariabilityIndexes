"""
Module for computing variability timescales.

@juliaroquette: At the moment
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import warnings


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
        FAP_level = ls.false_alarm_level(fap_prob, method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
        if bool(periodogram):
            return frequency, power, FAP_level
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
    
    def get_MSE_timescale(self):
        pass
    
    def get_CPD_timescale():
        pass


class StructureFunction:
    '''
    This package computes the Structure function as implemented by 
    Sergison+19 and Venuti+21, in order to apply it to GaiaDR3 data
    Author: @ChloeMas
    '''    
    ###################################
    # Structure function
    def __init__(self, time, mag, err, **kwargs):
        self.mag = mag
        self.time = time
        self.err = np.mean(err)        
        self.num_bins = kwargs.get('num_bins', 100)
        self.epsilon = kwargs.get('epsilon', 4)
        self.thresh = kwargs.get('thresh', 75)

        

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
        log_bins  = np.logspace(np.log10(tmin), np.log10(tmax), num=self.num_bins + 1, endpoint=True, base=10.0)
        time_bins = (log_bins[:-1] + log_bins[1:]) / 2.0
    

        sf       = np.zeros(len(time_bins))
        it_list  = []

        #Compute all time lags, corresponding delta_mags
        delta_t_  = []
        delta_mag = []

        for i in range(len(self.time)):
            for j in range(i+1, len(self.time)):
                dt = abs(self.time[i] - self.time[j])
                delta_t_.append(dt)
                delta_mag.append((self.mag[i] - self.mag[j])**2)
            
        delta_t_  = np.array(delta_t_)
        delta_mag = np.array(delta_mag)
        
        sort      = np.argsort(delta_t_)
        delta_t_, delta_mag = delta_t_[sort], delta_mag[sort]

        
        #Loop to adjust the time bins to have at least epsilon pairs per time bin :
        for i, time_bin in enumerate(time_bins) :
            
            bin_diff_d = abs(time_bin - log_bins[i])
            bin_diff_u = abs(time_bin - log_bins[i+1])

            lags_ind   = np.where((delta_t_>time_bin-bin_diff_d) & (delta_t_<time_bin+bin_diff_u))[0]
            
            while len(lags_ind) < self.epsilon :
            
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
        self.SF = sf[sel]
        self.time_lag = np.array(time_bins)[sel]
        self.N_bin = np.array(it_list)[sel]
    
    
    ################################################
    #Derive timescale    
        
    def find_timescale(self):

        """
        Finds the characteristic timescale of a time-series from the Structure Function.

    #     Args:
    #         - sf, self.time_lag (ndarray): Structure function and log time bins (structure function must be in mag**2)
    #         - err (float):         Error threshold. Defaults to 0.01. Used to define the noise-dominated regime of the sf
    #         - thresh (int):        Thresh_th percentile is used for a first approximation of tau_peak (timescale where most variability occurs). Default is Q3           

    #     Returns:
    #         - ts_sf (float):       The characteristic timescale. For a periodic time series, should be 1/2 the period
    #         - dict_fit (dict):     Fitting parameters for each regime of the SF
    #         - flag(bool):          True is no ts was retrieved with the code (in which case ts_sf will be NaN)
    #     """
        
        #Divide the SF into 3 regimes
        taus = [min(self.time_lag),\
                self.time_lag[np.where(self.SF >= 2 * self.err**2)[0][0]],\
                self.time_lag[np.where(self.SF >= np.percentile(self.SF, self.thresh))[0][0]],\
                max(self.time_lag)]

        sfs, ts = zip(*[(self.SF[(self.time_lag >= taus[i]) & (self.time_lag <= taus[i+1])],\
              self.time_lag[(self.time_lag >= taus[i]) & (self.time_lag <= taus[i+1])]) for i in range(len(taus)-1)])

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
        peaks   = find_peaks(self.SF)[0]
        t_peaks = self.time_lag[peaks]
        sel     = np.where((t_peaks >= tau_eq))[0]
        peaks   = t_peaks[sel]
        flag    = False
        
        if len(peaks) > 0 :
            ts_sf   = peaks[0]
        else :
            print('no ts found')
            ts_sf = np.nan
            flag  = True                
        self.dict_fit = dict_fit
        return 2.*ts_sf
