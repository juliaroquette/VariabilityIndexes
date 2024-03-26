"""
Module for computing variability timescales.

@juliaroquette: At the moment only the LombScargle periodogram is implemented.
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
                          fmin=1./100., 
                          fmax=1./0.5,
                          osf=10., 
                          periodogram=False, 
                          fap_prob=[0.001, 0.01, 0.1]):
        """
        Simple Period estimation code using Lomb-Scargle. 
        This adopts an heuristic approach for the frequency grid where,
        given the max/min values in the frequency, the grid is oversampled 
        by a default value of 10 (this is the recommendation for Gaia DR4).
        

        Args:
            astropy.timeseries.LombScargle arguments:
            osf (int, optional): samples_per_peak Defaults to 5.
            fmin (int, optional): minimum frequency for the periodogram
            fmax (int, optional): maximum frequency for the periodogram
                NOTE: default min and max value consider:
                        - fmax is set by a 0.5 days period, which is about 
                        the break-up speed for very young stars. 
                        - fmin is arbitrary set to 100 days.
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
        frequency, power = ls.autopower(method='slow', #should be similar to the VariPipe
                                        samples_per_peak=osf,
                                        minimum_frequency=fmin,
                                        maximum_frequency=fmax) 
        # get False alarm probability levels
        FAP_level = ls.false_alarm_level(fap_prob, 
                                         method='baluev', #same method in VariPipe
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
        if bool(periodogram):
            return frequency, power, FAP_level
        else:
            highest_peak = power[np.argmax(power)]
            FAP_highest_peak = ls.false_alarm_probability(power.max(),
                                                          method='baluev', 
                                         minimum_frequency=fmin, 
                                         maximum_frequency=fmax, 
                                         samples_per_peak=osf)
            return frequency[np.argmax(power)], highest_peak, FAP_highest_peak

    def get_structure_function_timescale(self):
        raise NotImplementedError("This hasn't been implemented yet.")

    