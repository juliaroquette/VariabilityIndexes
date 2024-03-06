"""
Module for computing variability timescales.

@juliaroquette: At the moment only the Lomb Scargle Periodogram is implemented. 
"""

import numpy as np
from astropy.timeseries import LombScargle


class TimeScale:
    def __init__(self, lc):
        from variability.lightcurve import LightCurve
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")        
        # super().__init__(time, mag, err, mask=None)
        self.lc = lc
        
    def get_LSP_period(self,
                          fmin=1./250., 
                          fmax=1./0.5,
                          osf=5., 
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
            type_flag: [1] if less than 1% of FAP - interpret as period
                       [0] if less than 10% - interpret as timescale
                       [-1] if more than 10% - interpret as probably spurious
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
            if highest_peak >= FAP_level[0]:
                type_flag = 1 # see it as a periodicity
            elif highest_peak >= FAP_level[1]:
                type_flag = 0 # see it as a timescale
            else:
                type_flag = -1 # see it as probably spurious
            return frequency[np.argmax(power)], highest_peak, type_flag
