"""
Module for computing variability timescales.
"""

import numpy as np
import pandas as pd

# from sklearn.gaussian_process         import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF


class TimeScale(lightcurve.LightCurve):
    def __init__(self):
        super().__init__()
        
    def get_LSP_period(self,
                          spp=10, 
                          nqf=10, 
                          fmin=1./62., 
                          fmax=1./0.1, periodogram=False, no_limit=False):
        """
        Simple Period estimation code using Lomb-Scargle

        Args:
            astropy.timeseries.LombScargle arguments:
            spp (int, optional): samples_per_peak Defaults to 10.
            nqf (int, optional): nyquist_factor Defaults to 10.
            fmin (int, optional): minimum frequency for the periodogram
            fmax (int, optional): maximum frequency for the periodogram
                NOTE: min and max value was calculated considering a generic fully
                populated light-curve for the CFHT 2017 campaign. 

        Returns:
            frequency float: frequency of the highest peak
            power float: power of the highest peak
        """
        ls = LombScargle(self.time, self.mag)        
        if bool(no_limit):
            frequency, power = ls.autopower()
        else: 
            frequency, power = ls.autopower(samples_per_peak=spp,
                                        nyquist_factor=nqf,
                                        minimum_frequency=fmin,
                                        maximum_frequency=fmax) 
        if bool(periodogram):
            return frequency, power
        else:
            return frequency[np.argmax(power)], power[np.argmax(power)]

    def get_structure_function_timescale(self):
        pass
    
    def get_MSE_timescale(self):
        pass
    
    def get_CPD_timescale():
        pass

    def get_gaussian_timescale(self):
        pass
    #     """
    #     Computes the timescale of variability using Gaussian process regression.

    #     Returns:
    #         float: The timescale of variability.
        
    #     @Chloé
    #     """

    #     kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    #     gp     = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    #     self.time_gs = self.time_red.reshape(-1, 1)

    #     gp.fit(self.time_gs, self.mag_red)
    #     log_marginal_likelihood = gp.score(self.time_gs, self.mag_red)
    #     timescale = gp.kernel_.length_scale

    #     return timescale
    
    
    def fold(self, tau):
        """
        Folds the light curve in phase for a given period. 
        Returns magnitudes sorted by phase.

        Parameters:
        - mag: array-like, magnitudes (or uncertainties) of the light curve
        - tau: array-like, time values of the light curve
        - period: float, period of the light curve

        Returns:
        - phase: array-like, phase values of the folded light curve
        - mag_sorted: array-like, magnitudes of the folded light curve sorted by phase
        
        """
        # Calculate the phase values
        phase = np.mod(self.time, tau) / tau
        
        # Sort the phase and magnitude arrays based on phase values
        sort = np.argsort(self.phase)
        self.phase = phase[sort]
        self.mag_sorted = self.mag[sort]