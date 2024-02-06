"""
Base class for reading light curves

@juliaroquette:
Defines a base class for reading light curves, which can be inherited by other classes that analyse it.

Last update: 31 January 2024 

TO DO:
Define a multi-wavelength light curve class
Define a class/function that generates a sample of light-curves
"""

import numpy as np

class LightCurve:
    """
    Class representing a light curve.

    Attributes:
        time (ndarray): Array of time values.
        mag (ndarray): Array of magnitude values.
        err (ndarray): Array of magnitude error values.
        mask (ndarray): Array of boolean values indicating valid datapoints. 
                        if no mask is provided, all finite values existing
                        in time, mag and err are considered valid.
    
    Properties:
        - N (int): Number of datapoints in the light curve.
        - time_span (float): Total time-span of the light curve.
        - std (float): Standard deviation of the magnitude values.
        - mean (float): Mean of the magnitude values.
        - weighted_average (float): Weighted average of the magnitude values.
        - median (float): Median of the magnitude values.
    """

    def __init__(self, time, mag, err, mask=None):
        """
        Initializes a LightCurve object.

        Args:
            time (ndarray): Array of time values.
            mag (ndarray): Array of magnitude values.
            err (ndarray): Array of error values.
            mask (ndarray, optional): Array of boolean values indicating which data points to include. Defaults to None.
        """
        if bool(mask):
            mask = np.asarray(mask, dtype=bool)
        else:
            mask = np.where(np.all(np.isfinite([mag, time, err]), axis=0))[0]
        self.time = np.asarray(time, dtype=float)[mask]
        self.mag = np.asarray(mag, dtype=float)[mask]
        self.err = np.asarray(err, dtype=float)[mask]
    
    @property    
    def N(self):
        """
        Returns the number of datapoints in the light curve.

        Returns:
            int: Number of datapoints.
        """
        return len(self.mag)
    
    @property
    def time_span(self):
        """
        Returns the total time-span of the light-curve

        Returns:
            float: light-curve time-span.
        """
        return np.max(self.time) - np.min(self.time)
    
    @property
    def std(self):
        """
        Returns the standard deviation of the magnitude values.

        Returns:
            float: Standard deviation.
        """
        return np.std(self.mag)

    @property
    def mean(self):
        """
        Returns the mean of the magnitude values.

        Returns:
            float: Mean value.
        """
        return self.mag.mean()
    
    @property
    def max(self):
        """
        Returns the max value of the magnitudes.

        Returns:
            float: mags value.
        """
        return self.mag.max()
    
    @property
    def min(self):
        """
        Returns the min value of the magnitudes.

        Returns:
            float: max mags value.
        """
        return self.mag.min()
    
    @property
    def time_max(self):
        """
        Returns the max value of the observations times.

        Returns:
            float: max time value.
        """
        return self.time.max()
    
    @property
    def time_min(self):
        """
        Returns the min value of the observations times.

        Returns:
            float: min time value.
        """
        return self.time.min()

    @property
    def weighted_average(self):
        """
        Returns the weighted average of the magnitude values.

        Returns:
            float: Weighted average.
        """
        return np.average(self.mag, weights=1./(self.err**2))

    @property
    def median(self):
        """
        Returns the median of the magnitude values.

        Returns:
            float: Median value.
        """
        return np.median(self.mag)
      

class FoldedLightCurve(LightCurve):
    """
    Represents a folded light curve with time values folded for a given timescale.

    Args:
        time (array-like): The time values of the light curve.
        mag (array-like): The magnitude values of the light curve.
        err (array-like): The error values of the light curve.
        timescale (float): The timescale used for folding the light curve.
        mask (array-like, optional): The mask to apply to the light curve.

    Attributes:
        timescale (float): The timescale used for folding the light curve.
        phase (array-like): The phase values of the folded light curve.
        mag_pahsed (array-like): The magnitude values of the folded light curve, sorted based on phase.
        err_pahsed (array-like): The error values of the folded light curve, sorted based on phase.
    """

    def __init__(self, time, mag, err, timescale, mask=None):
        super().__init__(time, mag, err, mask=mask)
        self.timescale = timescale 
        # Calculate the phase values
        phase = np.mod(self.time, self.timescale) / self.timescale
        # Sort the phase and magnitude arrays based on phase values
        sort = np.argsort(phase)
        self.phase = phase[sort]
        self.mag_pahsed = self.mag[sort]       
        self.err_pahsed = self.err[sort]

class SyntheticLightCurve:
    """
    A class to generate synthetic light curves.
    
    Has the same structure as a LightCurve object. 
    
    TODO:
    - Create a series of functions to generate different types of observational windows
    """
    def __init__(self, 
                 mean_magnitude = 15.0, 
                 noise_level=0., # Standard deviation of Gaussian noise
                 mean_error=0.01, # Mean error of the light curve
                 **kargs):
        #time=time, N=N, ptp=0.1, seed=None, mean=0.0, e_std=1.0, ):
        if 'time' in kargs.keys():
            mask = np.where(np.isfinite(kargs['time']))[0]
            self.time = np.asarray(kargs['time'], dtype=float)[mask]
            self.n_epochs = len(self.time)
        elif 'n_epochs' in kargs.keys():
            self.time = np.linspace(0, 100, kargs['n_epochs'])
        else:
            raise ValueError('Either time or N must be provided')
        
        self.noisy_mag = mean_magnitude + np.random.normal(scale=noise_level, size=n_epochs)
        self.mag_err = abs(np.random.normal(loc=mean_error, scale=noise_level, size=n_epochs))
        
    def periodic(self, ptp_amp = 1, period=2., phi=0.):
        return self.noisy_mag + 0.5*ptp_amp * np.sin(2 * np.pi * self.time / period + phi)

    
    def quasiperiodic(self):
        pass
    
    def eclipsing_binary(self):
        pass
    
    def periodic_dipper(self):
        pass
    
    def quasiperiodic_dipper(self):
        pass
    
    def aperiodic_dippers(self):
        pass
    
    def periodic_bursting(self):
        pass
    
    def quasiperiodic_bursting(self):
        pass
    
    def aperiodic_bursting(self):
        pass
    
    