"""
Base class for reading light curves

@juliaroquette:
Defines a base class for reading light curves, which can be inherited by other classes that analyse it. 

TO DO:
Define a multi-wavelength light curve class
"""

import numpy as np

class LightCurve:
    """
    Class representing a light curve.

    Attributes:
        time (ndarray): Array of time values.
        mag (ndarray): Array of magnitude values.
        err (ndarray): Array of magnitude error values.
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
        
    def N(self):
        """
        Returns the number of datapoints in the light curve.

        Returns:
            int: Number of datapoints.
        """
        return len(self.mag)
    
    def time_span(self):
        """
        Returns the total time-span of the light-curve

        Returns:
            float: light-curve time-span.
        """
        return np.max(self.time) - np.min(self.time)
    
    def std(self):
        """
        Returns the standard deviation of the magnitude values.

        Returns:
            float: Standard deviation.
        """
        return np.std(self.mag)

    def mean(self):
        """
        Returns the mean of the magnitude values.

        Returns:
            float: Mean value.
        """
        return np.mean(self.mag)


    def weighted_average(self):
        """
        Returns the weighted average of the magnitude values.

        Returns:
            float: Weighted average.
        """
        return np.average(self.mag, weights=1./(self.err**2))

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
