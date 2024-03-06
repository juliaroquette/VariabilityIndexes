"""
Base class for reading light curves

@juliaroquette:
Defines a base class for reading light curves, which can be inherited by other classes that analyse it.

Last update: 19 Feb 2024

"""

import numpy as np
import warnings
np.random.seed(42)

class LightCurve:
    """
    Class representing a light curve.

    Attributes:
        time (ndarray): Array of time values.
        mag (ndarray): Array of magnitude values.
        err (ndarray): Array of error values.
        mask (ndarray, optional): Array of boolean values indicating which data points to include. Defaults to None.
    """

    def __init__(self,
                 time,
                 mag,
                 err,
                 mask=None):
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
        Returns the number of data points in the light curve.

        Returns:
            int: Number of data points.
        """
        return len(self.mag)
    
    @property
    def time_span(self):
        """
        Returns the total time span of the light curve.

        Returns:
            float: Light curve time span.
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
        Returns the maximum value of the magnitudes.

        Returns:
            float: Maximum magnitude value.
        """
        return self.mag.max()
    
    @property
    def min(self):
        """
        Returns the minimum value of the magnitudes.

        Returns:
            float: Minimum magnitude value.
        """
        return self.mag.min()
    
    @property
    def time_max(self):
        """
        Returns the maximum value of the observation times.

        Returns:
            float: Maximum time value.
        """
        return self.time.max()
    
    @property
    def time_min(self):
        """
        Returns the minimum value of the observation times.

        Returns:
            float: Minimum time value.
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

    def __init__(self,
                 time=None,
                 mag=None,
                 err=None,
                 timescale=None,
                 lc=None,
                 mask=None):
        if lc is not None:
            time = lc.time
            mag = lc.mag
            err = lc.err
        assert timescale is not None, "A timescale must be provided"
        super().__init__(time, mag, err, mask=mask)
        self.timescale = timescale 
        # Calculate the phase values
        phase = np.mod(self.time, self.timescale) / self.timescale
        # Sort the phase and magnitude arrays based on phase values
        sort = np.argsort(phase)
        self.phase = phase[sort]
        self.mag_phased = self.mag[sort]       
        self.err_phased = self.err[sort]

    def get_waveform(self, **kwargs):
        from variability.filtering import WaveForm
        if 'waveform_type' in kwargs.keys():
            waveform_type = kwargs['waveform_type']
        else:
            waveform_type = 'uneven_savgol'
            warnings.warn('No waveform type provided, using default value of {0}'.format(waveform_type))
        return WaveForm(self, 
                        waveform_type=waveform_type).get_waveform(**kwargs)