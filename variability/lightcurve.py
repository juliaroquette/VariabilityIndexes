"""
Base class for reading light curves

@juliaroquette:
Defines a base class for reading light curves, which can be inherited by other classes that analyse it.

Last update: 19 Feb 2024

"""

import numpy as np
import warnings
np.random.seed(42)
from variability.filtering import WaveForm
 
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
    """
    def __init__(self,
                 timescale=None,
                 **kwargs):
        
        # makes sure this is also a LightCurve object
        if 'lc' in kwargs:
            super().__init__(kwargs['lc'].time, kwargs['lc'].mag, kwargs['lc'].err)
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            super().__init__(kwargs['time'], kwargs['mag'], kwargs['err'], kwargs.get('mask', None))
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")
        
        # FlodedLightCurve needs a timescale
        if timescale is not None:
            self._timescale = timescale
        else:
            raise NotImplementedError("Automatic timescale derivation not implemented, please provide timescale as input")

        # phasefold lightcurve to a given timescale
        self._get_phased_values()        
                
        # phasefold lightcurves have a waveform
        # define a WaveForm Object
        self.wf = WaveForm(self.phase, self.mag_phased)
        #  check if specific window parameters were passed as input
        self._waveform_params = kwargs.get('waveform_params', {'window': round(.25*self.N),
                                                    'polynom': 3})
        # check if a specific waveform type was passed as input
        self._waveform_type = kwargs.get('waveform_type', 'uneven_savgol')
        self._get_waveform()

    def _get_phased_values(self):
        # Calculate the phase values
        phase = np.mod(self.time, self._timescale) / self._timescale
        # Sort the phase and magnitude arrays based on phase values
        sort = np.argsort(phase)
        self.phase = phase[sort]
        self.mag_phased = self.mag[sort]       
        self.err_phased = self.err[sort]
        
    @property
    def timescale(self):
        return self._timescale
    
    @timescale.setter
    def timescale(self, new_timescale):
        if new_timescale > 0.:
            self._timescale = new_timescale
            # update phase-folded values
            self._get_phased_values()
            # update the waveform for the new timescale
            self.wf = WaveForm(self.phase, self.mag_phased)
            self._get_waveform()
        else:
            raise ValueError("Please enter a valid _positive_ timescale")        
        
    def _get_waveform(self, **kwargs):
        self.waveform = self.wf.get_waveform(waveform_type= kwargs.get('waveform_type', self._waveform_type), 
                                             waveform_params=kwargs.get('waveform_params', self._waveform_params))
        # phasefolded lightcurves also have a residual curve between the waveform and the lightcurve
        self.residual = self.wf.residual_magnitude(self.waveform)