"""
Colection of fitlers

This module provides a collection of filters for processing light curves.
It includes methods for filtering, masking, and waveform analysis.

Classes:
- Filtering: A class for applying filters to a light curve.
- WaveForm: A class for analyzing the waveform of a folded light curve.

Functions:
- uneven_savgol: Applies a Savitzky-Golay filter to non-uniformly spaced data.

@juliaroquette
Last update: 19 Feb 2024
"""

import warnings
import numpy as np

class Filtering:
    def __init__(self, lc):
        """
        Initialize the Filtering class.

        Parameters:
        - lc (LightCurve): The light curve to be filtered.

        Raises:
        - TypeError: If lc is not an instance of LightCurve.
        """
        from variability.lightcurve import LightCurve
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc

    def filter(self, method=None, **kwargs):
        """
        Apply a filter to the light curve.

        Parameters:
        - method (str): The filter method to be applied.
        - **kwargs: Additional keyword arguments specific to each filter method.

        Returns:
        - np.array: The filtered light curve.

        Raises:
        - ValueError: If the specified filter method is not implemented.
        """
        if method == 'uneven_savgol':
            polynom = kwargs.get('polynom', 3)
            window = kwargs.get('window', 4)
            return self.uneven_savgol(window=window, polynom=polynom)     
        elif method == 'smooth_per_timescale':
            window_days = kwargs.get('window_days', 10.)
            return self.smooth_per_timescale(window_days=window_days)
        else:
            raise ValueError("Method _{0}_ not implemented.".format(method))
    
    def sigma_clip(self, sigma=5):
        """
        Remove outliers from the light curve using sigma clipping.

        Parameters:
        - sigma (float): The number of standard deviations to use for clipping.

        Returns:
        - np.array: A mask for values to be kept after filtering.
        """
        return np.abs(self.lc.mag - self.lc.median) <= sigma * self.lc.std

    def uneven_savgol(self, window, polynom):
        """
        Apply a Savitzky-Golay filter to the light curve with non-uniform spacing.

        Parameters:
        - window (int): Window length of datapoints. Must be odd and smaller than the light curve size.
        - polynom (int): The order of polynomial used. Must be smaller than the window size.

        Returns:
        - np.array: The smoothed light curve.

        Raises:
        - ValueError: If the data size is smaller than the window size or if the window is not an odd integer.
        - TypeError: If window or polynom are not integers.
        """
        x = self.lc.time
        y = self.lc.mag
        return uneven_savgol(x, y, window, polynom)
    
    def smooth_per_timescale(self, window_days=10.):
        """
        Smooth the light curve using a window size in days.

        Parameters:
        - window_days (float): The window size in days.

        Returns:
        - np.array: The smoothed light curve.
        """
        smoothed_values = np.zeros_like(self.lc.mag, dtype=float)
        for i in range(len(self.lc.time)):
            start_time = self.lc.time[i] - window_days/2
            end_time = self.lc.time[i] + window_days/2
            select = np.where((self.lc.time >= start_time) & (self.lc.time <= end_time))
            smoothed_values[i] = np.mean(self.lc.mag[select])
        return smoothed_values

class WaveForm:
    def __init__(self, phase, mag_phased):
        """
        Initialize the WaveForm class.

        Parameters:
        - phase (np.array): The phase of the folded light curve.
        - mag_phased (np.array): The magnitude of the folded light curve.

        """
        self.phase = phase
        self.mag_phased = mag_phased
        self.N = len(self.mag_phased)
  
    def circular_rolling_average_number(self, window_size=5):
        """
        Calculate the circular rolling average of the folded light curve.

        Parameters:
        - window_size (int): The window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform.
        """
        extended_mag = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
        extended_waveform = np.array([
            np.mean(extended_mag[i - int(0.5 * window_size):
                i + int(np.round((0.5 * window_size)))])\
                    for i in np.arange(int(0.5 * window_size),\
                        len(extended_mag) -\
                            int(np.round(0.5*window_size + 1)))])
        
        return extended_waveform[self.N - int(0.5 * window_size):
            2 * self.N - int(0.5 * window_size)]


    def circular_rolling_average_phase(self, wd_phase=0.1):
        """
        Calculate the circular rolling average of the folded light curve based on phase.

        Parameters:
        - wd_phase (float): The phase window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform based on phase.
        """
        waveform = np.full(len(self.phase), np.nan)
        extended_phase = np.concatenate((self.phase - 1, self.phase, 1 + self.phase))
        extended_mag = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
        for i, p in enumerate(self.phase):
            select = np.where((extended_phase <= p + wd_phase/2.) & (extended_phase > p - wd_phase/2.))[0]
            waveform[i] = np.mean(extended_mag[select])
        return waveform

    def uneven_savgol(self, window, polynom):
        """
        Apply a Savitzky-Golay filter to the folded light curve with non-uniform spacing.

        Parameters:
        - window (int): Window length of datapoints. Must be odd and smaller than the light curve size.
        - polynom (int): The order of polynomial used. Must be smaller than the window size.

        Returns:
        - np.array: The smoothed folded light curve.

        Raises:
        - ValueError: If the data size is smaller than the window size or if the window is not an odd integer.
        - TypeError: If window or polynom are not integers.
        """
        x = np.concatenate((self.phase - 1, self.phase, 1 + self.phase))
        y = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased)) 
        return  uneven_savgol(x, y, window, polynom)  [self.N:2*self.N]

    def get_waveform(self, waveform_type='uneven_savgol', waveform_params={}):
        if waveform_type == 'circular_rolling_average_phase':
            wd_phase = waveform_params.get('wd_phase', 0.1)
            waveform = self.circular_rolling_average_phase(wd_phase=wd_phase)
        elif waveform_type == 'circular_rolling_average_number':
            window_size = waveform_params.get('window_size', 0.1*self.N)
            waveform = self.circular_rolling_average_number(window_size=window_size)
        elif waveform_type == 'uneven_savgol':
            window = waveform_params.get('window', round(0.25*self.N))
            if window % 2 == 0:
                window += 1
            polynom = waveform_params.get('polynom', 3)
            waveform = self.uneven_savgol(window, polynom)
        else:
            raise ValueError("Method _{0}_ not implemented.".format(self._waveform_type))
        return waveform
    
    def residual_magnitude(self, waveform):
        """
        Calculate the residual magnitude after waveform subtraction.
        """
        return self.mag_phased - waveform

def uneven_savgol(x, y, window, polynom):
    """
    Applies a Savitzky-Golay filter to y with non-uniform spacing
    as defined in x

    This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do

    Parameters
    ----------
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size

    Returns
    -------
    np.array of float
        The smoothed y values
    """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')
    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1
    
    # Initialize variables
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Calculate powers of t
    powers_of_t = np.empty((window, polynom))
    for j in range(window):
        powers_of_t[j] = np.power(x[half_window + j] - x[half_window], np.arange(polynom))

    # Start smoothing
    for i in range(half_window, len(x) - half_window):
        # Calculate t vector
        t = x[i - half_window:i + half_window + 1] - x[i]

        # Create the initial matrix A and its transposed form tA
        A = np.vander(t, polynom, increasing=True)
        tA = A.T

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA_inv = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA_inv, tA)

        # Calculate smoothed y value
        y_smoothed[i] = np.dot(coeffs[0], y[i - half_window:i + half_window + 1])

    return y_smoothed

