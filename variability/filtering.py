"""
Colection of fitlers
"""

import warnings
import numpy as np
import scipy as sp
import pandas as pd
from astropy.convolution import Box1DKernel, convolve
from variability.lightcurve import LightCurve, FoldedLightCurve


class Filtering:
    def __init__(self, lc):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self.even = self.is_evenly_spaced()
        if not self.even:
            warnings.warn("Time series may not be evenly spaced.", UserWarning)

    def filter(self, method=None, **kargs):
        """
        Apply a filter of choice to detrend light-curve.

        Parameters:
        - method (str): The filtering method to be applied. Available options are:
            - 'savgol': Apply the Savitzky-Golay filter to remove long-term trends.
                - **kargs:
                    - timescale (int): The length of the filter window.
                    - polynom (int): The order of the polynomial used to fit the samples.
            - 'Cody': Apply the Cody filter to remove long-term trends.
                - **kargs:
                    - window_length (int): The length of the filter window.
                    - polyorder (int): The order of the polynomial used to fit the samples.
            - 'rolling_average': Apply the rolling average filter to remove long-term trends.
                - **kargs:
                    - window_size (int): The size of the moving window.
            - 'uneven_savgol': Apply the uneven Savitzky-Golay filter to remove long-term trends.
                - **kargs:
                    - breakpoints (list): A list of breakpoints where the filter window changes.
                    - window_length (int): The length of the filter window for each segment.
                    - polyorder (int): The order of the polynomial used to fit the samples for each segment.
            - 'smooth_per_timescale': Apply the smooth per timescale filter to remove long-term trends.
                - **kargs:
                    - timescale (float): The timescale parameter for the filter.

        Returns:
        - The filtered light-curve.

        Raises:
        - ValueError: If the specified filtering method is not implemented.
        """
        if method == 'savgol':
            return self.savgol_longtrend(**kargs)
        elif method == 'Cody':
            return self.Cody_long_trend(**kargs)
        elif method == 'rolling_average':
            return self.rolling_average(**kargs)
        elif method == 'uneven_savgol':
            return self.uneven_savgol(**kargs)
        elif method == 'smooth_per_timescale':
            return self.smooth_per_timescale(**kargs)
        else:
            raise ValueError("Method _{0}_ not implemented.".format(method))
    
    def is_evenly_spaced(self, tolerance=0.1):
        """
        tests if a function is close to evenly spaced.
        The default tolerance check if less than 0.01% of t
        """
        dt = self.lc.time[1:] - self.lc.time[:-1]
        num_outliers = len(np.where(abs(dt - np.mean(dt)) > 3*np.std(dt))[0])
        return bool(num_outliers < (tolerance/100.)*len(self.lc.time))

    #masking
    def sigma_clip(self, sigma=5):
        """
        remove outliers from the light-curve using a sigma clipping
        returns as mask for values to be kept after filtering
        """
        return np.abs(self.lc.mag - self.lc.median) <= sigma * self.lc.std

    def get_num_time(self, timescale=10.):
        """
        Assuming the data is in days, this 
        function conts how many points you need 
        to cover a given timescale
        """
        if not self.even:
            warnings.warn("Time series may not be evenly spaced - no direct conversion between number of points and timescale.", UserWarning)
        time = self.lc.time - np.min(self.lc.time)
        return len(time[time < timescale])
    
    #detrending
    def Cody_long_trend(self, timescale=10.):
        """
        Code used in Cody et al. 2018,2020
        window=715 is appropriate for K2 data and equivalent to 10days
        Source:
        """
        window = self.get_num_time(timescale)
        return scipy.ndimage.median_filter(self.lc.mag, size=window, mode='nearest')
    
        
    def savgol_longtrend(self, timescale=10., polynom=3) :
        """
        To be changed to timescale rather than window size
        """
        if not self.even:
            warnings.warn("SavGol Scipy expects even time series - which may not be true here.", UserWarning)        
        window = self.get_num_time(timescale)
        return sp.signal.savgol_filter(self.lc.mag, window, polynom)
    
    def uneven_savgol(self, window, polynom):
        x = self.lc.time
        y = self.lc.mag
        return uneven_savgol(x, y, window, polynom)
    
    def smooth_per_timescale(self, window_days=10.):
        """
        for given light-curve with a time and mag, 
        use a window wd in days to smooth the light-curve
        then remove the smoothed curve form the 
        """
        smoothed_values = np.zeros_like(self.lc.mag, dtype=float)
        for i in range(len(self.lc.time)):
            start_time = self.lc.time[i] - window_days/2
            end_time = self.lc.time[i] + window_days/2
            select = np.where((self.lc.time >= start_time) & (self.lc.time <= end_time))
            smoothed_values[i] = np.mean(self.lc.mag[select])
        return smoothed_values

    def rolling_average(self, window=5):
        """
        Uses
        """
        if not self.even:
            warnings.warn("Using window size in datapoints number for uneven data.", UserWarning)                
        return pd.Series(self.lc.mag).rolling(window, min_periods=window, win_type='boxcar', center=True, closed='neither').mean().to_numpy()


class WaveForm:
    def __init__(self, folded_lc, waveform_type='uneven_savgol'):
        """
        Initialize the WaveForm class.

        Parameters:
        - folded_lc (FoldedLightCurve): The folded light curve to be analyzed.
        - method (str): The waveform analysis method to be used.

        Raises:
        - TypeError: If folded_lc is not an instance of FoldedLightCurve.
        """
        if not isinstance(folded_lc, FoldedLightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        else:
            self._lc = folded_lc
        self._waveform_type = waveform_type
  
    def circular_rolling_average_number(self, window_size=5):
        """
        Calculate the circular rolling average of the folded light curve.

        Parameters:
        - window_size (int): The window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform.
        """
        extended_mag = np.concatenate((self._lc.mag_phased, self._lc.mag_phased, self._lc.mag_phased))
        extended_waveform = np.array([
            np.mean(extended_mag[i - int(0.5 * window_size):
                i + int(np.round((0.5 * window_size)))])\
                    for i in np.arange(int(0.5 * window_size),\
                        len(extended_mag) -\
                            int(np.round(0.5*window_size + 1)))])
        
        return extended_waveform[self._lc.N - int(0.5 * window_size):
            2 * self._lc.N - int(0.5 * window_size)]
        
    def savgol(self, window=10, polynom=3):
        return sp.signal.savgol_filter(self._lc.mag_phased, window, polynom)
    

    def circular_rolling_average_phase(self, wd_phase=0.1):
        """
        Calculate the circular rolling average of the folded light curve based on phase.

        Parameters:
        - wd_phase (float): The phase window size for the rolling average.

        Returns:
        - np.array: The circular rolling average waveform based on phase.
        """
        waveform = np.full(len(self._lc.phase), np.nan)
        extended_phase = np.concatenate((self._lc.phase - 1, self._lc.phase, 1 + self._lc.phase))
        extended_mag = np.concatenate((self._lc.mag_phased, self._lc.mag_phased, self._lc.mag_phased))
        for i, p in enumerate(self._lc.phase):
            select = np.where((extended_phase <= p + wd_phase/2.) & (extended_phase > p - wd_phase/2.))[0]
            waveform[i] = np.mean(extended_mag[select])
        return waveform

    def waveform_H22(self, kernel=4):
        """
        Code used in Hillenbrand et al. 2022:
        Source: https://github.com/HarritonResearchLab/NAPYSOs
        """
        # Create the residual curve
        # We use three periods and extract the middle to prevent edge effects
        three_periods = np.concatenate((self._lc.mag_phased, self._lc.mag_phased, self._lc.mag_phased))
        boxcar = Box1DKernel(len(self._lc.mag_phased) // kernel)
        smooth_mag = convolve(three_periods, boxcar)
        smooth_mag = smooth_mag[np.size(self._lc.mag_phased):2*np.size(self._lc.mag_phased)]
        return smooth_mag

    def waveform_Cody(self, n_point=50):
        return sp.ndimage.filters.median_filter(self._lc.mag_phased, size=n_point, mode='wrap')   
    
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
        x = np.concatenate((self._lc.phase - 1, self._lc.phase, 1 + self._lc.phase))
        y = np.concatenate((self._lc.mag_phased, self._lc.mag_phased, self._lc.mag_phased)) 
        return  uneven_savgol(x, y, window, polynom)  [self._lc.N:2*self._lc.N]
    
    def residual_magnitude(self, waveform_type='uneven_savgol', **kwargs):
        """
        Calculate the residual magnitude after waveform subtraction.
        """
        if waveform_type == 'savgol':
            #needs to catch kwarfgs
            waveform = self.savgol(**kwargs)
        elif waveform_type == 'Cody':
            #needs to catch kwarfgs
            waveform = self.waveform_Cody(**kwargs)
        elif self._waveform_type == 'circular_rolling_average_phase':
            wd_phase = kwargs.get('wd_phase', 0.1)
            waveform = self.circular_rolling_average_phase(wd_phase=wd_phase)
        elif self._waveform_type == 'circular_rolling_average_number':
            window_size = kwargs.get('window_size', 0.1*self._lc.N)
            waveform = self.circular_rolling_average_number(window_size=window_size)
        elif self._waveform_type == 'H22':
            kernel = kwargs.get('kernel', 4.)
            waveform = self.waveform_H22(kernel=kernel)
        elif self._waveform_type == 'uneven_savgol':
            wd = round(0.1*self._lc.N)
            if wd % 2 == 0:
                wd += 1
            window = kwargs.get('window', wd)
            polynom = kwargs.get('polynom', 3)
            waveform = self.uneven_savgol(window, polynom)
        else:
            raise ValueError("Method _{0}_ not implemented.".format(self._waveform_type))
        return self._lc.mag_phased - waveform

def uneven_savgol_(x, y, window, polynom):
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
    A = np.empty((window, polynom))     # Matrix
    tA = np.empty((polynom, window))    # Transposed matrix
    t = np.empty(window)                # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed

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