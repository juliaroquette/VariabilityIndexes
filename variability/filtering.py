"""
Colection of fitlers
"""

import warnings
import numpy as np
import scipy as sp
import pandas as pd
from variability.lightcurve import LightCurve, FoldedLightCurve


class Filtering:
    def __init__(self, lc):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self.even = is_evenly_spaced(self)
        if not self.even:
            warnings.warn("Time series may not be evenly spaced.", UserWarning)

    def filter(self, methhod='', **kargs):
        """
        Apply a filter of choise to detrend light-cruve.
        
        TODO:
        Polish the list of functions for filtering and include
        their application in this method.
        """
        pass

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

    @classmethod
    def get_num_time(time, timescale=10.):
        """
        Assuming the data is in days, this 
        function conts how many points you need 
        to cover a given timescale
        """
        if not self.even:
            warnings.warn("Time series may not be evenly spaced - no direct conversion between number of points and timescale.", UserWarning)
        time = time - np.nanmin(time)
        return len(time[time < timescale])
    
    #detrending
    def Cody_long_trend(self, timescale=10.):
        """
        Code used in Cody et al. 2018,2020
        window=715 is appropriate for K2 data and equivalent to 10days
        Source:
        """
        window = get_num_time(self.lc.time, timescale)
        return scipy.ndimage.median_filter(self.lc.mag, size=window, mode='nearest')
    
        
    def savgol_longtrend(self, timescale=10., polynom=3) :
        """
        To be changed to timescale rather than window size
        """
        if not self.even:
            warnings.warn("SavGol Scipy expects even time series - which may not be true here.", UserWarning)        
        window = get_num_time(self.lc.time, timescale)
        return savgol_filter(self.lc.mag, window, polynom)
    
    def uneven_savgol(self, window, polynom):
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
        x = self.lc.time
        y = self.lc.mag

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
    def __init__(self, folded_lc, type=''):
        # self.phase, self.mag_pahsed, self.err_pahsed
        if not isinstance(folded_lc, FoldedLightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        else:
            self.folded_lc = folded_lc
    
              
    # def waveform(self, window_size=5, min_per=1.):
    #     """
    #     returns the rolling percentile of an ordered light-curve
    #     if window_size is an integer larger or equal 1: then rolling 
    #     percentage is used for waveform. 
    #     if window_size is a fraction between 0 and 1, then window_size is a 
    #     percentage of the phase to be used for the mooving average
    #     """
    #     if (window_size >= 1):
    #         window_size = int(window_size)
    #         min_per = int(min_per)
    #         if window_size > 1:
    #             extended_data = np.concatenate((self.mag_phased[-(window_size
    #                                                               - 1):],
    #                                             self.mag_phased, 
    #                                             self.mag_phased[:(window_size
    #                                                               - 1)]))
    #         else:
    #             extended_data = self.mag_phased
    #         waveform = pd.Series(extended_data).rolling(
    #                 window_size, min_periods=window_size, win_type='boxcar',
    #                 center=True,closed='neither').mean()
    #         self.mag_waveform = waveform[(window_size):-(window_size - 2)]
    #     elif (window_size < 1) & (window_size > 0):
    #         wd_phase = window_size
    #         waveform = np.full(len(self.phase), np.nan)
    #         extended_phase = np.concatenate((self.phase - 1, self.phase, 1 + self.phase))
    #         extended_mag = np.concatenate((self.mag_phased, self.mag_phased, self.mag_phased))
    #         for i, p in enumerate(self.phase[3:]):
    #             select = np.where((extended_phase <= p + wd_phase/2.) &
    #                               (extended_phase > p - wd_phase/2.))[0]
    #             waveform[i] = np.nanmean(extended_mag[select])
    #         self.mag_waveform = waveform
    #     else:
    #         self.mag_waveform = np.full(len(self.phase), np.nan)

  
#     def circular_rolling_average_waveform(self, window_size=5):
#         """
#         Calculates the rolling average while centering the data around the phase value.
        
#         Parameters:
#             mag_phased (array-like): Array of phased data.
#             window_size (int): Size of the rolling window in number of datapoints. Default is 5.
            
#         Returns:
#             waveform (array-like): Array with the waveform .
            
#         Assumes phased data is circular.
#         """        
#         extended_mag = np.concatenate((self.mag_phased, self.mag_phased,
#                                        self.mag_phased))
#         extended_waveform = np.array([
#             np.mean(extended_mag[i - int(0.5 * window_size):
#                 i + int(np.round((0.5 * window_size)))])\
#                     for i in np.arange(int(0.5 * window_size),\
#                         len(extended_mag) -\
#                             int(np.round(0.5*window_size + 1)))])
        
#         return extended_waveform[self.folded_lc.N - int(0.5 * window_size):
#             2 * self.folded_lc.N - int(0.5 * window_size)]
        
#     def savgol(self):
#         return sp.signal.savgol_filter(self.mag_phased, window, 3)
    
#     def waveform_Cody(self, n_point=50):
#         return sp.ndimage.filters.median_filter(self.mag_phased, size=n_point, mode='wrap')        


# def waveform_phase_fraction(phase, mag_phased, wd_phase=0.1):
#     """
#     Calculate the waveform phase fraction.

#     Parameters:
#     - phase (array-like): Array of phase values.
#     - mag_phased (array-like): Array of magnitude values.
#     - wd_phase (float, optional): Width of the phase window in % of phase. Default is 0.1.

#     Returns:
#     - waveform (array-like): Array with the waveform .
#     """
#     waveform = np.full(len(phase), np.nan)
#     extended_phase = np.concatenate((phase - 1, phase, 1 + phase))
#     extended_mag = np.concatenate((mag_phased, mag_phased, mag_phased))
#     for i, p in enumerate(phase[3:]):
#         select = np.where((extended_phase <= p + wd_phase/2.) & (extended_phase > p - wd_phase/2.))[0]
#         waveform[i] = np.nanmean(extended_mag[select])
#     return waveform

# def waveform_H22(mag_phased, kernel=4):
#     """
#     Code used in Hillenbrand et al. 2022:
#     Source: https://github.com/HarritonResearchLab/NAPYSOs
#     """
#     # Create the residual curve

#     # We use three periods and extract the middle to prevent edge effects
#     three_periods = np.concatenate((mag_phased, mag_phased, mag_phased))
#     boxcar = Box1DKernel(len(mag_phased) // kernel)
#     smooth_mag = convolve(three_periods, boxcar)
#     smooth_mag = smooth_mag[np.size(mag_phased):2*np.size(mag_phased)]
#     return smooth_mag



# mag_roll   = pd.Series(mag_phased).rolling(window, min_periods=window, win_type='boxcar', center=True, closed='neither').mean()

#         # sav gol filter
        
# mag_mf = medfilt(mag_phased, window)




