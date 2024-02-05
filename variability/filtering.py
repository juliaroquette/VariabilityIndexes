"""
Colection of fitlers
"""
import numpy as np
import scipy as sp
import warnings
from variability.lightcurve import LightCurve, FoldedLightCurve


class Filtering:
    def __init__(self, lc):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self.even = is_evenly_spaced(self)
        if not self.even:
        warnings.warn("Time series may not be evenly spaced.", UserWarning)



    def is_evenly_spaced(self, tolerance=0.1):
        """
        tests if a function is close to evenly spaced.
        The default tolerance checsk if less than 0.01% of t
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
    
        
    def savgol_longtrend(time, mag) :
        """
        To be changed to timescale rather than window size
        """
        ws =  int(0.25*len(mag))
        return savgol_filter(mag, ws, 3)
    


   

    
    
    # def smooth(self, window_days=20.):
    
    #     for given light-curve with a time and mag, 
    #     use a window wd in days to smooth the light-curve
    #     then remove the smoothed curve form the 
    #     """
    #     pass
        # smoothed_values = np.zeros_like(self.lc.mag, dtype=float)
        # for i in range(len(self.lc.time)):
        #     start_time = self.lc.time[i] - window_days/2
        #     end_time = self.lc.time[i] + window_days/2
        #     select = np.where((self.lc.time >= start_time) & (self.lc.time <= end_time))
        #     smoothed_values[i] = np.nanmean(self.lc.mag[select])
        # self.detrended_mag = self.lc.mag - smoothed_values + np.nanmean(self.lc.mag)
    


    # def get_waveform():
    #     pass
        
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

# class WaveForm:
#     def __init__(self, folded_lc, type=''):
#         # self.phase, self.mag_pahsed, self.err_pahsed
#         if not isinstance(folded_lc, FoldedLightCurve):
#             raise TypeError("lc must be an instance of LightCurve")
#         else:
#             self.folded_lc = folded_lc
    
        
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




