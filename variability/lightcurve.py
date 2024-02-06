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
        
        self.noisy_mag = mean_magnitude + np.random.normal(scale=noise_level, size=self.n_epochs)
        self.err = abs(np.random.normal(loc=mean_error, scale=noise_level, size=self.n_epochs))
        
    def periodic(self, ptp_amp = 0.2, period=8., phi=0.):
        self.mag_sin = self.noisy_mag + 0.5*ptp_amp * np.sin(2 * np.pi * (self.time - np.min(self.time)) / period + phi)
    
    def quasiperiodic(self, std=0.02, ptp_amp= 1., period=10., phi=0.):
        '''
        Generates a lc with amplitude changing over time. For each time step, the amplitude is drawn from a Gaussian distribution

        Args :
            time : light curve time array
            std : std of Gaussian

        returns :
            mag
            
        TO DO: We need to add a few constraints to the degree of quasiperiodicity
        (for example, it has to be smaller than a fraction of the amplitude)
        '''
        random_steps     = np.random.normal(0, std, len(self.time))            
        amp_t            = np.cumsum(random_steps) + ptp_amp
        self.mag_qp = self.noisy_mag +\
            0.5 * amp_t * np.sin(2 * np.pi * (self.time - np.min(self.time)) / period + phi) 
    
    def eclipsing_binary(self, 
           ptp_amp = 0.3, 
           period=2., 
           phi=0., #in terms of phase, 
           eclipse_duration=0.3 #in terms of phase
           ):
        '''
        Generates a lc with for a strictly periodic and eclipsing-like lightcurve
        '''
        self.mag_ec = self.noisy_mag
        eclipse_start = phi   # Start time of the eclipse
        eclipse_end = phi + eclipse_duration  # End time of the eclipse
        eclipse_depth = ptp_amp  # Depth of the eclipse (0.0 to 1.0)
        # n = (2.*np.pi)*np.random.random(1) # get a random phase
        n = 0.
        while n < max(self.time):
            eclipse_mask = np.logical_and(self.time >= n + eclipse_start  , self.time <= n + eclipse_end)
            self.mag_ec[eclipse_mask] -= eclipse_depth
            n += period
    
    def quasiperiodic_dipper(self, 
                             amp_mean, 
                             amp_std, 
                             period, 
                             dip_factor):
        """
        Calculate the quasi-periodic dipper magnitude.

        Parameters:
        - t: Time array
        - amp_mean: Mean amplitude of the random semi-amplitude
        - amp_std: Standard deviation of the random semi-amplitude
        - period: Period of the sine wave
        - dip_factor: Dip factor quantifying dip strength

        Returns:
        - mag_qpd: Quasi-periodic dipper magnitude
        """
        amp_rand = np.random.normal(amp_mean, amp_std, len(t))

        term1 = amp_rand/2 * np.sin(2 * np.pi * t / period)
        term2 = 0.5 * dip_factor * np.sin(2 * np.pi * t / period) * (1 - np.sin(2 * np.pi * t / period))

        mag_qpd = term1 + term2 + np.random.normal(0, 0.05, len(self.time))

        self.mag_qpd = self.noisy_mag + mag_qpd
    
    def aperiodic_dippers(self, num_dips=3,
                          dip_depth_range=(0.3, 1), 
                          dip_width_range=(0.5, 2.5),
                          amp = 1):
        """
        Generate a synthetic light curve with aperiodic dips. 
        Based on a random walk.
        """
    
        rand_walk = np.cumsum(np.random.randn(len(self.time)))

        for _ in range(num_dips):
            dip_position = np.random.randint(0, len(self.time))
            dip_depth    = np.random.uniform(*dip_depth_range)
            dip_width    = np.random.uniform(*dip_width_range)

            gaussian_peak = dip_depth * np.exp(-((self.time - dip_position) / (dip_width / 2))**2)
            rand_walk    -= gaussian_peak

        # Normalize rand_walk to have a peak-to-peak amplitude of 1
        normalized_rand_walk = (rand_walk - np.min(rand_walk)) / (np.max(rand_walk) - np.min(rand_walk))

        # Scale with self.amp and add self.med_mag
        self.mag_apd = noisy_mag + normalized_rand_walk * amp
        
    
    def periodic_bursting(self):
        pass
    
    def quasiperiodic_bursting(self, std=0.02, ptp_amp=1, period=8., burst_factor=2):
        '''
        Generates a quasi periodic burst light curve with amplitude changing over time.
        Amplitude is higher on the upper part of the sine (bursts).

        Args:
            std: std of Gaussian for overall amplitude variation
            amp, frequency : parameters of the sine
            burst_factor: factor controlling burstiness (higher values result in stronger bursts)

        Returns:
            mag_qpb
        '''

        # Calculate cumulative amplitude with burstiness
        random_steps = np.random.normal(0, std, len(self.time))
        amp_t = np.cumsum(random_steps) + ptp_amp
        self.mag_qpb = self.noisy_mag + amp_t + burst_factor * 0.5 * (1 + np.sin(2 * np.pi * (self.time - min(self.time) / period)))

    
    def aperiodic_bursting(self, num_bursts=3, burst_depth_range=(0.3, 1.), burst_width_range=(.5, 2.5), ptp_amp = 1):
        """
        Generate a synthetic light curve with aperiodic bursts.
        Based on a random walk.
        """
        rand_walk = np.cumsum(np.random.randn(len(self.time)))

        for _ in range(num_bursts):
            burst_position = np.random.randint(0, len(self.time))
            burst_depth    = np.random.uniform(*burst_depth_range)
            burst_width    = np.random.uniform(*burst_width_range)

            gaussian_peak = burst_depth * np.exp(-((self.time - burst_position) / (burst_width / 2))**2)
            rand_walk    -= gaussian_peak

        # Normalize rand_walk to have a peak-to-peak amplitude of 1
        normalized_rand_walk = (rand_walk - np.min(rand_walk)) / (np.max(rand_walk) - np.min(rand_walk))

        # Scale with self.amp and add self.med_mag
        self.mag_apb = self.noisy_mag + normalized_rand_walk * ptp_amp
        
    def multiperiodic():
        pass

    
    