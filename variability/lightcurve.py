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
import scipy.stats as ss
import pandas as pd
import warnings
np.random.seed(42)

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
        - min (float): Minimum value of the magnitudes.
        - max (float): Maximum value of the magnitudes.
        - time_max (float): Maximum value of the observation times.
        - time_min (float): Minimum value of the observation times.
        - ptp (float): Peak-to-peak amplitude of the magnitude values. 
                       Defined as the difference between the median values for the datapoints 
                       in the 5% outermost tails of the distribution.
    """

    def __init__(self,
                 time,
                 mag,
                 err,
                 mask=None,
                 is_flux=False):
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
        self.is_flux = is_flux
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
    
    @property
    def ptp(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 5%outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        tail = round(0.05 * self.N)
        return  np.median(np.sort(self.mag)[-tail:]) - np.median(np.sort(self.mag)[:tail])
        
      

class FoldedLightCurve(LightCurve):
    """
    Represents a folded light curve with time values folded for a given timescale.
    """
    def __init__(self,
                 timescale=None,
                 **kwargs):
        from variability.filtering import WaveForm
        # makes sure this is also a LightCurve object
        if 'lc' in kwargs:
            time = kwargs['lc'].time
            mag = kwargs['lc'].mag
            err = kwargs['lc'].err
        elif all(key in kwargs for key in ['time', 'mag', 'err']):
            time = kwargs['time']
            mag = kwargs['mag']
            err = kwargs['err']
        else:
            raise ValueError("Either a LightCurve object or time, mag and err arrays must be provided")
        mask = kwargs.get('mask', None)
        super().__init__(time, mag, err, mask=mask)
        # phasefold lightcurve to a given timescale
        assert timescale is not None, "A timescale must be provided"
        self._timescale = timescale 
        self._get_phased_values()
        # phasefold lightcurves have a waveform
        self._window_param = kwargs.get('window_param', {})
        self.wf = WaveForm(self.phase, self.mag_phased)
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
            self._get_phased_values()
        else:
            raise ValueError("Please enter a valid _positive_ timescale")        
        
    def _get_waveform(self, 
                      waveform_type='uneven_savgol', 
                      waveform_params={}):
        self._waveform_type = waveform_type
        self._waveform_params = waveform_params
        self.waveform = self.wf.get_waveform(waveform_type=self._waveform_type, 
                                             waveform_params=waveform_params)
        # phasefolded lightcurves also have a residual curve between the waveform and the lightcurve
        self.residual = self.wf.residual_magnitude(waveform_type=waveform_type,
                                                   waveform_params=waveform_params)
    
    # @property
    # def waveform_type(self):
        # return self._waveform_type

    # @waveform_type.setter
    # def waveform_type(self, new_method):
        # self._waveform_type = new_method
        # self._get_waveform()
    
    # @property
    # def waveform_param(self, **kwargs):
        # return kwargs
    
    # @waveform_param.setter
    # def waveform_param(self, **kwargs):
        # self._get_waveform(**kwargs)

class SyntheticLightCurve:
    """
    A class to generate synthetic light curves.
    
    Has the same structure as a LightCurve object. 
    
    """
    def __init__(self, 
                 faint=False,
                 **kwargs):

        # check if an observational window was passed as input        
        if 'time' in kwargs.keys():
            mask = np.where(np.isfinite(kwargs['time']))[0]
            self.time = np.asarray(kwargs['time'], dtype=float)[mask]
        else:
            assert 'survey_window' in kwargs.keys(), "Either a time array or survey_window keyword must be provided"
            if kwargs['survey_window'] == 'K2':
                """ 
                Based on a typical light-curve from K2 as in Cody+ 2018AJ....156...71C
                """
                # 1 observation every 30 minutes
                cadence = 30./60./24. 
                # 78 days of observations
                timespan = 80
                
                    # Kp = 8--18 mag 
                    # 10 - .15/100
                    # 15  0.0032

                # typical noise level is 1.8mmag at 16th Kp magnitude
                if bool(faint):
                    mean_mag = 15.
                    noise_level = 0.0032
                    rms_noise = 0.0032
                else:
                    mean_mag = 10.0 #15.
                    noise_level = 0.0015 # 0.0032
                    rms_noise = 0.0015 #0.0032
                self.time = np.arange(0, timespan, cadence)

            elif kwargs['survey_window'] == 'CoRoT':
                """ 
                Based on a typical light-curve from CoRoT as in Cody+ 2014AJ....147...82
                """
                # 1 observation every 512 s
                cadence = 512./60./60/24. 
                # just over 37 days of observations
                timespan = 38.68
                # typical rms is 0.01-0.1 at 17th Kp magnitude
                if bool(faint):
                    mean_mag = 15.0 #12
                    noise_level = 0.01 #0.001
                    rms_noise = 0.01 # 0.001
                else:
                    mean_mag = 12.
                    noise_level = 0.001
                    rms_noise = 0.001  
                self.time = np.arange(0, timespan, cadence)
            elif kwargs['survey_window'] == 'TESS':
                raise NotImplementedError
            elif kwargs['survey_window'] == 'Rubin':
                raise NotImplementedError
            elif kwargs['survey_window'] == 'ZTF':
                if bool(faint):
                    mean_mag = 18.
                    noise_level = 0.02
                    rms_noise = 0.02
                else:
                    mean_mag = 15. #18.
                    noise_level = 0.01 #0.02
                    rms_noise = 0.01 #0.02
                self.read_observational_window(kwargs['survey_window'])
            elif kwargs['survey_window'] == 'ASAS-SN-V':
                if bool(faint):                
                    mean_mag = 15. 
                    noise_level = 0.04
                    rms_noise = 0.04 #0.02
                else:
                    mean_mag = 12
                    noise_level = 0.02
                    rms_noise = 0.02
                self.read_observational_window(kwargs['survey_window'])
            elif kwargs['survey_window'] == 'ASAS-SN-g':
                if bool(faint):                
                    mean_mag = 15. 
                    noise_level = 0.04
                    rms_noise = 0.04 #0.02
                else:
                    mean_mag = 12
                    noise_level = 0.02
                    rms_noise = 0.02
                self.read_observational_window(kwargs['survey_window'])        
            elif kwargs['survey_window'] == 'GaiaDR3':
                if bool(faint):
                    mean_mag = 17.0 #12
                    noise_level = 0.0050 #0.001
                    rms_noise = 0.0050 # 0.001
                else:
                    mean_mag = 12.5
                    noise_level = 0.0008
                    rms_noise = 0.0008 
                self.read_observational_window(kwargs['survey_window'])
            elif kwargs['survey_window'] == 'GaiaDR4':
                if bool(faint):
                    mean_mag = 17.0 #12
                    noise_level = 0.0050 #0.001
                    rms_noise = 0.0050 # 0.001
                else:
                    mean_mag = 12.5
                    noise_level = 0.0008
                    rms_noise = 0.0008 
                self.read_observational_window(kwargs['survey_window'])
            elif kwargs['survey_window'] == 'GaiaDR4-geq20':
                if bool(faint):
                    mean_mag = 17.0 #12
                    noise_level = 0.0050 #0.001
                    rms_noise = 0.0050 # 0.001
                else:
                    mean_mag = 12.5
                    noise_level = 0.0008
                    rms_noise = 0.0008 
                self.read_observational_window(kwargs['survey_window'])                
            elif kwargs['survey_window'] == 'GaiaDR5':
                if bool(faint):
                    mean_mag = 17.0 #12
                    noise_level = 0.0050 #0.001
                    rms_noise = 0.0050 # 0.001
                else:
                    mean_mag = 12.5
                    noise_level = 0.0008
                    rms_noise = 0.0008 
                self.read_observational_window(kwargs['survey_window'])                
            # elif kwargs['survey_window'] == 'AllWISE':
                # if bool(faint):
                #     mean_mag = 17.0 #12
                #     noise_level = 0.0050 #0.001
                #     rms_noise = 0.0050 # 0.001
                # else:
                #     mean_mag = 12.5
                #     noise_level = 0.0008
                #     rms_noise = 0.0008 
                # self.read_observational_window(kwargs['survey_window'])
                # raise NotImplementedError                
            else:
                raise ValueError('Invalid survey window, possible values are: K2, TESS, Rubin, ZTF, ASAS-SN, GaiaDR3, GaiaDR4, AllWISE, CoRoT')
        
        self.n_epochs = len(self.time)
        self.time.setflags(write=False)  # Set the array as read-only      
        self.err = abs(np.random.normal(loc=rms_noise, scale=noise_level, size=self.n_epochs))  
        self._noisy_mag = mean_mag + 1.* self.err  # np.random.normal(loc=mean_mag, scale=noise_level, size=self.n_epochs)
        self.err = abs(np.random.normal(loc=rms_noise, scale=noise_level, size=self.n_epochs))
        self._noisy_mag.setflags(write=False)  # Set the array as read-only        
        self.err.setflags(write=False)  # Set the array as read-only
        #
        # Defines some parameters for the light-curves
        #
        if 'ptp_amp' in kwargs.keys():
            self.ptp_amp = kwargs['ptp_amp']
        else:
            self.ptp_amp = 0.1
            warnings.warn(f'Peak-to-peak amplitude not provided, using default value of {self.ptp_amp}')
        if 'period' in kwargs.keys():
            self.period = kwargs['period']
        else:
            self.period = 8.
            warnings.warn(f'Period not provided, using default value of {self.period}')

    def read_observational_window(self, survey_window):
        """
        Create a SyntheticLightCurve object from a file.

        Args:
            filename (str): The name of the file to read from.

        Returns:
            SyntheticLightCurve: The light curve object.
        """
        observational_windows_filenames = {
            'AllWISE': 'AllWISE.csv',
            'GaiaDR3': 'gaia_DR3.csv',
            'GaiaDR4': 'gaia_DR4.csv',
            'GaiaDR4-geq20': 'gaia_DR4_high.csv',
            'GaiaDR5': 'gaia_DR5.csv',
            'TESS': 'TESS.csv',
            'ZTF': 'ZTF.csv',
            'ASAS-SN-V': 'ASAS-SN_V.csv',
            'ASAS-SN-g': 'ASAS-SN_g.csv',
        }
        df = pd.read_csv('../data/' + \
            observational_windows_filenames[survey_window])
        self.time = df['jd_norm'].to_numpy()
        
    
    def periodic_symmetric(self, 
                           ptp_amp=None,
                           period=None, 
                           phi_0=0.):
        """
        Periodic Symmetric light-curve
        """
        if ptp_amp is None:
            ptp_amp = self.ptp_amp
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')
        if period is None:
            period = self.period
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')            
        self.mag_ps = self._noisy_mag + 0.5 * ptp_amp * \
            np.sin(2. * np.pi * (self.time - np.min(self.time))\
                / period + phi_0)
    
    def quasiperiodic_symmetric(self, 
                                std=0.05, 
                                ptp_amp= None, 
                                period=None, 
                                phi_0=0.):
        '''
      quasiperiodic symmetric
            
        TO DO: We need to add a few constraints to the degree of quasiperiodicity
        (for example, it has to be smaller than a fraction of the amplitude)
        '''
        if ptp_amp is None:
            ptp_amp = self.ptp_amp
            warnings.warn(f'quasiperiodic_symmetric: \n Using class default value of {self.ptp_amp}')
        if period is None:
            period = self.period
            warnings.warn(f'quasiperiodic_symmetric: \n Using class default value of {self.ptp_amp}')          
        
        amp_t = self.random_walk_1D(n_steps=len(self.time),
                                    ptp=ptp_amp,
                                    std=std,
                                    type_of_step='normal')
        print(ptp_amp, max(amp_t), min(amp_t))
        self.mag_qps = self._noisy_mag +\
            0.5 * amp_t * np.sin(2 * np.pi * (self.time - np.min(self.time)) / period + phi_0) 
    
    def eclipsing_binary(self, 
           ptp_amp = None, 
           secondary_fraction = 0.2, # primary_ptp/secondary_ptp
           period=None, 
           eclipse_duration=0.15, #in terms of phase
           primary_eclipse_start = 0.    # Start time of the eclipse
           ):
        '''
        Generates a lc with for a strictly periodic and eclipsing-like lightcurve
        '''
        if ptp_amp is None:
            ptp_amp = self.ptp_amp
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')
        if period is None:
            period = self.period
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')    
        assert secondary_fraction < 1, "Secondary fraction must be smaller than 1"
        assert primary_eclipse_start + eclipse_duration < 0.5, "First eclipse must happen in the first half of the period"
        # create a base noisy light-curve
        self.mag_eb = self._noisy_mag.copy()
        # calculate the phase for given period
        phase =  (self.time - min(self.time))/period - np.floor((self.time-min(self.time))/period)
        # define eclipse parameters
        secondary_eclipse_start = primary_eclipse_start + 0.5  # Start time of the eclipse
        primary_eclipse_depth = ptp_amp  # Depth of the eclipse (0.0 to 1.0)
        # select parts of the phased light-curve to be eclipsed
        primary_eclipse = np.logical_and(phase >= primary_eclipse_start, phase <= primary_eclipse_start + eclipse_duration)
        secondary_eclipse = np.logical_and(phase >= primary_eclipse_start + 0.5, phase <= primary_eclipse_start + 0.5 + eclipse_duration)
        # Convert phase fraction to be eclipsed into appropriate radian values for the model
        phi_ = (phase[primary_eclipse] - primary_eclipse_start) * np.pi / eclipse_duration 
        # eclipse relevant parts of the light-curve
        self.mag_eb[primary_eclipse] -= primary_eclipse_depth * \
            np.sin(phi_)
        # repeat for secondary eclipse
        phi_ = (phase[secondary_eclipse] - secondary_eclipse_start) * np.pi / eclipse_duration            
        self.mag_eb[secondary_eclipse] -= secondary_fraction * primary_eclipse_depth * np.sin(phi_)
    
    def AATau(self, 
           ptp_amp = None, 
           period=None, 
           dip_width=0.9, #in terms of phase
           dip_start = 0.05    # Start time of the eclipse
           ):
        '''
        Generates a lc with for a AA Tau-like lightcurve
        '''
        if ptp_amp is None:
            ptp_amp = self.ptp_amp
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')
        if period is None:
            period = self.period
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.ptp_amp}')            
        assert dip_width < 1, "Dip-width must be smaller than 1"
        # assert primary_eclipse_start + eclipse_duration < 0.5, "First eclipse must happen in the first half of the period"
        self.mag_AATau = self._noisy_mag.copy()
        phase =  (self.time - min(self.time))/period - np.floor((self.time - min(self.time))/period)
        # generates amplitudes for each phase
        n_phase = np.floor((self.time - min(self.time))/period)
        
        #
        dip_in = np.logical_and(phase >= dip_start, phase <= dip_start + dip_width)
        phi_ = (phase[dip_in] - dip_start) * np.pi / dip_width
        ptp_A = self.random_walk_1D(len(self.time), ptp_amp)
        self.mag_AATau[dip_in] -= abs(ptp_A[dip_in]) * np.sin(phi_)
    
    
    # def quasiperiodic_dipper(self, 
    #                          ptp_amp, 
    #                          amp_std, 
    #                          period, 
    #                          dip_factor):
    #     """
    #     Calculate the quasi-periodic dipper magnitude.

    #     Parameters:
    #     - t: Time array
    #     - amp_mean: Mean amplitude of the random semi-amplitude
    #     - amp_std: Standard deviation of the random semi-amplitude
    #     - period: Period of the sine wave
    #     - dip_factor: Dip factor quantifying dip strength

    #     Returns:
    #     - mag_qpd: Quasi-periodic dipper magnitude
    #     """
    #     ptp_A = self.random_walk_1D(len(self.time), ptp_amp)

    #     amp_rand = np.random.normal(amp_mean, amp_std, len(t))

    #     term1 = amp_rand/2 * np.sin(2 * np.pi * t / period)
    #     term2 = 0.5 * dip_factor * np.sin(2 * np.pi * t / period) * (1 - np.sin(2 * np.pi * t / period))

    #     mag_qpd = term1 + term2 + np.random.normal(0, 0.05, len(self.time))

    #     self.mag_qpd = self._noisy_mag + mag_qpd
    
    # def aperiodic_dippers(self, num_dips=3,
    #                       dip_depth_range=(0.3, 1), 
    #                       dip_width_range=(0.5, 2.5),
    #                       amp = 1):
    #     """
    #     Generate a synthetic light curve with aperiodic dips. 
    #     Based on a random walk.
    #     """
    
    #     rand_walk = np.cumsum(np.random.randn(len(self.time)))

    #     for _ in range(num_dips):
    #         dip_position = np.random.randint(0, len(self.time))
    #         dip_depth    = np.random.uniform(*dip_depth_range)
    #         dip_width    = np.random.uniform(*dip_width_range)

    #         gaussian_peak = dip_depth * np.exp(-((self.time - dip_position) / (dip_width / 2))**2)
    #         rand_walk    -= gaussian_peak

    #     # Normalize rand_walk to have a peak-to-peak amplitude of 1
    #     normalized_rand_walk = (rand_walk - np.min(rand_walk)) / (np.max(rand_walk) - np.min(rand_walk))

    #     # Scale with self.amp and add self.med_mag
    #     self.mag_apd = self._noisy_mag + normalized_rand_walk * amp
        
    
    # def periodic_bursting(self):
    #     pass
    
    # def quasiperiodic_bursting(self, std=0.02, ptp_amp=1, period=8., burst_factor=2):
    #     '''
    #     Generates a quasi periodic burst light curve with amplitude changing over time.
    #     Amplitude is higher on the upper part of the sine (bursts).

    #     Args:
    #         std: std of Gaussian for overall amplitude variation
    #         amp, frequency : parameters of the sine
    #         burst_factor: factor controlling burstiness (higher values result in stronger bursts)

    #     Returns:
    #         mag_qpb
    #     '''

    #     # Calculate cumulative amplitude with burstiness
    #     random_steps = np.random.normal(0, std, len(self.time))
    #     amp_t = np.cumsum(random_steps) + ptp_amp
    #     self.mag_qpb = self._noisy_mag + amp_t + burst_factor * 0.5 * (1 + np.sin(2 * np.pi * (self.time - min(self.time) / period)))

    
    # def aperiodic_bursting(self, num_bursts=3, burst_depth_range=(0.3, 1.), burst_width_range=(.5, 2.5), ptp_amp = 1):
    #     """
    #     Generate a synthetic light curve with aperiodic bursts.
    #     Based on a random walk.
    #     """
    #     rand_walk = np.cumsum(np.random.randn(len(self.time)))

    #     for _ in range(num_bursts):
    #         burst_position = np.random.randint(0, len(self.time))
    #         burst_depth    = np.random.uniform(*burst_depth_range)
    #         burst_width    = np.random.uniform(*burst_width_range)

    #         gaussian_peak = burst_depth * np.exp(-((self.time - burst_position) / (burst_width / 2))**2)
    #         rand_walk    -= gaussian_peak

    #     # Normalize rand_walk to have a peak-to-peak amplitude of 1
    #     normalized_rand_walk = (rand_walk - np.min(rand_walk)) / (np.max(rand_walk) - np.min(rand_walk))

    #     # Scale with self.amp and add self.med_mag
    #     self.mag_apb = self._noisy_mag + normalized_rand_walk * ptp_amp
        
    # def multiperiodic():
    #     pass
    
    # @staticmethod
    # def new_observational_window():
    #     pass
    

    
    @staticmethod
    def random_walk_1D(n_steps, 
                       ptp=1., # final peak-to-peak amplitude of the random walk
                       type_of_step='normal', # normal, unit, skewed-normal
                       skewness=5., # if using skewed-normal, this is the skewness parameter, 
                                   # it can be any real number (positive for dipper)
                        std = None
                       ):
        """
        Perform a 1-dimensional random walk.

        Parameters:
        - n_steps (int): The number of steps in the random walk 
                        (or the number of observations in the light-curve).
        - ptp (float): The desired peak-to-peak amplitude for the light-curve.
                       This is used to rescale the generated random walk ptp. 

        Returns:
        - positions (ndarray): The positions after each step of the random walk,
                               scaled to the desired peak-to-peak amplitude.

        References:
        - Random Walk: https://en.wikipedia.org/wiki/Random_walk
        """
        if std is None:
            std = 1/3.
        if type_of_step == 'normal':
            steps = np.random.normal(loc=0, scale=std, size=n_steps)
        elif type_of_step == 'unit':
            steps = np.random.choice([-1, 1], size=n_steps)
        elif type_of_step == 'skewed-normal':
            steps = ss.skewnorm.rvs(loc=0, scale=std, a=skewness, size=n_steps)
        else:
            raise ValueError('Invalid type of step, possible values are: normal, unit, skewed-normal')
        
        positions = np.cumsum(steps)
        tail = round(0.05 * len(positions)) 
        ptp_0 = np.median(np.sort(positions)[-tail:]) - np.median(np.sort(positions)[:tail])
        return (positions - np.median(np.sort(positions)[-tail:])) * (ptp / ptp_0)

    @staticmethod
    def Ornstein_Uhlenbeck(time,
                          tau=10.,
                          mu=15,
                           sigma=0.1 
                           ):
        """
        Implements a mean-reverting stochastic process.
        
        Ornstein-Uhlenbeck process (OU process):
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        
        OU-process stochastic differential equation:
        -> dXt = theta * (mu - Xt) dt + sigma * dWt
        where:
        - Xt is the state of the process at time t
        - theta is the "rate of mean reversion"
        - mu is the mean value of the process
        - sigma is the volatility of the process
        - Wt is a Wiener process (Brownian motion)
        
        In this data-science ebook Chapter 13, section 4:
        https://github.com/ipython-books/cookbook-2nd/
        I found a version of this equation that is parametrised as a function
        of a time-scale tau, rather than theta, which can be interpreted as 
        the mean time-reverting tendency of the process. 
        For this, we can use the relationship:
        -> theta = 1/tau
        -> sigma = sigma_ * sqrt(2/tau)
        
        This can be solved using the Euler-Maruyama solution:
        https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
        
        for a stochastic differential equation:
            -> dXt = a(Xt, t) * dt + b(Xt, t) * dWt
        with initial condition X0 = x0, the approximation to the solution X is
        a Markov chain {Xt} with time increment dt:
            1-> dt = T/N for an evenly spaced interval [0, T]. Although here 
                I am inputing the time array, so I am using the time array to 
                calculate dt
            2-> X0 = mu 
                I am imposing that the initial condition is actually the mean
                of the process
            3-> for i = 1, 2, ..., N:
                X[n+1] = X[n] + a(X[n], t[n]) * dt + b(X[n], t[n]) * dW[n]
                where dW[n] = W[n+1] - W[n] ~ N(0, dt) (Normal distributed 
                                                with mean 0 and variance dt)
        Translating this to the modified version of the OU-process:
            -> a(X[n], t) =  (mu - X[n]) / tau
            -> b(X[n], t) = sigma * sqrt(2/tau)
            -> dW[n] = W[n+1] - W[n] ~ N(0, dt)
        
        
        """
        dt = time[1:] - time[:-1]
        mag = np.full(len(time), mu, dtype=float) #I am imposing that the initial condition is actually de mean of the process
        
        def a_Xnt(tau,
                  mag,
                  mu):
            return (mu - mag) / tau
        
        def b_Xnt(sigma, tau):
            return sigma * np.sqrt(2/tau)
        
        def dW(dt_):
            """
            Implementing dW like that guarantees that dW is calculate for the
            current dt. This makes sure the correct variance properties are kept.
            """
            return np.random.normal(loc=0, scale=np.sqrt(dt_))
        
        dWs = dW(dt)
        mag[1:] = mag[:-1] + a_Xnt(tau, mag[:-1],  mu) * dt + b_Xnt(sigma, tau) * dWs
        return mag

