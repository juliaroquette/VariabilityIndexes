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
import inspect
import warnings
np.random.seed(42)
from variability.filtering import WaveForm
 
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
        return int(len(self.mag))
    
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
    def mean_err(self):
        """
        Returns the mean of the uncertainty values.

        Returns:
            float: Mean value.
        """
        return self.err.mean()    
    
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
        
    @property
    def range(self):
        """
        Returns the range of the magnitude values.

        Returns:
            float: Range value.
        """
        return self.mag.max() - self.mag.min()
    
    @property
    def SNR(self):
        """
        Returns the signal-to-noise ratio of the light curve.

        Returns:
            float: Signal-to-noise ratio.
        """
        return np.sqrt(np.sum((self.mag - self.mean)**2)/np.sum(self.err**2))
    
    def _list_properties(self):
        """
        list properties of the class LightCurve
        """
        property_names = [name for name, value in inspect.getmembers(self.__class__, lambda o: isinstance(o, property))]
        return property_names    
        
    def __str__(self):
        return f'A LightCurve instance has the following properties: {repr(self._list_properties())}'

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
            from variability.timescales import TimeScale
            ts = TimeScale(lc=self)
            frequency_highest_peak, power_highest_peak, FAP_highest_peak = ts.get_LSP_period(periodogram=False)
            self._timescale = 1./frequency_highest_peak
            self._timescale_FAP = FAP_highest_peak*100
            warnings.warn("Automatic timescale estimated from LSP - FAP: {0}".format(self.timescale_FAP))

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
        phase_number = np.floor(self.time/self._timescale)
        sort = np.argsort(phase)
        self.phase = phase[sort]
        self.mag_phased = self.mag[sort]       
        self.err_phased = self.err[sort]
        self.phase_number = phase_number[sort]
        
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
    
    @property
    def timescale_FAP(self):
        """
        False Alarm Probability for the timescale estimated from the Lomb-Scargle periodogram. It is None if timescale was input by the user.
        """
        if hasattr(self, '_timescale_FAP') and self._timescale_FAP is not None:
            return self._timescale_FAP
        else:
            return None              
        
    def _get_waveform(self, **kwargs):
        self.waveform = self.wf.get_waveform(waveform_type= kwargs.get('waveform_type', self._waveform_type), 
                                             waveform_params=kwargs.get('waveform_params', self._waveform_params))
        # phasefolded lightcurves also have a residual curve between the waveform and the lightcurve
        self.residual = self.wf.residual_magnitude(self.waveform)
    
    def _list_properties(self):
        """
        list properties of the class LightCurve
        """
        property_names = [name for name, value in inspect.getmembers(self.__class__, lambda o: isinstance(o, property))]
        return property_names    
        
    def __str__(self):
        return f'A FoldedLightCurve instance has the following properties: {repr(self._list_properties())}'        

    @classmethod
    def suppress_warnings_globally(cls):
        """
        This is class method that enable to suppress warnings globally
        for FoldedLightCurve instances.
        
        usage:
        FoldedLightCurve.suppress_warnings_globally()
        """
        print("Warning for FoldedLightCurves are not supressed")
        cls._suppress_warnings = True

    @classmethod
    def enable_warnings_globally(cls):
        """
        This is a class method to enable
        warnings globally for FoldedLightCurve instances.
        Usage:
        FoldedLightCurve.enable_warnings_globally()
        """
        cls._suppress_warnings = False        


class SyntheticLightCurve:
    """
    A class to generate synthetic light curves.
    
    Being refactored
    """
    def __init__(self, 
                 model='periodic_symmetric',
                 **kwargs):
        # Generate a high-cadence, long time-span light-curve for reference
        # Cadence is the same as CoRoT
        _CADENCE = 512./60./60/24. 
        # Total timespan of a decade
        _T = 365.5*10 
        # time array for ground truth
        self._time = np.linspace(0, _T, int(np.ceil(_T/_CADENCE)))
        # print(self._time.shape)
        # activate parameters common to all models
        if 'ptp_amp' in kwargs.keys():
            self._PTP_AMP = kwargs['ptp_amp']
        else:
            self._PTP_AMP = 0.1
            warnings.warn(f"Peak-to-peak amplitude not provided, using default value of {self._PTP_AMP}")
        if 'timescale' in kwargs.keys():
            self._TIMESCALE = kwargs['timescale']
        else:
            self._TIMESCALE = 8.
            warnings.warn(f"Timescale not provided, using default value of {self._TIMESCALE}")
        # activate the model

        if model == 'periodic_symmetric':
            # requires ptp_amp, timescale and phase
            self._model_name = model
            # load model specific attributes
            self._phase = kwargs.get('phase', 0.)
            # define magnitude variation ground truth
            self._mag = self.periodic_symmetric(self._PTP_AMP, 
                                           self._TIMESCALE, 
                                           phi_0 = self._phase)
            
        elif model == 'eclipsing_binary':
            self._model_name = model
            self.secondary_fraction = kwargs.get('secondary_fraction', 0.2)
            self.eclipse_duration = kwargs.get('eclipse_duration', 0.15)
            self.primary_eclipse_start = kwargs.get('primary_eclipse_start', 0.)
            self._mag = self.eclipsing_binary(self._PTP_AMP,
                                            self._TIMESCALE, 
                                            secondary_fraction=self.secondary_fraction,
                                            eclipse_duration=self.eclipse_duration,
                                            primary_eclipse_start=self.primary_eclipse_start)
            
        elif model == "AATau":
            self._model_name = model
            self._mag = self.AATau(self._PTP_AMP, self._TIMESCALE,
                                  dip_width=0.9, #in terms of phase
                                  dip_start = 0.05    # Start time of the eclipse
                                    )
        elif model == 'quasiperiodic':
            pass
        elif model == 'stochastic':
            pass

        else:
            raise ValueError('Invalid model, possible values are: sinusoidal, quasiperiodic, stochastic, eclipsing_binary')
        # create a light-curve object for each survey
        for survey in ObservationalWindow.surveys_object_names.keys():
            setattr(self, survey, 
                    ObservationalWindow(survey, 
                                        ground_truth=(self._time, self._mag),
                                        timescale=self._TIMESCALE))
        # Ground Truth LCs need to be undersampled otherwise waveform estimator is too slow
        self.ground_truth = FoldedLightCurve(time=self._time[::50], 
                                            mag=self._mag[::50], 
                                            err=np.zeros(len(self._mag[::50])), 
                                            timescale=self._TIMESCALE)
    
    def periodic_symmetric(self, 
                           ptp_amp,
                           period, 
                           phi_0=0.):
        """
        Periodic Symmetric light-curve
        """
        return  0.5 * ptp_amp * np.sin(2. * np.pi * \
            (self._time - np.min(self._time))/ period + phi_0)
    
    def eclipsing_binary(self,
                         ptp_amp,
                         period, 
                         secondary_fraction=0.2, # primary_ptp/secondary_ptp
                         eclipse_duration=0.15, #in terms of phase
                         primary_eclipse_start=0.    # Start time of the eclipse
                         ):
        '''
        Generates a lc with for a strictly periodic and eclipsing-like lightcurve
        '''
        assert secondary_fraction < 1, "Secondary fraction must be smaller than 1"
        assert primary_eclipse_start + eclipse_duration < 0.5, "First eclipse must happen in the first half of the period"
        # make a base magnitude array
        mag_eb = np.zeros_like(self._time)
        # calculate the phase for given period
        phase =  (self._time - min(self._time))/period - np.floor((self._time-min(self._time))/period)
        # define eclipse parameters
        secondary_eclipse_start = primary_eclipse_start + 0.5  # Start time of the eclipse
        primary_eclipse_depth = ptp_amp  # Depth of the eclipse (0.0 to 1.0)
        # select parts of the phased light-curve to be eclipsed
        primary_eclipse = np.logical_and(phase >= primary_eclipse_start, phase <= primary_eclipse_start + eclipse_duration)
        secondary_eclipse = np.logical_and(phase >= primary_eclipse_start + 0.5, phase <= primary_eclipse_start + 0.5 + eclipse_duration)
        # Convert phase fraction to be eclipsed into appropriate radian values for the model
        phi_ = (phase[primary_eclipse] - primary_eclipse_start) * np.pi / eclipse_duration 
        # eclipse relevant parts of the light-curve
        mag_eb[primary_eclipse] += primary_eclipse_depth * \
            np.sin(phi_)
        # repeat for secondary eclipse
        phi_ = (phase[secondary_eclipse] - secondary_eclipse_start) * np.pi / eclipse_duration            
        mag_eb[secondary_eclipse] += secondary_fraction * primary_eclipse_depth * np.sin(phi_)
        return mag_eb
    
    def quasiperiodic_symmetric(self,
                                ptp_amp, 
                                period, 
                                phi_0=0., 
                                std=0.05, 
                                ptp_min=0.):
        '''
      quasiperiodic symmetric
            
        TO DO: We need to add a few constraints to the degree of quasiperiodicity
        (for example, it has to be smaller than a fraction of the amplitude)
        '''
        
        amp_t = ptp_min + abs(self.random_walk_1D(n_steps=len(self._time),
                                    ptp=ptp_amp,
                                    std=std,
                                    type_of_step='normal'))
        return 0.5 * amp_t * np.sin(2 * np.pi * (self._time - np.min(self._time)) / period + phi_0) 
    
    def AATau(self, 
           ptp_amp, 
           period, 
           dip_width=0.9, #in terms of phase
           dip_start = 0.05    # Start time of the eclipse
           ):
        '''
        Generates a lc with for a AA Tau-like lightcurve
        '''
        assert dip_width < 1, "Dip-width must be smaller than 1"
        # assert primary_eclipse_start + eclipse_duration < 0.5, "First eclipse must happen in the first half of the period"
        #
        #
        # make a base magnitude array
        import random
        def generate_random_numbers(n, x, y):
            random_numbers = [random.uniform(x, y) for _ in range(n)]
            return random_numbers        
        mag = np.zeros_like(self._time)
        ptp_A = np.zeros_like(self._time)
        # generates amplitudes for each phase
        n_phase = np.floor((self._time - min(self._time))/period) 
        _n_phase = np.unique(n_phase)
        _ptp_A = generate_random_numbers(len(_n_phase), 0.2*ptp_amp, 1.1*ptp_amp) 
        for i, n in enumerate(_n_phase):
            ptp_A[n_phase == n] = _ptp_A[i]

        # calculate the phase for given period
        phase =  (self._time - min(self._time))/period - np.floor((self._time-min(self._time))/period)
        # define eclipse parameters
        # select parts of the phased light-curve to be eclipsed
        dip_in = np.logical_and(phase >= dip_start, phase <= dip_start + dip_width)
        # Convert phase fraction to be eclipsed into appropriate radian values for the model
        phi_ = (phase - dip_start) * np.pi / dip_width 
        __ptp_A = self.Ornstein_Uhlenbeck(self._time, tau=4, sigma=0.02, mu=0)
        # eclipse relevant parts of the light-curve
        mag[dip_in] += ptp_A[dip_in]  * \
            np.sin(phi_[dip_in])
        mag += __ptp_A
        return mag
    
    
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
        dt =  512./60./60/24./2 # the default cadence is half of the CoRoT cadence.
        N = int((max(time) - min(time))/dt)        
        mag = np.full(N, mu, dtype=float) #I am imposing that the initial condition is actually de mean of the process
        _time = np.linspace(min(time), max(time), N)
        
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
        
        for i in range(N - 1):
            mag[i + 1] = mag[i] + a_Xnt(tau, mag[i],  mu) * dt + b_Xnt(sigma, tau) * dW(dt)
        
        
        return np.interp(time, _time, mag)
   
 
    
class SyntheticLightCurve_:
    """
    A class to generate synthetic light curves.
    
    Has the same structure as a LightCurve object. 
    
    TO DO: 
    restructure the code to first generate a highly populated light-curve 
    and just then convolute to the observational window. 
    
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
                if bool(faint):
                    mean_mag = 16.
                    noise_level = 0.01
                    rms_noise = 0.01
                else:
                    mean_mag = 10.
                    noise_level = 0.0003 
                    rms_noise = 0.0003
                self.read_observational_window(kwargs['survey_window'])
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
        self.err = np.random.normal(loc=noise_level, scale=rms_noise, size=self.n_epochs) 
        self._noisy_mag = mean_mag + 1.* self.err  # np.random.normal(loc=mean_mag, scale=noise_level, size=self.n_epochs)
        self.err = abs(self.err)
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
            'ASAS-SN-V': 'ASASSN_V.csv',
            'ASAS-SN-g': 'ASASSN_g.csv',
        }
        df = pd.read_csv('../data/' + \
            observational_windows_filenames[survey_window])
        self.time = df['jd_norm'].to_numpy()
        
    def stochastic(self):
        np.random.seed(42)
        self.mag_s = self.Ornstein_Uhlenbeck(self.time, tau=10, sigma=0.05, mu=15)
        
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
            warnings.warn(f'periodic_symmetric: \n Using class default value of {self.period}')            
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
        self.mag_eb[primary_eclipse] += primary_eclipse_depth * \
            np.sin(phi_)
        # repeat for secondary eclipse
        phi_ = (phase[secondary_eclipse] - secondary_eclipse_start) * np.pi / eclipse_duration            
        self.mag_eb[secondary_eclipse] += secondary_fraction * primary_eclipse_depth * np.sin(phi_)
    
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

    
    def resample_from_lc(t_in, y_in, t_out):
        """
        Simple function for resampling a light-curve to a new time array.
        """
        #include tests for when t_out is outside the range of t_in
        
        return np.interp(t_out, t_in, y_in)

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
        dt =  512./60./60/24./2 # the default cadence is half of the CoRoT cadence.
        N = int((max(time) - min(time))/dt)        
        mag = np.full(N, mu, dtype=float) #I am imposing that the initial condition is actually de mean of the process
        _time = np.linspace(min(time), max(time), N)
        
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
        
        for i in range(N - 1):
            mag[i + 1] = mag[i] + a_Xnt(tau, mag[i],  mu) * dt + b_Xnt(sigma, tau) * dW(dt)
        
        
        return np.interp(time, _time, mag)

class ObservationalWindow:
    """
    Class for loading observational windows for different variability surveys
    of interest.
    
    Surveys currently implemented:
    'K2', 'CoRoT', 'TESS', 'ZTF', 'ASAS-SN-V', 'ASAS-SN-g', 'GaiaDR3',
    'GaiaDR4', 'GaiaDR4-geq20', 'GaiaDR5', 'AllWISE'
    
     self.faint.FoldedLightCurve and self.bright.FoldedLightCurve
     
    Not implemented yet:
    'Rubin'
    
    TO DO:
    include external files for the observational windows
    Goal:
    simulation.survey.faint.LC
    """
    surveys_object_names = {
            'AllWISE': 'AllWISE',
            'GaiaDR3': 'GaiaDR3',
            'GaiaDR4': 'GaiaDR4',
            'GaiaDR4-geq20': 'GaiaDR4geq20',
            'GaiaDR5': 'GaiaDR5',
            'TESS': 'TESS',
            'ZTF': 'ZTF',
            'ASAS-SN-V': 'ASASSN_V',
            'ASAS-SN-g': 'ASASSN_g',
            'K2': 'K2',
            'CoRoT': 'CoRoT'
        }
    def __init__(self, 
                 survey, 
                 ground_truth=(None, None), 
                 timescale=None):
        # load the survey window
        if survey == 'K2':
            time, faint, bright, waveform_type, waveform_params = self._K2_window()
        elif survey == 'CoRoT':
            time, faint, bright, waveform_type, waveform_params = self._CoRoT_window()
        elif survey == 'TESS':
            time, faint, bright, waveform_type, waveform_params = self._TESS_window()
        elif survey == 'Rubin':
            raise NotImplementedError
        elif survey == 'ZTF':
            time, faint, bright, waveform_type, waveform_params = self._ZTF_window()
        elif survey == 'ASAS-SN-V':
            time, faint, bright, waveform_type, waveform_params = self._ASAS_SN_V_window()
        elif survey == 'ASAS-SN-g':
            time, faint, bright, waveform_type, waveform_params = self._ASAS_SN_g_window()
        elif survey == 'GaiaDR3':
            time, faint, bright, waveform_type, waveform_params = self._GaiaDR3_window()
        elif survey == 'GaiaDR4':
            time, faint, bright, waveform_type, waveform_params = self._GaiaDR4_window()
        elif survey == 'GaiaDR4-geq20':
            time, faint, bright, waveform_type, waveform_params = self._GaiaDR4_geq20_window()
        elif survey == 'GaiaDR5':
            time, faint, bright, waveform_type, waveform_params = self._GaiaDR5_window()
        elif survey == 'AllWISE':
            time, faint, bright, waveform_type, waveform_params = self._AllWISE_window()
        elif survey == 'custom':
            raise NotImplementedError     
        # faint star model
        noisy_mag_faint, err_faint = self.make_noisy_mag(faint['mean_mag'], 
                                          faint['noise_level'], 
                                          len(time))
        #bright star model
        noisy_mag_bright, err_bright = self.make_noisy_mag(bright['mean_mag'], 
                                          bright['noise_level'], 
                                          len(time))
     
        # test if I ground truth is provided
        if ground_truth[0] is not None:
            # resample faint star to the ground truth
            time_ground, mag_ground = ground_truth
            mag_ground_to_obs_win = self.resample_from_lc(time_ground, mag_ground, time)
            self.ground_truth_faint = FoldedLightCurve(time=time,
                                        mag=mag_ground_to_obs_win + faint['mean_mag'],
                                        err=np.zeros(len(mag_ground_to_obs_win)),
                                        timescale=timescale,
                                        waveform_type=waveform_type, 
                                        waveform_params=waveform_params)
            self.ground_truth_bright = FoldedLightCurve(time=time,
                                        mag=mag_ground_to_obs_win + bright['mean_mag'],
                                        err=np.zeros(len(mag_ground_to_obs_win)),
                                        timescale=timescale,
                                        waveform_type=waveform_type, 
                                        waveform_params=waveform_params)            

        else:
            mag_ground_to_obs_win = 0.
            self.ground_truth_faint = None
            self.ground_truth_bright = None

        mag_faint = mag_ground_to_obs_win + noisy_mag_faint
        mag_bright = mag_ground_to_obs_win + noisy_mag_bright            
        # create the light-curve objects
        self.faint = FoldedLightCurve(time=time, 
                                      mag=mag_faint, 
                                      err=err_faint, 
                                      timescale=timescale, 
                                        waveform_type=waveform_type, 
                                        waveform_params=waveform_params)
        self.bright = FoldedLightCurve(time=time,
                                        mag=mag_bright,
                                        err=err_bright,
                                        timescale=timescale,
                                        waveform_type=waveform_type, 
                                        waveform_params=waveform_params)   




    def read_observational_window(self, survey_window):
        """
        Load an observational window from a file.

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
            'ASAS-SN-V': 'ASASSN_V.csv',
            'ASAS-SN-g': 'ASASSN_g.csv',
        }
        df = pd.read_csv('../data/' + \
            observational_windows_filenames[survey_window])
        time = df['jd_norm'].to_numpy()
        # makes sure time is ordered
        return np.sort(time)
    
    @staticmethod
    def make_noisy_mag(mean_mag, 
                       noise_level, 
                       n_epochs):  
        """
        Given a mean magnitude, noise level and rms noise, 
        generate a noise baseline for a light-curve.
        """
        _noisy_mag = np.random.normal(loc=mean_mag, 
                                scale=noise_level, 
                                size=n_epochs)

        _err = np.full_like(_noisy_mag, noise_level)
        return _noisy_mag, _err
    
    @staticmethod
    def resample_from_lc(t_in, y_in, t_out):
        """
        Simple function for resampling a light-curve to a new time array.
        """
        # TO DO: include tests for when t_out is outside the range of t_in
        return np.interp(t_out, t_in, y_in)
    
    def _K2_window(self):
        """ 
        Based on a typical light-curve from K2 as in Cody+ 2018AJ....156...71C
        Cadence: 1 observation every 30 minutes
        Timespan: 78 days of observations
        Kp = 8--18 mag 
        10 - .15/100
        15  0.0032
        typical noise level is 1.8mmag at 16th Kp magnitude
        """
        cadence = 30./60./24. 
        timespan = 80
        faint = {'mean_mag': 15.,
                'noise_level': 0.0032}
        bright = {'mean_mag': 10.0,
                'noise_level': 0.0015}
        time = np.arange(0, timespan, cadence)
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.1*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params

    def _CoRoT_window(self):
        """ 
        Based on a typical light-curve from CoRoT as in Cody+ 2014AJ....147...82
        Cadence: 1 observation every 512 s
        Timespan: just over 37 days of observations
        typical rms is 0.01-0.1 at 17th Kp magnitude
        """
        cadence = 512./60./60/24. 
        timespan = 38.68
        faint = {'mean_mag': 15.,
                'noise_level': 0.01}
        bright = {'mean_mag': 12.0,
                'noise_level': 0.001}
        time = np.arange(0, timespan, cadence)
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.1*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
    
    def _TESS_window(self):
        faint = {'mean_mag': 16.,
                'noise_level': 0.01}
        bright = {'mean_mag': 10.0,
                'noise_level': 0.0003}
        time = self.read_observational_window('TESS')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.1*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params

    def _Rubin_window(self):
        raise NotImplementedError
    
    def _ZTF_window(self):
        faint = {'mean_mag': 18.,
                'noise_level': 0.02}
        bright = {'mean_mag': 15.0,
                'noise_level': 0.01}
        time = self.read_observational_window('ZTF')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
        
    def _ASAS_SN_V_window(self):
            # 'ASAS-SN-V':
        faint = {'mean_mag': 15.,
                 'noise_level': 0.04}
        bright = {'mean_mag': 12.0,
                'noise_level': 0.02}
        time = self.read_observational_window('ASAS-SN-V')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params

    def _ASAS_SN_g_window(self):
        # elif kwargs['survey_window'] == 'ASAS-SN-g':
        faint = {'mean_mag': 15.,
                'noise_level': 0.04}
        bright = {'mean_mag': 12.0,
            'noise_level': 0.02}
        time = self.read_observational_window('ASAS-SN-g')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
    
    def _GaiaDR3_window(self):
            # elif kwargs['survey_window'] == 'GaiaDR3':
        faint = {'mean_mag': 17.0,
                'noise_level': 0.0050}
        bright = {'mean_mag': 12.5,
                'noise_level': 0.0008}
        time = self.read_observational_window('GaiaDR3')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
                
    def _GaiaDR4_window(self):
        faint = {'mean_mag': 17.0,
                 'noise_level': 0.0050}
        bright = {'mean_mag': 12.5,
                  'noise_level': 0.0008}
        time = self.read_observational_window('GaiaDR4')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
             
    def _GaiaDR4_geq20_window(self):
        faint = {'mean_mag': 17.0,
                'noise_level': 0.0050}
        bright = {'mean_mag': 12.5,
                'noise_level': 0.008}
        time = self.read_observational_window('GaiaDR4-geq20')     
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
    
    def _GaiaDR5_window(self):        
            # elif kwargs['survey_window'] == 'GaiaDR5':
        faint = {'mean_mag': 17.0,
                'noise_level': 0.0050}
        bright = {'mean_mag': 12.5,
                'noise_level': 0.0008}
        time = self.read_observational_window('GaiaDR5')            
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
    
    def _AllWISE_window(self):
        faint = {'mean_mag': 17.0,
                'noise_level': 0.0050}
        bright = {'mean_mag': 12.5,
                'noise_level': 0.0008}
        time = self.read_observational_window('AllWISE')
        waveform_type = "uneven_savgol"
        waveform_params = {'window': round(.25*len(time)),
                                                    'polynom': 3}
        return time, faint, bright, waveform_type, waveform_params
    
    def __str__(self) -> str:
        return f"Observational window for {self.survey} survey"