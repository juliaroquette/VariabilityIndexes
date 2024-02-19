"""
@juliaroquette: 

This version of class contains only the Q&M variability indexes
from Cody et al. 2014

Last update: Mon Feb 19 2024
"""
import numpy as np
from variability.lightcurve import LightCurve, FoldedLightCurve
from variability.filtering import WaveForm
import scipy.stats as ss
from warnings import warn

class VariabilityIndex:
    def __init__(self, lc, **kargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        
        if 'M_percenile' in kargs.keys():
            M_percenile = kargs['M_percenile']
        else:
            M_percenile = 10.
        if 'M_is_flux' in kargs.keys():
            M_is_flux = kargs['M_is_flux']
        else:
            M_is_flux = False
        
        self.M_index = self.M_index(self, percentile=M_percenile, is_flux=M_is_flux)
        
        if 'timescale' in kargs.keys():
            timescale = kargs['timescale']
        else:
            raise NotImplementedError("automatic timescale not implemented yet")
        if 'waveform_method' in kargs.keys():
            waveform_method = kargs['waveform_method']
        else:
            waveform_method = 'savgol'
        
        self.Q_index = self.Q_index(self, timescale, waveform_method=waveform_method)
        
 
    class M_index:
        def __init__(self, parent, percentile=10., is_flux=False):
            #default
            self._percentile = percentile
            self.is_flux = is_flux
            self.parent = parent

        @property
        def percentile(self):
            """ 
            Percentile used to calculate the M-index 
            """
            return self._percentile
	
        @percentile.setter
        def percentile(self, new_percentile):
            if (new_percentile > 0) and (new_percentile < 49.):
                self._percentile = new_percentile
            else:
                print("Please enter a valid percentile (between 0. and 49.)")
       
        @percentile.deleter
        def percentile(self):
            del self._percentile

        def get_percentile_mask(self):
            return (self.parent.lc.mag <= \
                                np.percentile(self.parent.lc.mag, self._percentile))\
                               | (self.parent.lc.mag >= \
                                  np.percentile(self.parent.lc.mag, 100 - self._percentile))
                               
        @property
        def value(self):
            return (1 - 2*int(self.is_flux))*(np.mean(self.parent.lc.mag[self.get_percentile_mask()]) - self.parent.lc.median)/self.parent.lc.std
    
    class Q_index:
        def __init__(self, parent, timescale, waveform_method='savgol'):
            self.parent = parent
            self._timescale = timescale
            self.waveform_method = waveform_method
            # defines a folded light-curve object
            self.lc_p = FoldedLightCurve(lc=self.parent.lc, timescale=self.timescale)
            # estimates the residual magnitude
            self.residual_mag = WaveForm(self.lc_p, method=self.waveform_method).residual_magnitude()
            
        @property
        def timescale(self):
            return self._timescale
        
        @timescale.setter
        def timescale(self, new_timescale):
            if new_timescale > 0:
                self._timescale = new_timescale
            else:
                print("Please enter a valid _positive_ timescale")
        
        @timescale.deleter
        def timescale(self):
            del self._timescale

        @property
        def waveform_method(self):
            return self._waveform_method
        
        @waveform_method.setter
        def waveform_method(self, new_waveform_method):
            implemente_waveforms = ['savgol',
                                       'circular_rolling_average_number',
                                       'circular_rolling_average_phase',
                                       'H22', 'Cody', 'uneven_savgol'
                                       ]
            if new_waveform_method in implemente_waveforms:
                self._waveform_method = new_waveform_method
            else:
                print("Please enter a valid method:", implemente_waveforms)

        @waveform_method.deleter
        def waveform_method(self):
            del self._waveform_method

        @property
        def value(self):
            """
            calculates the Q-index
            """
            return (np.std(self.residual_mag)**2 - np.mean(self.lc_p.err_phased)**2)\
                /(np.std(self.lc_p.mag_phased)**2 - np.mean(self.lc_p.err_phased)**2)
