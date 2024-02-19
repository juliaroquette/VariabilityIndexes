"""
@juliaroquette: 

This version of class contains only the Q&M variability indexes
from Cody et al. 2014

Last update: Mon Feb 19 2024
"""

import numpy as np
from variability.lightcurve import LightCurve, FoldedLightCurve
from variability.filtering import WaveForm

class VariabilityIndex:
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        
        M_percentile = kwargs.get('M_percentile', 10.)
        M_is_flux = kwargs.get('M_is_flux', False)
        
        self.M_index = self.M_index(parent=self,percentile=M_percentile, is_flux=M_is_flux)
       
        timescale = kwargs.get('timescale', NotImplementedError("automatic timescale not implemented yet"))
        waveform_method = kwargs.get('waveform_method', 'uneven_savgol')
        # print('Using', waveform_method, 'method')
    
        self.Q_index = self.Q_index(parent=self, timescale=timescale, waveform_method=waveform_method)
        
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
                raise ValueError("Please enter a valid percentile (between 0. and 49.)")

        @property
        def get_percentile_mask(self):
            return (self.parent.lc.mag <= \
                                np.percentile(self.parent.lc.mag, self._percentile))\
                               | (self.parent.lc.mag >= \
                                  np.percentile(self.parent.lc.mag, 100 - self._percentile))
                               
        @property
        def value(self):
            return (1 - 2*int(self.is_flux))*(np.mean(self.parent.lc.mag[self.get_percentile_mask]) - self.parent.lc.median)/self.parent.lc.std
    
    class Q_index:
        def __init__(self, parent, timescale, waveform_method='savgol'):
            self.parent = parent
            self._timescale = timescale
            self._waveform_method = waveform_method

        @property        
        def get_residual(self):
            # defines a folded light-curve object
            self.lc_p = FoldedLightCurve(lc=self.parent.lc, timescale=self._timescale)
            # estimates the residual magnitude
            return WaveForm(self.lc_p, method=self._waveform_method).residual_magnitude()
            
        @property
        def timescale(self):
            return self._timescale
        
        @timescale.setter
        def timescale(self, new_timescale):
            if new_timescale > 0.:
                self._timescale = new_timescale
                print(self._timescale)
            else:
                raise ValueError("Please enter a valid _positive_ timescale")
        
        @property
        def waveform_method(self):
            return self._waveform_method
        
        @waveform_method.setter
        def waveform_method(self, new_waveform_method):
            implemented_waveforms = ['savgol',
                                       'circular_rolling_average_number',
                                       'circular_rolling_average_phase',
                                       'H22', 'Cody', 'uneven_savgol'
                                       ]
            if new_waveform_method in implemented_waveforms:
                self._waveform_method = new_waveform_method
            else:
                raise ValueError("Please enter a valid method:", implemented_waveforms)

        @property
        def value(self):
            """
            calculates the Q-index
            """
            return (np.std(self.get_residual)**2 - np.mean(self.lc_p.err_phased)**2)\
                /(np.std(self.lc_p.mag_phased)**2 - np.mean(self.lc_p.err_phased)**2)
