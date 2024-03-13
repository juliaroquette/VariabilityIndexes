"""
@juliaroquette: 

This version of class contains only the Q&M variability indexes
from Cody et al. 2014

Last update: Mon Feb 19 2024
"""

import numpy as np
from variability.lightcurve import LightCurve, FoldedLightCurve
import warnings

class VariabilityIndex:
    def __init__(self, lc, **kwargs):
        from variability.lightcurve import LightCurve, FoldedLightCurve
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        
        # calculate M-index
        M_percentile = kwargs.get('M_percentile', 10.)
        M_is_flux = kwargs.get('M_is_flux', False)
        self.M_index = self.M_index(parent=self,percentile=M_percentile, is_flux=M_is_flux)

        # calculate Q-index
        if not isinstance(self.lc, FoldedLightCurve):
            warnings.warn("Q-index is only available for folded light-curves")
            self.Q_index = None
        else:
            Q_waveform_type = kwargs.get('waveform_type', 'uneven_savgol')
            Q_waveform_params = kwargs.get('waveform_params', {})
            self.Q_index = self.Q_index(parent=self, waveform_type=Q_waveform_type, waveform_params=Q_waveform_params)
        if 'timescale' in kwargs:
            warnings.warn('timescale is not a valid parameter for VariabilityIndex - please use the timescale attribute of the light curve object instead')

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
            
    @property
    def timescale(self):
        return self._timescale
    
    @timescale.setter
    def timescale(self, new_timescale):
        self._timescale = new_timescale
        if isinstance(self.lc, FoldedLightCurve):
            self.lc.timescale = new_timescale
            # Recalculate Q_index
            self.lc._get_waveform
            # self.Q_index.parent = self.parent.lc._get_waveform(waveform_type=self.waveform_type, 
                                #   waveform_params=self.waveform_params)
            # self.Q_index.value = self.Q_index.value()
    
    class Q_index:
        def __init__(self, parent, waveform_type, waveform_params):
            self.parent = parent
            print(parent.lc.timescale)
            self._waveform_type = waveform_type
            self._waveform_params = waveform_params

        @property
        def waveform_type(self):
            """ 
            Waveform estimator method used to calculate the Q-index 
            """
            return self._waveform_type
            
        @waveform_type.setter
        def waveform_type(self, new_waveform_type):
            implemented_waveform_types = ['circular_rolling_average_phase',
                                     'circular_rolling_average_number',
                                     'uneven_savgol']
            if new_waveform_type in implemented_waveform_types:
                self._waveform_type = new_waveform_type
            else:
                raise ValueError("Please enter a valid waveform type {0}".format(implemented_waveform_types))
        
        @property
        def waveform_params(self):
            """ 
            Parameters used to calculate the Q-index 
            """
            return self._waveform_params
        
        @waveform_params.setter
        def waveform_params(self, new_waveform_params):
            self._waveform_params = new_waveform_params
            
        @property
        def residual(self):
            self.parent.lc._get_waveform()
            return self.parent.lc.residual
           
        @property
        def value(self):
            return (np.std(self.residual)**2 - np.mean(self.parent.lc.err_phased)**2)\
                /(np.std(self.parent.lc.mag_phased)**2 - np.mean(self.parent.lc.err_phased)**2)