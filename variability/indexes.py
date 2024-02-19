"""
@juliaroquette: 
Class that calculates a series of variability indexes for
a given light-curve.

__This currently includes__
- M_index (Cody et al. 2014)
- Shapiro-Wilk
- Chisquare
- reducedChiSquare
- IQR
- RoMS
- andersonDarling
- skewness
- kurtosis
- normalisedExcessVariance
- Lag1AutoCorr
- VonNeumann
- norm_ptp
- mad


__ Under implementation __
- stetsonK
- Abbe
- Q_index

__TO DO__
- Add documenation to each method
- Add references 


Last update: 02-02-2024
"""
import numpy as np
from variability.lightcurve import LightCurve, FoldedLightCurve
from variability.filtering import WaveForm
import scipy.stats as ss
from warnings import warn

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
    
    @property    
    def Abbe(self):
        raise NotImplementedError("This hasn't been implemented yet.")

    # this is bugged     
    # @property   
    def stetsonK(self):
        """
        Calcula Stetson index K
        """
        print('odl implementation has a bug')
        return None    
    #     residual = np.sqrt(self.lc.N)/(self.lc.N - 1.)*\
    #         (self.mag - self.lc.weighted_average)/self.err
    #     return np.sum(np.fabs(residual)
    #                   )/np.sqrt(self.lc.N*np.sum(residual**2))

    @property
    def ShapiroWilk(self):
        return ss.shapiro(self.lc.mag)[0]

    @property
    def mad(self):
        """
        median absolute deviation
        """
        return ss.median_abs_deviation(self.lc.mag, nan_policy='omit')

    @property
    def chisquare(self):
        return ss.chisquare(self.lc.mag)[0]

    @property
    def reducedChiSquare(self):
        return np.sum((self.lc.mag - self.lc.weighted_average)**2/self.lc.err**2)/np.count_nonzero(
                           ~np.isnan(self.lc.mag)) - 1
    
    @property
    def IQR(self):
        """
        inter-quartile range
        """
        return ss.iqr(self.lc.mag)
    
    @property
    def RoMS(self):
        """
        Robust-Median Statistics (RoMS)
        """
        return np.sum(np.fabs(self.lc.mag - self.lc.median
                              )/self.lc.err)/(self.lc.N - 1.)
    
    @property
    def normalisedExcessVariance(self):
        return np.sum((self.lc.mag - np.nanmean(self.lc.mag))**2 - self.lc.err**2
                      )/len(self.lc.mag)/np.nanmean(self.lc.mag)**2
    
    @property
    def Lag1AutoCorr(self):
        return np.sum((self.mag[:-1] - self.lc.mean) *
                      (self.mag[1:] - self.lc.mean))/np.sum(
                          (self.mag - self.lc.mean)**2)
    
    @property
    def VonNeumann(self):
        return np.sum((self.lc.mag[1:] - self.lc.mag[:-1])/(self.lc.N - 1))/np.sum((self.lc.mag - 
                                           self.lc.mean)/(self.lc.N - 1))

    @property
    def norm_ptp(self):
        return (max(self.lc.mag - self.lc.err) - 
                min(self.lc.mag + self.lc.err))/(max(self.lc.mag - self.lc.err) 
                                           + min(self.lc.mag + self.lc.err))    

    @property
    def andersonDarling(self):
        return ss.anderson(self.lc.mag)[0]

    @property
    def skewness(self):
        return ss.skew(self.lc.mag, nan_policy='omit')

    @property
    def kurtosis(self):
        return ss.kurtosis(self.lc.mag)


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
            return WaveForm(self.lc_p, waveform_type=self._waveform_method).residual_magnitude()
            
        @property
        def timescale(self):
            return self._timescale
        
        @timescale.setter
        def timescale(self, new_timescale):
            if new_timescale > 0.:
                self._timescale = new_timescale
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
