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
import scipy.stats as ss
from warnings import warn

class VariabilityIndex:
    def __init__(self, lc, **kwargs):
        from variability.lightcurve import LightCurve, FoldedLightCurve
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        if isinstance(lc, FoldedLightCurve):
            self.timescale = self.lc.timescale
        else:
            self.timescale = kwargs.get('timescale', None)
            # defines a folded light-curve object
            if bool(self.timescale):
                self.lc = FoldedLightCurve(lc=self.lc, timescale=self.timescale)
            else:
                warn("No timescale defined. Q_index will not be calculated")
        
        M_percentile = kwargs.get('M_percentile', 10.)
        M_is_flux = kwargs.get('M_is_flux', False)
        
        self.M_index = self.M_index(parent=self,percentile=M_percentile, is_flux=M_is_flux)

        Q_waveform_type = kwargs.get('waveform_type', 'uneven_savgol')
        Q_waveform_params = kwargs.get('waveform_params', {})
        if bool(self.timescale):
            self.Q_index = self.Q_index(parent=self, waveform_type=Q_waveform_type, waveform_params=Q_waveform_params)
        else:
            self.Q_index = None

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
        def __init__(self, parent, waveform_type, waveform_params):
            self.parent = parent
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
            implemented_waveform_types = ['savgol', 'Cody',
                                     'circular_rolling_average_phase',
                                     'circular_rolling_average_number',
                                     'H22', 'uneven_savgol']
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
        def value(self):
            self.parent.lc._get_waveform(waveform_type=self.waveform_type, 
                                  waveform_params=self.waveform_params)
            return (np.std(self.parent.lc.residual)**2 - np.mean(self.parent.lc.err_phased)**2)\
                /(np.std(self.parent.lc.mag_phased)**2 - np.mean(self.parent.lc.err_phased)**2)