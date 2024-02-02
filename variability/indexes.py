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
import scipy.stats as ss

class VariabilityIndex:
    def __init__(self, lc, **kwargs):
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
        
        self.M_index = self.M_index(self, percenile=M_percenile, is_flux=M_is_flux)
        
        #IS THIS THE PLACE TO ADD THE FILTERING?
 
    class M_index:
        def __init__(self, parent, percentile=10., percentile=False):
            #default
            self._percentile = percentile
            self.is_flux = percentile
            self.parent = parent

        @property
        def percentile(self):
            
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
    
    @property    
    def Abbe(self):
        print('Not implemented yet')
        return None

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
    

    def Q_index(self, lc_phased):
        """
        Calculates the Q-index which measures the level of periodicity in the light-curve.

        Parameters:
        mag_phased (array-like): Array of phase-folded magnitudes for the "raw" light-curve.
        residual_mag (array-like): Array of phase-folded residual magnitudes after waveform-subtraction
        err_phased (array-like): Array of errors for the phase-folded light-curve.

        Returns:
        - Q-index float: The calculated Q-index.

        """
        if not hasattr(my_instance, 'timescale'):
        # raise 
            raise TypeError("lc must be an instance of FoldedLightCurve with a timescale attribute")
        else:
            return (np.std(residual_mag)**2 - np.mean(err_phased)**2)/(np.std(mag_phased)**2 - np.mean(err_phased)**2)
    
