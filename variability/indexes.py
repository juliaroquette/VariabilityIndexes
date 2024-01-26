"""
Variability indexes
"""
import numpy as np
from lightcurve import LightCurve

class VariabilityIndex(LightCurve):
    def __init__(self):
        super().__init__()
        
    def M_index(self):
        pass
    
    def Q_index(self):
        pass
    
    def Abbe(self):
        pass
    
    def stetsonK(self):
        """
        Calcula Stetson index K
        """
        residual = np.sqrt(len(self.mag)/(len(self.mag) - 1)
                           )*(self.mag - np.average(self.mag, weights=1./(
                               self.err**2)))/self.err
        return np.sum(np.fabs(residual)
                      )/np.sqrt(len(self.mag)*np.sum(residual**2))

    def ShapiroWilk(self):
        return sp.stats.shapiro(self.mag)[0]

    def mad(self):
        """
        median absolute deviation
        """
        return sp.stats.median_abs_deviation(self.mag, nan_policy='omit')

    def chisquare(self):
        return sp.stats.chisquare(self.mag)[0]

    def reducedChiSquare(self):
        return np.sum((self.mag - np.average(self.mag,
                                             weights=1./(self.err**2))
                       )**2/self.err**2)/np.count_nonzero(
                           ~np.isnan(self.mag)) - 1

    def IQR(self):
        """
        inter-quartile range
        """
        return sp.stats.iqr(self.mag)

    def RoMS(self):
        """
        Robust-Median Statistics (RoMS)
        """
        return np.sum(np.fabs(self.mag - np.nanmedian(self.mag)
                              )/self.err)/(len(self.mag) - 1.)

    def normalisedExcessVariance(self):
        return np.sum((self.mag - np.nanmean(self.mag))**2 - self.err**2
                      )/len(self.mag)/np.nanmean(self.mag)**2

    def Lag1AutoCorr(self):
        return np.sum((self.mag[:-1] - np.mean(self.mag)) *
                      (self.mag[1:] - np.mean(self.mag)))/np.sum(
                          (self.mag - np.mean(self.mag))**2)

    def VonNeumann(self):
        return np.sum((self.mag[1:] - self.mag[:-1])/(
            len(self.mag) - 1))/np.sum((self.mag - np.mean(self.mag)
                                        )/(len(self.mag) - 1))
    def norm_ptp(self):
        return (max(self.mag - self.err) - 
                min(self.mag + self.err))/(max(self.mag - self.err) 
                                           + min(self.mag + self.err))    
 