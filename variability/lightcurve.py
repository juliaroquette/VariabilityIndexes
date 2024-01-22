
import numpy as np
import pandas as pd
import scipy as sp
from astropy.timeseries import LombScargle

class LightCurve:
    def __init__(self, time, mag, err, mask=None):
        self.time = np.asarray(time, dtype=float)
        self.mag = np.asarray(mag, dtype=float)
        self.err = np.asarray(err, dtype=float)
        if bool(mask):
            self.mask = np.asarray(mask, dtype=bool)
        else:
            self.mask = np.where(np.all(np.isfinite([mag, time, err]), axis=0))[0]
        self.time = time[self.mask]
        self.mag = mag[self.mask]
        self.err = err[self.mask]
        # if isinstance(period, float):
            # self.period = period        
        # elif isinstance(period, bool):
            # if bool(period):
                # print("Estimating a period from the LSP")
                # f, p = self.get_simple_period()
                # self.period = 1./f
            # else:
                # self.period = np.nan
        # else:
            # self.period = np.nan
        
    def N(self):
        """
        returns the number of datapoints in the light curve
        """
        return len(self.mag)

    def std(self):
        """
        from numpy
        """
        return np.std(self.mag)

    def mean(self):
        """
        form numpy
        """
        return np.mean(self.mag)

    def weighted_average(self):
        return np.average(self.mag, weights=1./(self.err**2))

    def median(self):
        return np.median(self.mag)

    def andersonDarling(self):
        return sp.stats.anderson(self.mag)[0]

    def skewness(self):
        return sp.stats.skew(self.mag, nan_policy='omit')

    def kurtosis(self):
        return sp.stats.kurtosis(self.mag)