"""
@juliaroquette: 
Class that calculates a series of variability indexes for
a given light-curve.

__This currently includes__
- asymmetry_index (Cody et al. 2014)
- shapiro_wilk
- chi_square
- reduced_chi_square
- iqr
- roms
- anderson_darling
- skewness
- kurtosis
- normalised_excess_variance
- lag1_auto_corr
- von_neumann (abbe)
- norm_ptp
- mad
- periodicity_index

-> Add:
- gaia_AG_proxy

- double check against the Gaia implementation 



__ Under implementation __
- stetsonK

__TO DO__
- Add documenation to each method
- Add references 
- Add Saunders
- Add SL
- Add periodic/aperiodic timescales as an index?


Last update: 02-02-2024
"""
import inspect
import numpy as np
import scipy.stats as ss
from warnings import warn
from variability.lightcurve import LightCurve, FoldedLightCurve


class VariabilityIndex:
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self._params = kwargs.copy()


    @property
    def periodicity_index(self):
        if (self.lc.n_epochs > self.lc.min_epochs) and isinstance(self.lc, FoldedLightCurve):
            return PeriodicityIndex(parent=self).value
        else:
            # warn("Q-index is only available for folded light-curves")
            return None

    @property
    def asymmetry_index(self):
        # calculate M-index
        M_percentile = self._params.get('M_percentile', 10.)
        M_is_flux = self._params.get('M_is_flux', False)
        if self.lc.n_epochs > self.lc.min_epochs:
            return AsymmetryIndex(parent=self,percentile=M_percentile, is_flux=M_is_flux).value
        else:
            return None

    # @property    
    def Abbe(self):
        """
        Calculate Abbe value as in Mowlavi 2014A%26A...568A..78M
        https://www.aanda.org/articles/aa/full_html/2014/08/aa22648-13/aa22648-13.html
        """
        if self.lc.n_epochs > self.lc.min_epochs:
            return self.lc.n_epochs* np.sum((self.lc.mag[1:] - self.lc.mag[:-1])**2) /\
                2 / np.sum((self.lc.mag - self.lc.mean)**2) / (self.lc.n_epochs- 1)
        else:
            return None

    # this is bugged     
    # @property   
    def stetsonK(self):
        """
        Calculate Stetson index K
        """
        print('odl implementation has a bug')
        return None    
    #     residual = np.sqrt(self.lc.n_epochs)/(self.lc.n_epochs- 1.)*\
    #         (self.mag - self.lc.weighted_average)/self.err
    #     return np.sum(np.fabs(residual)
    #                   )/np.sqrt(self.lc.n_epochs*np.sum(residual**2))

    @property
    def shapiro_wilk(self):
        if self.lc.n_epochs > self.lc.min_epochs:
            return ss.shapiro(self.lc.mag)[0]
        else:
            return None

    @property
    def mad(self):
        """
        median absolute deviation
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return ss.median_abs_deviation(self.lc.mag, nan_policy='omit')

    @property
    def chi_square(self):
        """
        Raw Chi-square value
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return np.sum((self.lc.mag - self.lc.weighted_average)**2 / self.lc.err**2)

    @property
    def reduced_chi_square(self):
        """
        Reduced Chi-square value:
        raw chi-square divided by the number of degrees of freedom (N-1)
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return self.chi_square/(np.count_nonzero(
                           ~np.isnan(self.lc.mag)) - 1)
    
    @property
    def iqr(self):
        """
        inter-quartile range
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return ss.iqr(self.lc.mag)
    
    @property
    def roms(self):
        """
        Robust-Median Statistics (RoMS)
        """
        if self.lc.n_epochs > self.lc.min_epochs:
            return np.sum(np.abs(self.lc.mag - np.median(self.lc.mag))/self.lc.err)/(self.lc.n_epochs- 1)
        else:
            return None

    @property
    def normalised_excess_variance(self):
        if self.lc.n_epochs > self.lc.min_epochs:
            return (self.lc.std**2 - self.lc.mean_err**2)/self.lc.mean**2
        else:
            return None

    @property
    def lag1_auto_corr(self):
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return np.sum((self.lc.mag[:-1] - self.lc.mean) *
                      (self.lc.mag[1:] - self.lc.mean))/np.sum(
                          (self.lc.mag - self.lc.mean)**2)
    

    @property
    def norm_ptp(self):
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return (max(self.lc.mag - self.lc.err) - 
                min(self.lc.mag + self.lc.err))/(max(self.lc.mag - self.lc.err) 
                                           + min(self.lc.mag + self.lc.err))    

    @property
    def anderson_darling(self):
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return ss.anderson(self.lc.mag)[0]

    @property
    def skewness(self):
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return ss.skew(self.lc.mag, nan_policy='omit')

    @property
    def kurtosis(self):
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return ss.kurtosis(self.lc.mag)

    @property
    def ptp_5(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 5% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return  self.ptp_perc(percentile=5.)

    @property
    def ptp_10(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 10% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return  self.ptp_perc(percentile=10.)

    @property
    def ptp_20(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 20% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        if self.lc.n_epochs < self.lc.min_epochs:
            return None
        else:
            return  self.ptp_perc(percentile=20.)

    def ptp_perc(self, percentile=10.):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the `percentile`% outermost tails of the distribution.

        Args:
            percentile (float, optional): Percentile to use for the tails. Defaults to 10..

        Returns:
            float: Peak-to-peak amplitude.
        """
        if (percentile <= 0.) or (percentile >= 49.):
            raise ValueError("Please enter a valid percentile (between 0. and 49.)")
        p = percentile/100.
        tail = round(p * self.lc.n_epochs)
        return  np.median(np.sort(self.lc.mag)[-tail:]) - np.median(np.sort(self.lc.mag)[:tail])               
        

    def _list_properties(self):
        """
        list properties of the class LightCurve
        """
        property_names = [name for name, value in inspect.getmembers(self.__class__, lambda o: isinstance(o, property))]
        return property_names    
        
    def __str__(self):
        return f'A VariabilityIndex instance has the following properties: {repr(self._list_properties())}'
    
    @classmethod
    def suppress_warnings_globally(cls):
        """
        This is class method that enable to suppress warnings globally
        for FoldedLightCurve instances.
        
        usage:
        FoldedLightCurve.suppress_warnings_globally()
        """
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

class AsymmetryIndex:
    def __init__(self, parent, percentile=10., is_flux=False):
        self.parent = parent
        self._percentile = float(percentile)
        self.is_flux = bool(is_flux)

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
        return (1. - 2*int(self.is_flux))*(np.mean(self.parent.lc.mag[self.get_percentile_mask]) - self.parent.lc.median)/self.parent.lc.std

class PeriodicityIndex:
    def __init__(self, parent):
        self.parent = parent

    @property
    def value(self):
        # print(self._waveform_type)
        # self.parent.lc._get_waveform()
        return (np.std(self.parent.lc.residual, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)\
            /(np.std(self.parent.lc.mag_phased, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)

def gaia_AG_proxy(phot_g_mean_flux, phot_g_mean_flux_error, phot_g_n_obs):
    """
    Following Mowlavi et al. 2021 (https://ui.adsabs.harvard.edu/#abs/2021A%26A...648A..44M)
    this function calculates a proxy for the variability of Gaia sources using the uncertainty of
    the Gaia G-band fluxes.
    This is provided in equation (2) of the paper, and is given by:
    
    AG = sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux

    For constant stars this is approximately the standard deviation of 
    G light curves due to noise and uncalibrated systematic effects. 
    _summary_

    Args:
        phot_g_mean_flux (_type_): _description_
        phot_g_mean_flux_error (_type_): _description_
        phot_g_n_obs (_type_): _description_
    """
    return np.sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux