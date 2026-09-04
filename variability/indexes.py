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
- Abbe (von Neumann ratio)
- norm_ptp
- mad
- periodicity_index
- stetsonK
- weighted_std
- saunders_norm
- lafler_kinman
- string_length

-> Add:
- gaia_AG_proxy

- double check against the Gaia implementation


__TO DO__
- Add references

__Resolved design questions__
- Periodic/aperiodic timescales are *not* indexes here: they live in
  `variability.timescales.TimeScale` instead, kept as a separate class
  because `TimeScale.get_peak_Q_periodicity` already depends on
  `VariabilityIndex` (folding the light curve at candidate periods and
  reading `periodicity_index`) - merging them back in would create a
  circular import.


Last update: 04-09-2026
"""
import inspect
import numpy as np
import scipy.stats as ss
from warnings import warn
from variability.lightcurve import LightCurve, FoldedLightCurve
import functools

def min_epochs_property(func):
    """
    Decorator defined to enforce the minimum number of epochs globally
    It returns None if lc.n_epochs < self.min_epochs; 
    else run the function.
    returned values are properties
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        if getattr(self.lc, "n_epochs", 0) < self.min_epochs:
            return None
        return func(self, *args, **kwargs)
    return property(wrapped)

class _tagged_property(property):
    """
    A property that carries a marker for folded-only use.
    This is used to decorate properties of VariabilityIndex that
    should only exists when a FoldedLightCurve is used."""
    def __init__(self, fget=None, *, folded_only=False, **kwargs):
        super().__init__(fget, **kwargs)
        self.folded_only = folded_only

def folded_property(func):
    """
    Decorator for the FoldedLightCurve VariabilityIndexes:
    returns None unless lc is folded
    while also enforcing n_epochs >= min_epochs.
    """
    @functools.wraps(func)
    def _f(self, *args, **kwargs):
        # global min-epochs policy
        if getattr(self.lc, "n_epochs", 0) < self.min_epochs:
            return None
        # folded-only policy
        if not getattr(self, "_is_folded", False):
            return None
        return func(self, *args, **kwargs)
    return _tagged_property(_f, folded_only=True)

class VariabilityIndex:
    _suppress_warnings = False  # class variable to control warning suppression globally
    def __init__(self, lc, min_epochs=5, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        self._is_folded = isinstance(lc, FoldedLightCurve)
        self._params = kwargs.copy()
        self.min_epochs = int(min_epochs)

    @min_epochs_property
    def std(self):
        """
        Returns the standard deviation of the magnitude values.

        Returns:
            float: Standard deviation.
        """
        return np.std(self.lc.mag,
                    # ddof=1 makes sure std is bias-corrects
                        # this means N-1 is used as the denominator rather than N
                    ddof=1)

    @min_epochs_property
    def weighted_average(self):
        """
        Returns the weighted average of the magnitude values.

        Returns:
            float: Weighted average.
        """
        # this avoids division by zero:
        weights = np.clip(1./(self.lc.err**2), 1e-12, None)
        return np.average(self.lc.mag, weights=weights)

    @min_epochs_property
    def weighted_std(self):
        """
        Returns the weighted standard deviation of the magnitude values,
        using the same per-epoch weights (1/err^2) as `weighted_average`.

        Returns:
            float: Weighted standard deviation.
        """
        # this avoids division by zero:
        weights = np.clip(1./(self.lc.err**2), 1e-12, None)
        variance = np.average((self.lc.mag - self.weighted_average)**2, weights=weights)
        # bias-correct with N/(N-1), matching the ddof=1 convention used by `std`
        return np.sqrt(self.lc.n_epochs / (self.lc.n_epochs - 1.) * variance)

    @min_epochs_property
    def signal_to_noise(self):
        """
        Returns the signal-to-noise ratio of the light curve
        defined as the ratio between the standard deviation of the magnitudes
        and the average uncertainty.

        Returns:
            float: Signal-to-noise ratio.
        """
        return self.std/self.lc.mean_err

    @min_epochs_property
    def shapiro_wilk(self):
        """
        Shapiro-Wilk statistic (Shapiro & Wilk 1965, Biometrika, 52, 591)
        testing how consistent the magnitude distribution is with a
        Gaussian. Used here as a ranking feature rather than a formal
        statistical test (no p-value is returned). See
        `scipy.stats.shapiro`.

        Expected behavior:
        - For Gaussian noise (no variability): W~1.
        - For symmetric variability: W<=1.
        - For highly asymmetric variability: W<<1.
        """
        return ss.shapiro(self.lc.mag)[0]

    @folded_property
    def periodicity_index(self):
        """
        Cody+2014 Q-index: the periodicity index of the folded light-curve,
        see `PeriodicityIndex` below.

        Expected behavior:
        - For strictly periodic sources: Q~0.
        - For aperiodic sources: Q~1.

        Reference: Cody et al. (2014), AJ, 147, 82.
        """
        return PeriodicityIndex(parent=self).value

    @min_epochs_property
    def asymmetry_index(self):
        """
        Cody+2014 M-index: asymmetry of the magnitude distribution
        between its top/bottom `M_percentile` (default 10%, set via the
        `M_percentile` kwarg passed to `VariabilityIndex`) and the median,
        see `AsymmetryIndex` below. Pass `M_is_flux=True` if `lc.mag` is
        actually in flux rather than magnitude units, to flip the sign
        convention accordingly.

        Expected behavior:
        - For symmetric variability: M~0.
        - For dimming-dominated variability: M>>0.
        - For brightening-dominated variability: M<<0.

        Reference: Cody et al. (2014), AJ, 147, 82.
        """
        # calculate M-index
        M_percentile = self._params.get('M_percentile', 10.)
        M_is_flux = self._params.get('M_is_flux', False)
        return AsymmetryIndex(parent=self,percentile=M_percentile, is_flux=M_is_flux).value

    @min_epochs_property
    def Abbe(self):
        """
        Abbe value / von Neumann ratio (Mowlavi 2014, A&A, 568, A78), a
        test for serial correlation between consecutive epochs, closely
        related to `lag1_auto_corr` (Abbe ~= 1 - lag1_auto_corr for
        large N).

        Expected behavior:
        - For Gaussian noise (no serial correlation): Abbe ~ 1.
        - For slowly-varying (e.g. periodic, smoothly trending) sources,
          where consecutive epochs are similar: Abbe < 1.
        """
        return self.lc.n_epochs* np.sum((self.lc.mag[1:] - self.lc.mag[:-1])**2) /\
            2 / np.sum((self.lc.mag - self.lc.mean)**2) / (self.lc.n_epochs- 1)

    @min_epochs_property
    def stetsonK(self):
        """
        Stetson K index (Stetson 1996, PASP, 108, 851).

        A kurtosis-like statistic built from the per-epoch residuals
        normalised by their photometric uncertainty. It measures how
        "peaked" or "flat" the magnitude distribution is relative to a
        Gaussian, independently of the residuals' overall amplitude.
        """
        delta = np.sqrt(self.lc.n_epochs / (self.lc.n_epochs - 1.)) * \
            (self.lc.mag - self.weighted_average) / self.lc.err
        return np.sum(np.fabs(delta)) / np.sqrt(self.lc.n_epochs * np.sum(delta**2))


    @min_epochs_property
    def mad(self):
        """
        Median absolute deviation of the magnitudes. See
        `scipy.stats.median_abs_deviation`.

        Expected behavior:
        - For Gaussian noise: 1.4826*MAD ~ mean photometric uncertainty.
        - For symmetric variability: MAD > mean photometric uncertainty.
        - Insensitive to outliers (and thus to real asymmetric variability).

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.3.
        """
        return ss.median_abs_deviation(self.lc.mag, nan_policy='omit')

    @min_epochs_property
    def chi_square(self):
        """
        Raw chi-square value testing consistency with a constant
        (non-variable) source, using `weighted_average` as the reference
        magnitude.

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.1.
        """
        return np.sum((self.lc.mag - self.weighted_average)**2 / self.lc.err**2)

    @min_epochs_property
    def reduced_chi_square(self):
        """
        Reduced chi-square value: `chi_square` divided by the number of
        degrees of freedom (N-1).

        Expected behavior:
        - For Gaussian noise: ~1.
        - For real variability: >1, growing with variability amplitude.

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.1.
        """
        return self.chi_square/(np.count_nonzero(
                           ~np.isnan(self.lc.mag)) - 1)

    @min_epochs_property
    def iqr(self):
        """
        Inter-quartile range of the magnitudes (Q3-Q1). See
        `scipy.stats.iqr`. Related to `std` and `mad`
        (IQR ~ 0.761*MAD for a Gaussian).

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.4.
        """
        return ss.iqr(self.lc.mag)

    @min_epochs_property
    def roms(self):
        """
        Robust Median Statistics (RoMS): mean absolute deviation from the
        median, normalised by each epoch's photometric uncertainty.

        Expected behavior:
        - For Gaussian noise: ~1.
        - For real variables: >1.

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.5.
        """
        return np.sum(np.abs(self.lc.mag - np.median(self.lc.mag))/self.lc.err)/(self.lc.n_epochs- 1)


    @min_epochs_property
    def normalised_excess_variance(self):
        """
        Normalised excess variance: the variance in excess of the mean
        photometric noise, normalised by the mean magnitude.

        Expected behavior:
        - For Gaussian noise: ~0.
        - For symmetric variability: >0.
        - For asymmetric variability: >>0.

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.6.
        """
        return (self.std**2 - self.lc.mean_err**2)/self.lc.mean**2

    @min_epochs_property
    def lag1_auto_corr(self):
        """
        First-order (lag-1) autocorrelation coefficient of the magnitudes:
        how correlated the light curve is with itself at a one-epoch lag.

        Expected behavior:
        - For Gaussian noise: consecutive points are uncorrelated, ~0.
        - For slowly-varying (e.g. periodic) sources: >0.
        - For monotonic long-term trends: approaches 1.
        """
        if self.lc.n_epochs < self.min_epochs:
            return None
        else:
            return np.sum((self.lc.mag[:-1] - self.lc.mean) *
                      (self.lc.mag[1:] - self.lc.mean))/np.sum(
                          (self.lc.mag - self.lc.mean)**2)


    @min_epochs_property
    def norm_ptp(self):
        """
        Normalised peak-to-peak amplitude, accounting for the photometric
        uncertainty at the extreme epochs.

        Reference: Sokolovsky et al. (2017), MNRAS, 464, 274, Sec. 2.7.
        """
        return (max(self.lc.mag - self.lc.err) -
            min(self.lc.mag + self.lc.err))/(max(self.lc.mag - self.lc.err)
                                        + min(self.lc.mag + self.lc.err))

    @min_epochs_property
    def anderson_darling(self):
        """
        Anderson-Darling statistic testing consistency of the magnitude
        distribution with a Gaussian, weighting the tails of the
        distribution more heavily than `shapiro_wilk`. See
        `scipy.stats.anderson`.

        Expected behavior:
        - For Gaussian noise: A^2~0.
        - For symmetric variability: A^2>0.
        - For asymmetric variability: A^2>>0.
        """
        return ss.anderson(self.lc.mag)[0]

    @min_epochs_property
    def skewness(self):
        """
        Skewness of the magnitude distribution. See `scipy.stats.skew`.

        Expected behavior:
        - For Gaussian noise: ~0.
        - For symmetric variability: ~0.
        - For asymmetric variability: sign follows the same dimming/
          brightening convention as `asymmetry_index` (positive for
          dimming-dominated, negative for brightening-dominated).
        """
        return ss.skew(self.lc.mag, nan_policy='omit')

    @min_epochs_property
    def kurtosis(self):
        """
        Excess kurtosis of the magnitude distribution (Fisher's
        definition, Gaussian = 0). See `scipy.stats.kurtosis`.

        Expected behavior:
        - For Gaussian noise: ~0.
        - For broad, sinusoid-like variability: <0 (platykurtic).
        - For variability dominated by rare sharp extrema (eclipses,
          dips, bursts): >0 (leptokurtic).
        """
        return ss.kurtosis(self.lc.mag)

    @min_epochs_property
    def ptp_5(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 5% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return  self.ptp_perc(percentile=5.)

    @min_epochs_property
    def ptp_10(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 10% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return  self.ptp_perc(percentile=10.)

    @min_epochs_property
    def ptp_20(self):
        """
        Returns the peak-to-peak amplitude of the magnitude values.
        This is defined as the difference between the median values for the datapoints 
        in the 20% outermost tails of the distribution.

        Returns:
            float: Peak-to-peak amplitude.
        """
        return self.ptp_perc(percentile=20.)

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
        # it can't get a tail if there are not enough epochs
        if self.lc.n_epochs < 3:
            return None
        tail = int(np.floor(percentile * self.lc.n_epochs / 100.))
        if tail > 1:
            mag_sorted = np.sort(self.lc.mag)
            return  np.median(mag_sorted[-tail:]) - np.median(mag_sorted[:tail])
        else:
            # warn("Not enough epochs to calculate the peak-to-peak amplitude for the given percentile")
            return None

    @folded_property
    def residual_mean(self):
        """
        Average of the residuals of the folded light-curve
        """
        return np.mean(self.lc.residual)
    
    @folded_property
    def residual_std(self):
        """
        Standard deviation of the residuals of the folded light-curve
        """
        return np.std(self.lc.residual, ddof=1)
    
    @folded_property
    def qm_class(self):
        """
        Cody+2014 QM classification, from `asymmetry_index` (M) and
        `periodicity_index` (Q). See `cody14_q_m_classifier` below.
        """
        return cody14_q_m_classifier(self.asymmetry_index, self.periodicity_index)

    @folded_property
    def saunders_norm(self):
        """
        Saunders statistic (Saunders et al. 2006, Astronomische
        Nachrichten, 327, 783) diagnosing how clumpy the phase coverage
        of the folded light-curve is, relative to equally-spaced phase
        sampling. See `FoldedLightCurve.saunders_norm` for the definition.

        Expected behavior:
        - 0 for perfectly uniform phase coverage
        - ~1 for random uniform phase coverage
        - >1 for clumpy/poor phase coverage
        """
        return self.lc.saunders_norm

    @folded_property
    def lafler_kinman(self):
        """
        Lafler-Kinman string-length statistic (Lafler & Kinman 1965,
        ApJS, 11, 216) of the folded light-curve. Since the folded light
        curve is circular in phase, the sum includes the wraparound term
        between the last and first phase-sorted points.

        Expected behavior: minimised when the light curve is folded at
        (or near) its true period, since consecutive phase-sorted points
        then tend to have similar magnitudes; larger for a poor/aperiodic
        fold, where consecutive phase-sorted points are effectively
        uncorrelated in magnitude.
        """
        diffs = np.diff(self.lc.mag_phased, append=self.lc.mag_phased[0])
        return np.sum(diffs**2) / np.sum((self.lc.mag_phased - self.lc.mean)**2)

    @folded_property
    def string_length(self):
        """
        String-length statistic (Dworetsky 1983, MNRAS, 203, 917) of the
        folded light-curve: the total path length of the line connecting
        phase-sorted (phase, magnitude) points, including the circular
        wraparound term back to the first point.

        Expected behavior: same sense as `lafler_kinman` - minimised for
        a good fold at (or near) the true period, larger for a poor fold.
        """
        d_phase = np.diff(self.lc.phase, append=self.lc.phase[0] + 1.0)
        d_mag = np.diff(self.lc.mag_phased, append=self.lc.mag_phased[0])
        return np.sum(np.sqrt(d_mag**2 + d_phase**2))


    def _list_properties(self):
        """
        This tests if the LightCurve is Folded and return 
        the list of properties accordingly.
        """
        props = []
        is_folded = isinstance(self.lc, FoldedLightCurve)
        for name, value in inspect.getmembers(self.__class__,
                                              lambda o: isinstance(o, property)):
            # Skip folded-only properties if lc is not folded
            if isinstance(value, _tagged_property) and value.folded_only and not is_folded:
                continue
            props.append(name)
        return props
        
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
    """
    Cody+2014 M-index: asymmetry of the light curve between its extreme
    (top/bottom `percentile`) epochs and its median, normalised by the
    standard deviation. Set `is_flux=True` when `mag` is actually in flux
    units, to flip the sign convention so that M>0 still means
    dimming-dominated variability. See `VariabilityIndex.asymmetry_index`
    for the public entry point.

    Reference: Cody et al. (2014), AJ, 147, 82.
    """
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
        return (1. - 2*int(self.is_flux))*(np.mean(self.parent.lc.mag[self.get_percentile_mask]) - self.parent.lc.median)/self.parent.std

class PeriodicityIndex:
    """
    Cody+2014 Q-index: ratio of the residual (waveform-subtracted) variance
    to the raw phase-folded variance, both corrected for the mean
    photometric noise. See `VariabilityIndex.periodicity_index` for the
    public entry point.

    Reference: Cody et al. (2014), AJ, 147, 82.
    """
    def __init__(self, parent):
        self.parent = parent

    @property
    def value(self):
        return (np.std(self.parent.lc.residual, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)\
            /(np.std(self.parent.lc.mag_phased, ddof=1)**2 - np.mean(self.parent.lc.err_phased)**2)


def cody14_q_m_classifier(M_index, Q_index):
    """
    Simple classifier based on M and Q indices
    based on Cody et al. 2014 (https://ui.adsabs.harvard.edu/#abs/2014AJ....147...82C)
    """
    if M_index is None or Q_index is None:
        return None
    elif (M_index > 0.25) and (0 <= Q_index < 0.11):
        return 'EB'
    elif (M_index <= 0.25) and (0 <= Q_index < 0.11):
        return 'P'
    elif (abs(M_index) <= 0.25) and (0.11 <= Q_index <= 0.61):
        return 'QPS'
    elif (M_index > 0.25) and (0.11 <= Q_index <= 0.61):
        return 'QPD'
    elif (M_index > 0.25) and (0.61 < Q_index <= 1):
        return 'APD'
    elif (M_index < -0.25) and (Q_index > 0.11):
        return 'B'
    elif (abs(M_index) <= 0.25) and (0.61 < Q_index <= 1):
        return 'S'
    else:
        return 'Unclassified'


def gaia_AG_proxy(phot_g_mean_flux, phot_g_mean_flux_error, phot_g_n_obs):
    """
    Following Mowlavi et al. 2021 (https://ui.adsabs.harvard.edu/#abs/2021A%26A...648A..44M)
    this function calculates a proxy for the variability of Gaia sources using the uncertainty of
    the Gaia G-band fluxes.
    This is provided in equation (2) of the paper, and is given by:
    
    AG = sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux

    For constant stars this is approximately the standard deviation of
    G light curves due to noise and uncalibrated systematic effects.

    TODO: this has not yet been double-checked against Gaia's own
    implementation/values.

    Args:
        phot_g_mean_flux (float): Gaia `phot_g_mean_flux`.
        phot_g_mean_flux_error (float): Gaia `phot_g_mean_flux_error`.
        phot_g_n_obs (int): Gaia `phot_g_n_obs`.
    """
    return np.sqrt(phot_g_n_obs)*phot_g_mean_flux_error/phot_g_mean_flux