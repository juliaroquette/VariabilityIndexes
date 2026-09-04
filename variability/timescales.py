"""
Module for computing variability timescales.

@juliaroquette:

__Last Modified__: 04 September 2026
- Merged TimeScale_refactored into TimeScale: a single class now derives
  a timescale for any LightCurve using either the GLS periodogram (LSP)
  or the structure function (SF), or both ('auto').
- Wired up the structure-function branch (previously a stub).
- Dropped unused imports (find_peaks, minimize, pdist, Minuit, matplotlib).

previous update: 28 July 2025
- Removed old StructureFunction class
"""

import numpy as np
from astropy.timeseries import LombScargle
import warnings
from variability.lightcurve import LightCurve
from variability.structure_function import StructureFunction


class TimeScale:
    """
    Derives a characteristic variability timescale for a LightCurve.

    Two estimators are available:
    - 'LSP': period from a Generalized Lomb-Scargle periodogram, for
      (quasi-)periodic light curves.
    - 'SF': timescale from fitting the structure function turnover,
      for aperiodic light curves.

    With method='auto' (default), LSP is tried first; if the highest
    peak's False Alarm Probability is not below `fap_prob`, the
    aperiodic (SF) timescale is attempted instead.
    """
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        # initialize attributes
        self.ts = np.nan
        self.ts_err = np.nan
        self.method = None
        self.fap = np.nan
        self.power = np.nan
        self.LSP_ts = np.nan
        self.C0 = np.nan
        self.C1 = np.nan
        self.SF_ts = np.nan
        self.SF_ts_err = np.nan
        self.SF_cost_min = np.nan
        # deal with which timescale method to use
        # if method is SF or LSP, it will only return the timescale in the given method. Otherwise, it will attempt to get a timescale using LSP first, and then SF if no periodic timescale was found.
        if 'method' not in kwargs:
            warnings.warn("No 'method' specified. Using default 'auto' method.", UserWarning)
            kwargs['method'] = 'auto'
        elif kwargs['method'] not in ['LSP', 'SF', 'auto']:
            raise ValueError("Method must be 'LSP', 'SF' or 'auto'")
        method = kwargs['method']
        #
        fap_prob = kwargs.get('fap_prob', 0.01)
        definition = kwargs.get('definition', 'auto')
        osf, min_freq, max_freq = pre_defined_parameters(self.lc.time, definition=definition)
        # if methods set to lomb scargle or to auto, first try to get
        # timescales using the Lomb-Scargle periodogram
        if method in ['LSP', 'auto']:
            best_freq, best_power,\
            FAP_highest_peak = self.get_LSP_period(fmin=min_freq,
                                                   fmax=max_freq,
                                                   osf=osf,
                                                   periodogram=kwargs.get('periodogram', False),
                                                   definition=definition)
            self.LSP_ts = 1 / best_freq
            self.fap = FAP_highest_peak
            self.power = best_power

            if (FAP_highest_peak < fap_prob) or (method == "LSP"):
                self.ts = 1.*self.LSP_ts
                self.method = 'LSP'
        # if method is set to SF, or if no timescale was obtained from LS
        # then proceed to get timescale from SF
        if method == 'SF' or (method == 'auto' and (self.method is None)):
            try:
                ts, ts_err, C0, C1, cost_min = self.get_structure_function_timescale(
                    bin_params=kwargs.get('sf_bin_params'),
                    fit_params=kwargs.get('sf_fit_params'),
                )
                self.SF_ts = ts
                self.SF_ts_err = ts_err
                self.C0 = C0
                self.C1 = C1
                self.SF_cost_min = cost_min
                if ts is not None:
                    self.ts = 1.*ts
                    self.ts_err = ts_err
                    self.method = 'SF'
            except Exception as e:
                warnings.warn(f"Structure Function failed: {e}")

    def get_LSP_period(self,
                          fmin,
                          fmax,
                          osf,
                          periodogram,
                          definition='auto'):
        """
        Simple Period estimation code using Lomb-Scargle.
        This adopts an heuristic approach for the frequency grid where,
        given the max/min values in the frequency, the grid is oversampled
        by a default value of 5.


        Args:
            astropy.timeseries.LombScargle arguments:
            osf (int, optional): samples_per_peak Defaults to 5.
            fmin (int, optional): minimum frequency for the periodogram
            fmax (int, optional): maximum frequency for the periodogram
                NOTE: default min and max value consider:
                        - fmax is set by a 0.5 days period, which is about
                        the break-up speed for very young stars.
                        - fmin is arbitrary set to 250 days.
            periodogram (bool, optional): if True, returns the periodogram,
                                          otherwise returns the period.

        Returns:
        if periodogram is True:
            frequency float: frequency of the highest peak
            power float: power of the highest peak
            FAP_level float: False alarm probability level for 1%, 10% and 40%
        else:
            frequency of the highest peak: float
            power of the highest peak: float
            FAP_highest_peak: 0-1. float: False Alarm Probability for the highest peak
        """
        # define the base for the Lomb-Scargle
        ls = LombScargle(self.lc.time, self.lc.mag, self.lc.err)
        if definition == 'Chloe':
            frequency = np.arange(fmin, fmax, step=0.0002)
            power = ls.power(frequency, method='slow')
        elif definition == 'Gaia':
            STEP =  1. / 2000. / 5.
            frequency = np.arange(fmin, fmax, step=STEP)
            power = ls.power(frequency, method='slow')
        else:
            frequency, power = ls.autopower(samples_per_peak=osf,
                                            minimum_frequency=fmin,
                                            maximum_frequency=fmax,
                                            method='slow')
            # note here that method="slow" gives the astropy equivalente of GLS

        self.FAP_probs = ls.false_alarm_probability(power,method='baluev',
                                            minimum_frequency=fmin,
                                            maximum_frequency=fmax,
                                            samples_per_peak=osf)

        if bool(periodogram):
            return frequency, power, self.FAP_probs
        else:
            freq_highest_peak = frequency[np.argmax(power)]
            power_highest_peak = power.max()
            FAP_highest_peak = ls.false_alarm_probability(power_highest_peak,method='baluev',
                                         minimum_frequency=fmin,
                                         maximum_frequency=fmax,
                                         samples_per_peak=osf)
            return freq_highest_peak, power_highest_peak, FAP_highest_peak

    def get_structure_function_timescale(self, bin_params=None, fit_params=None):
        """
        Uses the StructureFunction class to derive an aperiodic timescale
        from the turnover of the light curve's structure function.

        Args:
            bin_params (dict, optional): overrides for StructureFunction.bin_sf
                keyword arguments (sf_err, log, hybrid, bin_min_size,
                max_bin_exp_factor, step_size, resolution).
            fit_params (dict, optional): overrides for StructureFunction.fit_sf
                keyword arguments (yerr, last_params, limits, log,
                cost_flavour, reduced_chi2, input_cost).

        Returns:
            ts (float or None): fitted turnover timescale, t0.
            ts_err (float or None): 1-sigma uncertainty on t0.
            C0 (float or None): fitted noise-floor parameter.
            C1 (float or None): fitted amplitude parameter.
            cost_min (float or None): minimized cost-function value.
        """
        bin_kwargs = dict(sf_err=None,
                           log=True,
                           hybrid=False,
                           bin_min_size=5,
                           max_bin_exp_factor=3.0,
                           step_size=0.2,
                           resolution=0.02)
        if bin_params:
            bin_kwargs.update(bin_params)
        fit_kwargs = dict(yerr=None,
                           last_params=[1, 0.01, 0.1],
                           limits=[(0.07, 1800), (1e-6, 5), (1e-5, 100)],
                           log=True,
                           cost_flavour='L2_error',
                           reduced_chi2=True,
                           input_cost=None)
        if fit_params:
            fit_kwargs.update(fit_params)

        sf = StructureFunction(lc=self.lc)
        # estimate SF values
        sf.get_sf()
        # bin the structure function
        sf.bin_sf(**bin_kwargs)
        # fit the structure function
        sf.fit_sf(**fit_kwargs)
        (ts, ts_err), C0, C1, cost_min = sf.get_timescale()
        return ts, ts_err, C0, C1, cost_min

    def __repr__(self):
        return (f"<TimeScale(ts={self.ts}, method={self.method}, "
                f"LSP_ts={self.LSP_ts}, SF_ts={self.SF_ts})>")


def pre_defined_parameters(time, definition='Gaia'):
    """
    Returns pre-defined parameters for the TimeScale class.

    What is defined:
    - ofs (samples per peak:)
        This factor is called samples_per_peak in Astropy's LombScargle
        and it defaults to 5.
        Once fmin and fmax are defined, this is used to set the step in frequency for the .autopower method.
        In Gaia, this is called `stepFrequencyScaleFactor` and had
        recommended values of 5 in DR3 and 10 in DR4.
    - min_freq (minimum frequency):
        This is the minimum frequency for the periodogram.
    - max_freq (maximum frequency):
        This is the maximum frequency for the periodogram.
    """
    if definition == 'Gaia':
        osf = 5
        min_freq = 0.001
        max_freq = 2.5
    elif definition == 'Chloe':
        # Chloe defined that dynamically where she actually fiexes the df
        osf = 5
        min_freq = 1./ ((max(time) - min(time)))
        max_freq = 1 / (np.median(np.diff(time)) * 2)
    elif definition == 'auto':
        osf = 5
        # Guarantees at least one full period cycle is covered
        min_freq = 1 / ((max(time) - min(time))/2)
        max_freq = 1./ 0.5 / (np.median(np.diff(time)))
    else:
        raise ValueError("Definition must be 'Gaia', 'auto', or 'Chloe'")
    return osf, min_freq, max_freq
