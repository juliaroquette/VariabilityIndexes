"""
Module for computing variability timescales and temporal/frequency-domain
features, for use as ML features alongside the magnitude-domain indexes in
variability/indexes.py.

@juliaroquette:

__Last Modified__: 04 September 2026
- Added temporal/frequency-domain feature extraction directly to TimeScale:
  * top-k periodogram peaks (period, power, FAP) with alias/lobe-aware
    peak-finding, plus periodogram-shape summaries (spectral entropy,
    power concentration).
  * Cody+14 Q-index evaluated at each of the top candidate periods, to
    disambiguate true periodicity from aliases (a real period folds
    cleanly; an alias usually doesn't).
  * an ACF-derived timescale (1/e crossing), reusing the existing
    StructureFunction machinery (SF and ACF are related via
    SF(tau) ~= 2*sigma^2*(1-ACF(tau)) for a stationary process). This
    path does not require iminuit.

previous update: 28 July 2025
- Removed old StructureFunction class
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import warnings
from variability.lightcurve import LightCurve
from variability.structure_function import StructureFunction


class TimeScale:
    """
    Derives characteristic variability timescales and temporal/frequency-domain
    features for a LightCurve.

    Two timescale estimators are available:
    - 'LSP': period from a Generalized Lomb-Scargle periodogram, for
      (quasi-)periodic light curves.
    - 'SF': timescale from fitting the structure function turnover,
      for aperiodic light curves.

    With method='auto' (default), LSP is tried first; if the highest
    peak's False Alarm Probability is not below `fap_prob`, the
    aperiodic (SF) timescale is attempted instead.

    Whenever the LSP periodogram is computed (method='LSP' or 'auto'),
    additional temporal features are derived by default:
    - top `n_peaks` periodogram peaks (period, power, FAP)
    - periodogram-shape summaries (spectral entropy, power concentration)
    - Cody+14 Q-index folded at each of the top `q_top_k` peak periods
    These can be disabled/tuned via the `n_peaks`, `min_freq_sep` and
    `q_top_k` keyword arguments.

    Independently of `method`, an ACF-derived timescale is computed by
    default (disable with `compute_acf=False`).
    """
    def __init__(self, lc, **kwargs):
        if not isinstance(lc, LightCurve):
            raise TypeError("lc must be an instance of LightCurve")
        self.lc = lc
        # initialize timescale attributes
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
        # cached full periodogram (populated by get_LSP_period)
        self.periodogram_freq = None
        self.periodogram_power = None
        self.periodogram_fap = None
        # cached StructureFunction instance, reused by get_acf_timescale
        self._sf = None
        # periodogram peak / shape features
        self.peak_periods = None
        self.peak_powers = None
        self.peak_faps = None
        self.n_significant_peaks = np.nan
        self.spectral_entropy = np.nan
        self.power_concentration = np.nan
        # Q-index at candidate periods
        self.peak_Q = None
        self.best_Q_period = np.nan
        self.best_Q = np.nan
        # ACF-derived timescale
        self.ACF_lags = None
        self.ACF_values = None
        self.ACF_ts = np.nan

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
        self._fap_prob = fap_prob
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

            # periodogram peak / shape / Q features, always available once
            # the LSP periodogram has been computed, regardless of whether
            # it ended up being the chosen timescale method.
            n_peaks = kwargs.get('n_peaks', 10)
            if n_peaks > 0:
                try:
                    self.get_periodogram_peaks(n_peaks=n_peaks,
                                                min_freq_sep=kwargs.get('min_freq_sep'))
                    self.get_periodogram_shape()
                    q_top_k = kwargs.get('q_top_k', 3)
                    if q_top_k > 0:
                        self.get_peak_Q_periodicity(top_k=min(q_top_k, n_peaks))
                except Exception as e:
                    warnings.warn(f"Periodogram peak/shape/Q features failed: {e}")

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

        # ACF-derived timescale: independent of which method was used above
        if kwargs.get('compute_acf', True):
            try:
                self.get_acf_timescale(bin_params=kwargs.get('sf_bin_params'))
            except Exception as e:
                warnings.warn(f"ACF timescale failed: {e}")

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

        As a side effect, this always caches the full periodogram grid
        (frequency, power and per-frequency FAP) on `self.periodogram_freq`,
        `self.periodogram_power` and `self.periodogram_fap`, so that peak
        extraction later doesn't require recomputing the Lomb-Scargle
        periodogram.

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
        # cache the full periodogram grid for later peak/shape extraction
        self.periodogram_freq = frequency
        self.periodogram_power = power
        self.periodogram_fap = self.FAP_probs

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

    def get_periodogram_peaks(self, n_peaks=10, min_freq_sep=None, fap_prob=None):
        """
        Extracts the top `n_peaks` distinct peaks from the cached LSP
        periodogram (see `get_LSP_period`), enforcing a minimum frequency
        separation between peaks so that a single spectral lobe doesn't
        fill the whole top-N list.

        Populates `self.peak_periods`, `self.peak_powers`, `self.peak_faps`
        (fixed-length arrays of size `n_peaks`, NaN-padded if fewer than
        `n_peaks` distinct peaks are found) and `self.n_significant_peaks`
        (count of extracted peaks with FAP below `fap_prob`).

        Args:
            n_peaks (int): number of peaks to keep.
            min_freq_sep (float, optional): minimum frequency separation
                between kept peaks. Defaults to one native frequency
                resolution unit, 1/lc.time_span.
            fap_prob (float, optional): FAP threshold used to count
                `n_significant_peaks`. Defaults to the `fap_prob` used to
                select the timescale method (`self._fap_prob`).
        """
        if self.periodogram_freq is None:
            raise RuntimeError(
                "Periodogram not available; run get_LSP_period "
                "(or instantiate TimeScale with method='LSP' or 'auto') first."
            )
        freq = self.periodogram_freq
        power = self.periodogram_power
        fap = self.periodogram_fap
        if fap_prob is None:
            fap_prob = self._fap_prob

        if min_freq_sep is None:
            min_freq_sep = 1.0 / self.lc.time_span
        freq_step = freq[1] - freq[0]
        distance = max(1, int(round(min_freq_sep / freq_step)))

        peak_idx, _ = find_peaks(power, distance=distance)
        if len(peak_idx) == 0:
            # fall back to the single global maximum
            peak_idx = np.array([np.argmax(power)])

        # order candidate peaks by decreasing power, keep the top n_peaks
        order = peak_idx[np.argsort(power[peak_idx])[::-1]]
        top_idx = order[:n_peaks]

        periods = 1.0 / freq[top_idx]
        powers = power[top_idx]
        faps = fap[top_idx]

        pad = n_peaks - len(top_idx)
        if pad > 0:
            periods = np.concatenate([periods, np.full(pad, np.nan)])
            powers = np.concatenate([powers, np.full(pad, np.nan)])
            faps = np.concatenate([faps, np.full(pad, np.nan)])

        self.peak_periods = periods
        self.peak_powers = powers
        self.peak_faps = faps
        self.n_significant_peaks = int(np.sum(faps < fap_prob))

    def get_periodogram_shape(self):
        """
        Derives global shape summaries of the cached LSP periodogram:
        - spectral_entropy: normalized Shannon entropy of the periodogram
          power distribution (0 = all power in a single frequency bin,
          1 = power spread uniformly across the whole grid).
        - power_concentration: fraction of total periodogram power
          contained in the single highest peak.
        """
        if self.periodogram_power is None:
            raise RuntimeError(
                "Periodogram not available; run get_LSP_period "
                "(or instantiate TimeScale with method='LSP' or 'auto') first."
            )
        power = self.periodogram_power
        total = np.sum(power)
        p = power[power > 0] / total
        self.spectral_entropy = -np.sum(p * np.log(p)) / np.log(len(power))
        self.power_concentration = np.max(power) / total

    def get_peak_Q_periodicity(self, top_k=3, min_epochs=5, waveform_params=None):
        """
        Evaluates the Cody+14 Q-index (periodicity index) by phase-folding
        the light curve at each of the top `top_k` candidate periods found
        by `get_periodogram_peaks`. A real periodic signal folds cleanly
        (Q close to 0); an alias/sampling artifact usually does not, so
        this is a much stronger period-validity check than periodogram
        power alone.

        Populates `self.peak_Q` (array of length `top_k`, NaN where a
        candidate period could not be evaluated) and `self.best_Q_period`/
        `self.best_Q`, the candidate with the lowest Q among those tried.

        Each candidate period is folded/evaluated independently: a failure
        for one candidate (e.g. a period too close to the sampling cadence)
        does not prevent the others from being evaluated.
        """
        # local imports to avoid a module-level circular import
        # (variability.indexes imports variability.lightcurve, not
        # variability.timescales, so this is safe at call time)
        from variability.lightcurve import FoldedLightCurve
        from variability.indexes import VariabilityIndex

        if self.peak_periods is None:
            self.get_periodogram_peaks(n_peaks=top_k)

        periods = self.peak_periods[:top_k]
        Qs = np.full(len(periods), np.nan)
        for i, period in enumerate(periods):
            if not np.isfinite(period):
                continue
            try:
                folded = FoldedLightCurve(lc=self.lc, timescale=float(period),
                                           waveform_params=waveform_params or {})
                Qs[i] = VariabilityIndex(folded, min_epochs=min_epochs).periodicity_index
            except Exception as e:
                warnings.warn(f"Q-index failed for candidate period {period}: {e}")

        self.peak_Q = Qs
        finite = np.isfinite(Qs)
        if finite.any():
            best_i = np.nanargmin(Qs)
            self.best_Q_period = periods[best_i]
            self.best_Q = Qs[best_i]
        else:
            self.best_Q_period = np.nan
            self.best_Q = np.nan

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
        # cache the binned SF so get_acf_timescale can reuse it, even if
        # the fit below fails
        self._sf = sf
        # fit the structure function
        sf.fit_sf(**fit_kwargs)
        (ts, ts_err), C0, C1, cost_min = sf.get_timescale()
        return ts, ts_err, C0, C1, cost_min

    def get_acf_timescale(self, bin_params=None):
        """
        Derives an autocorrelation-function (ACF) timescale by reusing the
        binned structure function: for a (wide-sense) stationary process,
        SF(tau) ~= 2*sigma^2*(1 - ACF(tau)), so ACF(tau) = 1 - SF(tau)/(2*sigma^2).

        `self.ACF_ts` is the lag at which this derived ACF first drops
        below 1/e, found by linear interpolation between the bracketing
        SF bins. `self.ACF_lags`/`self.ACF_values` hold the full derived
        ACF curve for diagnostics.

        Unlike `get_structure_function_timescale`, this does not require
        fitting (no iminuit dependency) - only `StructureFunction.get_sf`
        and `.bin_sf`.

        Args:
            bin_params (dict, optional): overrides for StructureFunction.bin_sf
                keyword arguments. If None and a StructureFunction has
                already been binned (e.g. by get_structure_function_timescale),
                that cached result is reused instead of rebinning.
        """
        if self._sf is not None and bin_params is None:
            sf = self._sf
        else:
            bin_kwargs = dict(sf_err=None,
                               log=True,
                               hybrid=False,
                               bin_min_size=5,
                               max_bin_exp_factor=3.0,
                               step_size=0.2,
                               resolution=0.02)
            if bin_params:
                bin_kwargs.update(bin_params)
            sf = StructureFunction(lc=self.lc)
            sf.get_sf()
            sf.bin_sf(**bin_kwargs)
            self._sf = sf

        var = np.var(self.lc.mag, ddof=1)
        acf = 1.0 - sf.sf_binned / (2.0 * var)
        lags = sf.time_bins
        order = np.argsort(lags)
        lags, acf = lags[order], acf[order]
        self.ACF_lags = lags
        self.ACF_values = acf

        target = 1.0 / np.e
        below = np.where(acf < target)[0]
        if len(below) == 0:
            self.ACF_ts = np.nan
            return
        j = below[0]
        if j == 0:
            # already below 1/e at the shortest lag available
            self.ACF_ts = lags[0]
        else:
            x0, x1 = lags[j - 1], lags[j]
            y0, y1 = acf[j - 1], acf[j]
            self.ACF_ts = x0 + (target - y0) * (x1 - x0) / (y1 - y0)

    def __repr__(self):
        return (f"<TimeScale(ts={self.ts}, method={self.method}, "
                f"LSP_ts={self.LSP_ts}, SF_ts={self.SF_ts}, "
                f"ACF_ts={self.ACF_ts})>")


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
