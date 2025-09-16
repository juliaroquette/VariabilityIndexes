# A Python package for deriving variability features from any kind of light-curves

**@juliaroquette** Package under development for deriving Q&M indexes (and a few other variability indexes) for any time type of light-curves. 


**Last Update**: 29th July 2025

In this current version, one can import and use it by doing:

```python
import sys
sys.path.append('PAT/TO/THE/PACKAGE/LOCATION')  
```

TO DO:
<details>
- :white_large_square: Proper packaging
- :white_large_square: Proper installation tutorial
</details>

# `lightcurve` module:

Provides tools for loading light-curves as objects. Three distinct classes related to light-curves are currently included: 
- `LightCurve`s are the simplest light-curves
- `FoldedLightCurve`s are phase-folded light-curves with a known timescale
- `SyntheticLightCurve` (under construction) provide a suite of models of light-curves for different variability modes and survey fingerprints. 

Throughout this documentation, we approach an observed light-curve as:

$$x_i = x(t_i) + a(t_i) + \sigma_{i},$$

where $x_i$ is the $i$-th observation at the time $t_i$, $\{x(t_i)\}$ are snapshots of a time-dependent signal at the time $t_i$ (or the primary signal in the light-curve),  $\{a(t_i)\}$ is any secondary signal,  $\sigma_{i}$ is the photometric uncertainty for the $i$-th observation.

For example, let's consider the following light-curve:

````python
import numpy as np
N = 100 
time = np.linspace(0, 80, N)
err =  0.01 * np.random.random_sample(N)
period = 10. 
amplitude = 0.5
noise =  np.random.normal(scale=0.05, size=N)
mag_sin = 0.5 * amplitude * np.sin(2 * np.pi * time / period) + noise
````


## `LightCurve` class

To instantiate a `LightCurve` object:

```python
  from variability.lightcurve import LightCurve
  lc = LightCurve(time, mag, err, mask=None, is_flux=False)
```

Where the attributes `time`, `mag`, and `err` are numpy-arrays with the same length providing the observational time, magnitudes and magnitude uncertainties respectively. Optionally a `mask` boolean array can be passed to filter out missing data or spurious observations. The `is_flux` attribute informs if the light-curve is in terms of magnitudes or fluxes (this is important when calculating M-indexes.)

`LightCurve` objects have a series of properties:
- `N` : Number of datapoints in the light curve.
- `std`: Standard deviation of the magnitudes [(uses bias corrected `numpy.std`)](https://numpy.org/doc/stable/reference/generated/numpy.std.html).
$$
\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}
$$

- `mean` : simple average of the magnitudes.

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

- `mean_err`:  average uncertainty in magnitudes (or typical uncertainty)

$$\sigma_{phot} = \frac{1}{N} \sum_{i=1}^{N} \sigma_i$$


- `weighted_average`: Weighted average of the magnitude values.

$$\bar{x}_w = \frac{\sum_{i=1}^{N} w_i x_i}{\sum_{i=1}^{N} w_i}$$

where $w_i=\frac{1}{\sigma_i^2}$. Uses [`numpy.average`](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average).

- `median`: Median of magnitudes.

$$\text{Median} = \begin{cases}
x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\
\frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2} & \text{if } n \text{ is even}
\end{cases}$$

Uses [`np.median`](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html#numpy.nanmedian)
- `min`: Minimum value of the magnitudes.
- `max`: Maximum value of the magnitudes.
- `ptp`: Peak-to-peak amplitude of the magnitude values. Defined as the difference between the median values for the datapoints in the 5% outermost tails of the distribution.
- `time_max`: Maximum value of the observation times ($t_{max}$).
- `time_min`: Minimum value of the observation times ($t_{min}$).
- `time_span`: Total time-span of the light curve
 $$t_{max}-t_{min}$$ 
- `range`: another flavor of ptp amplitude bin in terms of maximum/minimum values of magnitude: $$x_{max}-x_{min}$$ 
- `SNR` signal-to-noise ratio (standard deviation of the data divided by average uncertainty)

$$\text{SNR}=\frac{\sigma}{\sigma_{phot}}$$



## `FoldedLightCurve` class



```python
  from variability.lightcurve import  FoldedLightCurve
  lc_f = FoldedLightCurve(time=time, mag=mag, err=err, timescale=period)
```

where `timescale` is a timescale to be used for phase-folding the light-curve (for example, the variability period). 

Alternatively:

```python
  lc_f = FoldedLightCurve(lc=lc, timescale=timescale)
```

Additionally to the attributes inherited from the `LightCurve`object, a `FoldedLightCurve`light curve has the following additional attributes:

- `timescale`: The timescale used for folding the light curve. Can be a variability period or any characteristic timescale inferred for the light-curve (This can be inferred using the `timescale` module)
- `reference_time`: Reference time for phase-folding the light-curve. It is set to 0 as default.
- `phase`: The phase values of the folded light curve (between 0 and 1).
- `phase_number`: phase number (integer part of the phase calculation)
- `mag_pahsed`: The magnitude values of the folded light curve, sorted based on phase.
- `err_pahsed`: The error values of the folded light curve, sorted based on phase.
- `waveform`: estimated waveform for phase-folded light-curve smoothed using uneven_savgol (default) unless other method is specified. 
- `residual`: residual phase-folded curve (`mag_phased - waveform`)

All returned values are sorted as a function of phase value. 


## `SyntheticLightCurve`
<details>
**@juliaroquette** Still under implementation, will allow to generate synthetic light-curves for given observational windows. 

```python 
from variability.lightcurve import FoldedLightCurve
```


## TO DO list

:white_large_square: **@juliaroquette** It may be worth it consider the possibility of merging `LightCurve` and `FoldedLightCurve` into a single class. <- Consider that after the `timescale.py` package has been implemented. 

:white_large_square: read observational windows from file

:white_large_square: Implement a list of observational windows
  - :white_large_square: TESS
  - :white_large_square: Rubin
  - :heavy_check_mark: ZTF
  - :heavy_check_mark: ASAS-SN (g and V)
  - :heavy_check_mark: Gaia-DR3
  - :heavy_check_mark: Gaia-DR4
  - :heavy_check_mark: Gaia-DR5
  - :heavy_check_mark: AllWISE
  - :white_large_square: input

:white_large_square: Include waveforms 
  - :heavy_check_mark: PS
  - :white_large_square: QPS
  - :white_large_square: EB
  - :white_large_square: AATAU
  - :white_large_square: QPD
  - :white_large_square: QPB
  - :white_large_square: B
  - :white_large_square: MP
  - :white_large_square: LP

:white_large_square: Function that generates a waveform for a refined timestep

  - :white_large_square: Function that convolves the waveform to an observational window
</details>


# Variability Indexes

Include a suite of widely used variability indexes. 

## `indexes`



To instantiate a `VariabilityIndex` object:

```python
from variability.indexes import VariabilityIndex

var = VariabilityIndex(lc_p, timescale=period)
```

you are expected to pass in a `LightCurve` object, or a `FoldedLightCurve` object. However,  **note that some variability indexes, like the Q-index itself, require either a `timescale` argument or a `FoldedLightCurve` instance (which already have an instance `timescale`).



### 'Usual' Variability indexes:

#### Shapriro-Wilk test

$$W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

Where $W$ is the Shapiro-Wilk statistic, $n$ is the number of observations, $x_i$ are the individual values of the dataset, $\bar{x}$ are the mean (average) of the dataset, and 
 $x_{(i)}$ are the $i$-th order statistic in the sorted dataset. The coefficients $a_i$
  are pre-calculated constants based on the sample size and are used in the Shapiro-Wilk test.

#### median absolute deviation (MAD)
  $$\text{MAD} = \text{median} \left( \left| x_i - \text{median}(x) \right| \right)$$

#### $\chi^2$
$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

with $\chi^2$ as the chi-squared statistic, $O_i$ is the observed frequency for each category or bin, $ E_i$ is the expected frequency for each category or bin, and $k$ are the total number of categories or bins.

#### reduced-$\chi^2$

$\chi_\nu^2 = \frac{\chi^2}{\nu}$, where $\nu$ are the degrees of freedom

#### Inter-quantitle  range (IRQ)
$$\text{IQR} = Q_3 - Q_1$$
Where $Q_1$ and $Q_3$ are the first and third quartile

#### Robust-Median Statistics (RoMS)

$$\text{Robust-Median} = \text{median}(|x_i - \text{median}(x)|)
$$

#### normalisedExcessVariance


$$\sigma_{\text{NXS}}^2 = \frac{S^2 - \langle \epsilon^2 \rangle}{\langle x \rangle^2}$$

#### Lag1AutoCorr ($l_1$)

####  andersonDarling
  $$A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} \left[ \frac{2i - 1}{n} \cdot \ln\left( F(X_{(i)}) \right) + \left( 1 - \frac{2i - 1}{n} \right) \cdot \ln\left( 1 - F(X_{(n-i+1)}) \right) \right]$$
Where $A^2$ is the Anderson-Darling statistics, $n$ is the number of observations, $X_{(i)}$ is the $i$-th order statistic in the sorted dataset and $F(X_{(i)})$  is the empirical distribution function at $X_{(i)}$.
#### skewness
$$\text{Skewness} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2\right)^{\frac{3}{2}}}$$
####  kurtosis
$$\text{Kurtosis} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{\left(\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2\right)^2}$$


### Normalized peak-to-peak variability

`norm_ptp`

Sokolovsky et al. (2017):
$$\nu = \frac{(m_i-\sigma_i)_\mathrm{max} - (m_i-\sigma_i)_\mathrm{min}}{(m_i+\sigma_i)_\mathrm{max} + (m_i+\sigma_i)_\mathrm{min}}$$

where $m_i$ is the magnitude measurement and $\sigma_i$ is the corresponding measurement error. 

### `VariabilityIndex.M_index`

$$M = \frac{<m_{10\%}>-m_{med}}{\sigma_m}$$

$<m_{10\%}>$ is all the data in the top and bottom decile of the light-curve. 
Not that there are conflicting definitions in the literature, where $\sigma_m$ is sometimes the overall rms of the light-curve and sometimes its standard-deviation! Here I am using the second one. 

### `VariabilityIndex.Q_index`

$$Q = \frac{\sigma_\mathrm{res}^2-\sigma_\mathrm{phot}^2}{\sigma^2_\mathrm{raw}-\sigma^2_\mathrm{phot}}$$ 

where:
- $\sigma_\mathrm{res}^2$ and $\sigma^2_\mathrm{raw}$ are the ~rms~ variance values of the raw light curve and the phase-subtracted light curve.
- $\sigma_\mathrm{raw}^2$ is the variance of the original light-curve
- $\sigma_\mathrm{phot}$ is the mean photometric error


1. Find a period (Lomb Scargle Periodogram for ex)
2. Fold the light curve to the period
3. Use mooving average to get the smoothed shape of the curve
4. subtract it from phased light curve 
5. estimnate $\sigma_\mathrm{res}$



## TO DO list

<details>
- :white_large_square: Fix Stetson-index implementation
- :white_large_square: Complete Variability-index documentation
  - :white_large_square: Complete description
  - :white_large_square: add examples
  - :white_large_square: add references
</details>

# Filtering

Include a suite of filters to aid pre-processing light-curves. Most importantly, the `WaveForm` class provides a waveform estimation given the assumption of a periodic timescale of variability. 

**@juliaroquette** mostly implemented, but still needs polishing and debugging. 


## `Filtering`

```python
from variability.filtering import Filtering
```
<details>

### `Filtering.filter`

### `Filtering.sigma_clip`

### `Filtering.Cody_long_trend`

### `Filtering.savgol_longtrend`

### `Filtering.uneven_savgol`

### `Filtering.smooth_per_timescale`

### `Filtering.rolling_average`

</details>

## `WaveForm`

The class `WaveForm` provide tools to compute a smoothed version of a given phase-folded light-curve (`FoldedLightCurve`) based on a choice of filtering algorithm. This smoothed version is called 'waveform`.  Additionally, a residual light-curve is estimated as the phase-folded light-curve subtracted by its waveform. 


```python
from variability.filtering import  WaveForm
```

In the context of the variability indexes implemented here, `WaveForm` was designed with phase-folded light curves in mind. There is thus an underlying assumption that `phase` (phases for magnitudes in a phase-folded light curve) and `mag_phased` (phase-folded magnitudes) are circular. This assumption is used to deal with edging issues in most waveform-estimators implemented here.

To instantiate a `WaveForm` object:

```python
wf = WaveForm(phase, mag_phased)
```
A waveform can be obtained as:

```python
waveform = wf.get_waveform(waveform_type='uneven_savgol',
                          waveform_params={'window': 21, 'polynom': 2})
```

Where `waveform_type` is the waveform estimator of choice as in the list below, and `waveform_params` is a dictionary with the appropriate parameters for the chosen waveform estimator.


<details>

### `WaveForm.circular_rolling_average_number`

Simple rolling average that uses a fixed number of points for its window. Window size is defined by the attribute `window_size`, which should be an odd integer number. As default `window_size=5` but this should be chosen based on the light-curves at hand.

*Note:* This only works well with regularly phase sampling.

### `WaveForm.circular_rolling_average_phase`

Rolling average adapted to uneven based on a phase window. Useful for very irregular sampling in phase space. Window size is defined by the attribute `wd_phase` and represents an interval in phase. Default is `wd_phase=0.1`, i.e., 10% of the phase is used for windowing. 


### `WaveForm.waveform_H22`

	Boxcar smoothing method as used for waveform estimation in [Hillenbrand et al. 2022](https://ui.adsabs.harvard.edu/abs/2022AJ....163..263H/abstract). Borrowed on [the code used by Hillenbrand et al.](https://github.com/HarritonResearchLab/NAPYSOs) for waveform estimation in ZTF light curves. Uses [`astropy.convolution.Box1DKernel`](https://docs.astropy.org/en/stable/api/astropy.convolution.Box1DKernel.html). The only attribute here is `kernel=4` which is the same as using 1/4 of the number of points as kernel. 

### `WaveForm.waveform_Cody`

This uses Scipy's  median filter,
[`scipy.ndimage.median_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html), as used by [Cody et al.](https://ui.adsabs.harvard.edu/abs/2018AJ....156...71C/abstract). The attribute `n_point=50` is passed to `scipy.ndimage.median_filter` as the `size` attribute, which is just the window size for filtering. This is expected to work for evenly spaced data. 

### `WaveForm.savgol`

[Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) for uniformly evenly spaced curves. The Sav-Gol filter smooth the data by fitting polynomials of a given degree to adjacent point within a given windew. Here this simply employs [`scipy.signal.savgol_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) with all default parameters except for `window` (default 10) and `polynom` (=3) which . 

### `WaveForm.uneven_savgol`

Adaptation of the [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) for uneven data. This was based on an [implementation discussed at StackExchange](https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data). Similarly to the standard Savitzky-Golay filter, here the attributes `window` (default 10) and `polynom` (=3) define the window size and polynomial degree for filtering.


</details>


Once a waveform has been derived, a residual light curve can be estimated as:

```python
residuals = wf.residual_magnitude(waveform)
```
# Variability timescale 


## `timescale` module:

```python
from variability.timescale import  TimeScale
```

The `TimeScale` class allows to quick estimation of variability light curves for either a trio of (`time, mag, err`) or an object `LightCurve`. There are two types of timescale estimator currently implemented:

- Periodic Timescale:
- Aperiodic timescale: 

## Structure functions

The **Structure Functions (SF)** are used to examine the timescales of variability in a light curve. It quantifies how the magnitude differences between observations evolve as a function of time lag.

#### **Definition**  

There are a few different definitions of SF out there, but here we work with definition as in [Hughes92](https://ui.adsabs.harvard.edu/abs/1992ApJ...396..469H) where:  

$$SF(\tau) = \frac{1}{N} \sum_{i,j} (m_i - m_j)^2, \,\, i>j$$

where $N$ is the number of pairs separated by a time lag $\tau$, and $m_i - m_j$ is the magnitude difference for each pair.


### **Key SF Properties**  

For the moment,  the SF analysis implemented is restricted to light curves with an underlying assumption of aperiodic variability (i.e., a Lomb-Scargle periodogram analysis is carried out in a previous step and no significant period was found).

Starting from this _aperiodic timescale_ hypothesis, we assume the structure function will have the form (`structure_function.model_function(x, t0, c0, c1)`):

$$y_{SF}(\tau) = C_0 + C_1(1 - \exp{(-\tau/t_0)}),$$


- At _small time lags_ ($\tau\rightarrow0$), the SF is dominated by measurement noise, which in this model is mapped to $C_0$
- At _intermediate time lags_, the SF increases following a linear slope in log-scale
- At _large time lags_ ($\tau\rightarrow t_{max}$), the SF saturates to a plateau ($C_0+C_1$). 
- The transition between the SF increase and the plateau occurs at the turnover timescale, $t_0$, which here we interporet as the characteristic timescale in the light curve. 

**Warnings:**

- This model assumes aperiodic variability. Periodic variability will introduce significant substructure to the SF and is not well represented by this model (see, for example [Roelens17](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3230R) Figure 1.). 
- This model assumes there exist a dominant variability mode in the light-curve with a a timescale $t_{min}\leq t_s \leq t_{max}$. This model does not well represents multi-mode variability.

### SF sampling

Before fitting the SF model to the SF data. There may be a few complications behind sampling the data, which will depend of the specific light curve sampling. 


#### unbinned

The first step for sampling is to estimate the ${SF(\tau),\tau}$. This step is done by `structure_function.get_sf_time_lags(time, mag, err)`, and includes as a default the propagation of magnitude uncertainties into SF uncertainties.  

*Why not simply fit the SF model to unbinned data?* Depending on the duration and cadence of the light-curve, the SF has many have many more points at longer time-lags. Consequently, when fitting the the SF model directly to the unbinned data, $SF(\tau)$ for large $\tau$ will have much more weight in the fit.

#### Binned

The alternative to unbinned data is _to bin_ the data while averaging the SF values within bins. Binning will thus smooth regions of the SF where the density of points is too large compared to not so well populated $\tau$-ranges. In many applications "binning is sinning". But in this case sinning allows avoiding biasing the SF fits towards larger timescales. 

The function `adaptative_binning(time_lag, sf_values, sf_err=None)` includes different _flavours_ of binning. 

All _flavours_ of binning share an euristic approach for picking the number of bins to use based on a desired timescale resolution. This is determined by the `resolution` keyword. 
- In log scale (`log=True`) `resolution` ($\Delta_\tau$) should be in dex. The number of bins is then difined as:

$$N = 1 + \left\lfloor \frac{\log_{10}(\text{max}(\tau))-\log_{10}(\text{min}(\tau))}{\Delta_\tau} \right\rfloor$$

- in linear scale (`log=False`) `resolution` ($\Delta_\tau$) should be in the same units as $\tau$ (days)
 
$$N = 1 + \left\lfloor \frac{\text{max}(\tau)-\text{min}(\tau)}{\Delta_\tau} \right\rfloor$$


This calculation is done within the auxiliary function `bin_edges_and_centers(time_lag, resolution, log=True)`, which supports the binning processing by properly deriving the number of bins and making sure the edge values of $\tau$ are the center of the last bins. 

**what resolution to use**?

For example, relatively well sampled SF around timescale related to rotational modulation in young stars, we may want to have timebins 0.5 days appart at a $\tau=10$ d

$$\log_{10}{(10 + 0.5)} - \log_{10}{(10)} \approx 0.02\,\text{dex}$$

 which results on $N_{bins}=240$

##### regular bins

Simple regular bins will be problematic for most light curves because some SF bins will be empty, and others will not have enough data to average. Still for comparison and completeness, the `adaptative_binning` function will yield regular bins if `adaptative_binning(time_lag, sf_values, hybrid=False, bin_min_size=1, max_bin_exp_factor=1.0`)

#####  Adaptative Binning

Bins are adapted according to some criteria.

`structure_function.adaptative_binning(time_lag, sf_values, sf_err=None,
                       log=True,
                       hybrid=False,
                       bin_min_size=5, 
                       max_bin_exp_factor=3.0, 
                       step_size=0.2, 
                       resolution=0.02)`

###### Expanded bins 

Bins are expanded around its centre until either it is composed by minimum number of data-points (`bin_min_size`) or it is completely merged with its two imediate neighbours.
This is similar to the approach in [Roelens17](https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.3230R) where a tolerance factor $\epsilon_i$ introduced to ensure that each bin contains enough pairs for reliable calculation. 

The maximum expansion allowed is set by `max_bin_exp_factor` which should be a value between `1.0` and `3.0` where:

- `max_bin_exp_factor=1` allows for no expansion (bins are fixed), 
- `max_bin_exp_factor=3.0` means 3 bins are merged (the one in question and each of its imediate bins)
- `max_bin_exp_factor=2.0` means the bin is expanded until half of their neighour bins.

Additionally, `step_size` sets the size of the expansion step, where:
- `step_size=0.05` is the minimum value allowed and sets a 2.5% expansion to each direction at each iteration
- `step_size=1.0` is the maximum and it means the bin grows 50% to each direction. 

If `bin_min_size` is never reached, the bin is discarded. In this context, `pair_counts` return the number of sources per bin, while `irregular_bin=0.0` if this was a regular bin, `irregular_bin=-1.0` if using individual pairs and `>0.0` it reflects final width of the bin after expansion. 

For example: `adaptative_binning(time_lag, sf_values, sf_err=None, hybrid=False, bin_min_size=5, max_bin_exp_factor=3.0, step_size=0.2)` will expand bins until either the bin is merged with its immediate neighbours or at least `bin_min_size=5` are within the bin, while gradually expanding bins by 20% of their width. 

###### Hybrid mode

For bins where the number of sources is lower than `bin_min_size`, the data points in that bin are appended rather than their average. `adaptative_binning(time_lag, sf_values, sf_err=None, hybrid=True, bin_min_size=5)` will count indivual points in bins with less than ` bin_min_size=5` sources. (see note on treatment of uncertainty)


##### Log/linear scale:

Binning can be done in either log (`log=True`) or linear scale (`log=False`). However note that the linear case won't work well for SF where $\tau$ covers too many orders of timescale, such as the use of linear binning will squeeze all small $\tau$ into a few bins. 

##### treatment of uncertainty

- When `sf_err=sf_error` uncertainty is propagated (from the photometric uncertainties) from the SF into the bins as  $\Delta \text{SF}_{bin} = \frac{\sqrt{\sum \sigma_{SF}^2}}{N}$. However, this will often underestimate uncertainties and bias SF-fits.

- when `sf_err=None` uncertainties will be ignored and the bin uncertainty will be estimated from the spread in the bin as $\Delta \text{SF}_{bin} = \frac{\sigma_{SF_{bin}}}{\sqrt{N}}$.
  - When hybrid mode is also used (`sf_err=None, hybrid=True`), the median absolute deviation of the whole set of `sf_values` is used rather than the SF propagate uncertainties: $\Delta \text{SF}_{ind} = 1.4826\times\text{median}(|SF-\tilde{SF}|)$. 

### SF-Fitting 

The fitting function, $ y_{SF}(\tau)$, was defined above. For fitting it to the SF data, we are using the MINUIT optimisation tool from CERN. 

- [MINUIT reference manual](https://root.cern.ch/download/minuit.pdf)  
- [iminuit library](https://scikit-hep.org/iminuit/)

Minuit is just a minimisation function and will use whatever a function (`cost_function_for_minuit`) I pass `m=Minuit(cost_function_for_minuit)`. Here I simply adapted Chloé's `fit_with_minuit` to take the refactored `cost_function`. 


#### Cost-functions for minimization:

- $y(\tau)$ is the SF data, with $t_i$ the SF data-point at the i-th $\tau$.
- $\sigma_i$ is the uncertainty/error at the i-th SF data point.
- $\sigma_y$ is the standard deviation for the full SF dataset
- $m\rightarrow y_\text{SF}$ is the model assumed for the SF


##### L2 (simple residual)

This is a simple for minimizing residuals. Can work well enough for minimization, but it does not take noise into consideration and it is is not really a statistical likehood function. It will work well only as long as the errors are regular/well-behaved (gausian)

$$\chi^2_\mathrm{L2} = \sum_{i} \bigl(y_i - m_i\bigr)^2.$$ 

$$\chi^2_\mathrm{L2,log} = \sum_{i} \bigl(\log_{10}(y_i) - \log_{10}(m_i)\bigr)^2.$$ 

##### L2 weighted by error ($\chi^2$):

This is the usual $\chi^2$ definition. Though it works well if errors are gaussian. If uncertainties are weird, which seems to be the case here, may not be the best.

$$\chi^2_\mathrm{L2,error} = \sum_{i} \left( \frac{y_i - m_i}{\sigma_i} \right)^2$$
  

##### L2 with error in log-scale

In log-scale, $y\rightarrow\log_{10}(y)$, but error needs to be properly propagated. Note that log-scale overweights the small lags (uncertainty propagation gets super weird for small y).


$$\sigma_{\log y,i} = \frac{\sigma_i}{y_i \ln 10}$$

$$
\chi^2_\mathrm{L2, error,log} = \sum_{i} \left( \frac{\log_{10}(y_i) - \log_{10}(m_i)}{\sigma_{\log y,i}} \right)^2.$$

##### Reduced $\chi^2$

For reduced-\Chi^2:

$$\chi_{red}^2 = \frac{\chi^2}{N-k}$$

Where $N$ is the number of datapoints in $y$ and $k$ is the number of fitting parameters in the model. In our case here $k=3$ since we have the parameters $(C_0, C_1, t_0)$. 

This reduced form allows to at least evaluate how bad the errors are, as we should have:

- $\chi_{red}^2\approx 1$ for good fits
- $\chi_{red}^2\gg 1$ for underfit with error likely understimated
- $\chi_{red}^2\ll 1$ for overfit with overestimated errors 

  
##### L1 norm:

The L1 norm offers an alternative where the residuals are not squared such as the cost-function is less penalised by outliers. 

$$\text{L1}= \sum_{i} \bigl|y_i - m_i\bigr|$$ 



##### L1 norm w/ error 

Similarly to the L2 version, we can weight it by uncertainties. Noting that similarly to the L2 version this will work well as long as the uncertainties are good estimates of the real uncertainty. 

$$
\text{L1}_{\text{error}} = \sum_{i=1}^{N} \frac{\lvert y_i - m_i \rvert}{\sigma_i}
$$


##### L1 norm log-scale w/error:

To deal with the several orders of magnitude covered by both SF and $\tau$, logscale is a good option, though it may overweight small lags

$$
\text{L1}_{\text{log,weighted}} = \sum_{i=1}^{N} \frac{\lvert \log_{10}(y_i) - \log_{10}(m_i) \rvert}{\dfrac{\sigma_i}{y_i \ln(10)}}
$$


## TO DO list
- :heavy_check_mark: **@juliaroquette** Polish Lomb Scargle 
- :white_large_square: **@juliaroquette** Add Structure function 
- :white_large_square: Include alternative fitting to iminuit?
<!-- - :white_large_square: **@juliaroquette**  -->
<!-- - :white_large_square: **@juliaroquette**  -->


**@juliaroquette** Still under implementation, will include our codes for estimating timescales. 

