**@juliaroquette** Package under development for deriving Q&M indexes (and a few other variability indexes) for any time type of light-curves.


**Last Update**: 22 February 2024

In this current version, one can import and use it by doing:

```python
import sys
sys.path.append('PAT/TO/THE/PACKAGE/LOCATION')  
```

[[_TOC_]]

# `lightcurve` module:

Provides tools for loading light-curves as objects. Three distinct classes related to light-curves are currently included:

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
- `time_span`: Total time-span of the light curve ($t_{max}-t_{min}$).
- `std`: Standard deviation of the magnitude values.
- `mean` : Mean of the magnitude values.
- `weighted_average`: Weighted average of the magnitude values.
- `median`: Median of the magnitude values.
- `min`: Minimum value of the magnitudes.
- `max`: Maximum value of the magnitudes.
- `time_max`: Maximum value of the observation times.
- `time_min`: Minimum value of the observation times.
- `ptp`: Peak-to-peak amplitude of the magnitude values. Defined as the difference between the median values for the datapoints in the 5% outermost tails of the distribution.

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
- `phase`: The phase values of the folded light curve (between 0 and 1).
- `mag_pahsed`: The magnitude values of the folded light curve, sorted based on phase.
- `err_pahsed`: The error values of the folded light curve, sorted based on phase.


## `SyntheticLightCurve`

**@juliaroquette** Still under implementation, will allow to generate synthetic light-curves for given observational windows. 

```python 
from variability.lightcurve import FoldedLightCurve
```


## TO DO list

<details>
:white_large_square: **@juliaroquette** It may be worth it consider the possibility of merging `LightCurve` and `FoldedLightCurve` into a single class. <- Consider that after the `timescale.py` package has been implemented. 

</details>

# `indexes`

## `VariabilityIndex``

To instantiate a `VariabilityIndex` object:

```python
from variability.indexes import VariabilityIndex

var = VariabilityIndex(lc_p, timescale=period)
```

you are expected to pass in a `LightCurve` object, or a `FoldedLightCurve` object. 

**Note that some variability indexes, like the Q-index itself, require either a `timescale` argument or a `FoldedLightCurve` instance (which already have an instance `timescale`).



### `VariabilityIndex.M_index``

### `VariabilityIndex.Q_index``

## TO DO list
- :white_large_square: **@juliaroquette** Implement the Abbe variability index into `indexes.py`

# `filtering
**@juliaroquette** mostly implemented, but still need polishing and debugging. 


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

```python
from variability.filtering import  WaveForm
```
<details>

### `WaveForm.residual_magnitude`

### `WaveForm.circular_rolling_average_number`

### `WaveForm.savgol`

### `WaveForm.circular_rolling_average_phase`

### `WaveForm.waveform_H22`

### `WaveForm.waveform_Cody`

### `WaveForm.uneven_savgol`

</details>

## `uneven_savgol`

# `timescale`

**@juliaroquette** Still under implementation, will include our codes for estimating timescales. 

