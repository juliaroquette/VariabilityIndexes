**@juliaroquette** Package under development for deriving Q&M indexes (and a few other variability indexes) for any time type of light-curves.

**Last Update**: 22 February 2024

In this current version, one can import and use it by doing:

```python
import sys
sys.path.append('PAT/TO/THE/PACKAGE/LOCATION')  
```

# `lightcurve` module:

Provides tools for loading light-curves as objects. Three distinct classes related to light-curves are currently included:

## `LightCurve` class

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
  lc_f = FoldedLightCurve(time=time, mag=mag, err=err, timescale=timescale)
```

where `timescale` is a timescale to be used for phase-folding the light-curve (for example, a variability period).

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
# `indexes`

## `VariabilityIndex``

```python
from variability.indexes import VariabilityIndex
```

# `filtering
**@juliaroquette** mostly implemented, but still need polishing and debugging. 

## `Filtering`

```python
from variability.filtering import Filtering
```
## `WaveForm`

```python
from variability.filtering import  WaveForm
```

## `uneven_savgol`

# `timescale`

**@juliaroquette** Still under implementation, will include our codes for estimating timescales. 

