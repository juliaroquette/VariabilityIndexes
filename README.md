**@juliaroquette** Package under development for deriving Q&M indexes (and a few other variability indexes) for any time type of light-curves.

**Last Update**: 22 February 2024



# `lightcurve` module:

Provides tools for loading light-curves as objects. Three distinct classes related to light-curves are currently included:

## `LightCurve` class

```python
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