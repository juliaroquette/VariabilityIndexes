**@juliaroquette** This is a simplified version of the pacakge for deriving Q&M indexes, which keeps only the parts required for analysing Gaia DR3 light-curves. This should, in principle, work as a reference for Màté to translate it into Java. 

The [`./jupyter/test_GaiaDR3data.ipynb`](jupyter/test_GaiaDR3data.ipynb) Jupyter Notebook contains a simple application test using real Gaia DR3 light-curves for Màté to use as a reference. 
  - [`jupyter/data`](jupyter/data) includes `csv` files with testing light-curves.


The [`./jupyter/test.ipynb` ](jupyter/test.ipynb) Jupyter Notebook contains a very simple application test for Màté to use as a reference. 

The code itself is located in the folder [`./variability`](variability):

- `[./variability/lightcurve.py`](variability/lightcurve.py) contains classes to read light-curves into two types of objects: `LightCurve` (simple light-curve) or `FoldedLightCurve` (it is just like a Light-curve, but has a characteristic `timescale`, hence it includes extra attributes related to the phase-folded light-curve for this `timescale`.

- [`./variability/indexes.py`](variability/indexes.py) contains a class called `VariabilityIndex`, which takes in  `LightCurve` or  `FoldedLightCurve` objects and uses two internal classes to estimate the Q&M indexes.

-  [`./variability/filtering.py`](variability/filtering.py)` contains two classes related to data-filtering:
  - `Filtering` contains a few filtering methods that may be useful later on, although we are not using it at this initial moment.
  - `WaveForm` is a filtering class that takes in `FoldedLightCurve` objects and estimate a statistical waveform for them, based on a choice of method (`waveform_type`). At the moment, our tests indicate we will be using the method `uneven_savgol`, which is defined at the end of `[`./variability/filtering.py`](variability/filtering.py)`.

- [`./variability/timescales.py`](variability/timescales.py): contains a class called `TimeScale` which is applied to get a time-scale for the light-curves. Only the Lomb-Scargle periodogram is currently implemented. 
  - `get_LSP_period`: estimates a timescale using the Lomb-Scargle Periogram 

**Last Update**: 26 March 2024

- `Q_index`, `FoldedLightCurve` and `WaveForm` have been refactored such as now `timescale` is a property of `FoldedLightCurves`
- [`test_GaiaDR3data.ipynb`](jupyter/test_GaiaDR3data.ipynb) includes a test case using Gaia DR3 light-curves.

**Previous major update:** 22 February 2024