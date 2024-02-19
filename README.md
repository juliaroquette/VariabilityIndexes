**@juliaroquette** This is a simplified version of the pacakge for deriving Q&M indexes, which keeps only the parts required for analysing Gaia DR3 light-curves. This should, in principle, work as a reference for Màté to translate it into Java. 

The `./jupyter/test.ipynb` Jupyter Notebook contains a very simple application test for Màté to use as a reference. 

The code itself is located in the folder `./variability`:
- `./variability/lightcurve.py` contains classes to read light-curves into two types of objects: `LightCurve` (simple light-curve) or `FoldedLightCurve` (it is just like a Light-curve, but has a characteristic `timescale`, hence it includes extra attributes related to the phase-folded light-curve for this `timescale`.
- `./variability/indexes.py` contains a class called `VariabilityIndex`, which takes in  `LightCurve` or  `FoldedLightCurve` objects and use two internal classes to estimate the Q&M indexes.
- `./variability/filtering.py` contains two classes related to data-filtering:
  - `Filtering` contains a few filtering methods that may be useful later on, although we are not using it at this initial moment.
  - `WaveForm` is a filtering class that takes in `FoldedLightCurve` objects and estimate a statistical waveform for them, based on a choice of method (`waveform_type`). At the moment, our tests indicate we will be using the method `uneven_savgol`, which is defined at the end of `./variability/filtering.py`.

**Last Update**: 22 February 2024
