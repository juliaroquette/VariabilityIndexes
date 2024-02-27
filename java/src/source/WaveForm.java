/*-----------------------------------------------------------------------------
*
*                      Gaia CU7 variability
*
*         Copyright (C) 2005-2020 Gaia Data Processing and Analysis Consortium
*
*
* CU7 variability software is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* CU7 variability software is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this CU7 variability software; if not, write to the
* Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
* MA  02110-1301  USA
*
*-----------------------------------------------------------------------------
*/

package source;

/**
 *
 * Implements the WaveForm class for analyzing the waveform of a folded light
 * curve. This Java version is equivalent to the Python implementation provided
 * by @juliaroquette.
 * 
 * @author Mate Madarasz
 * @version 1.0
 *
 */

public class WaveForm {
	private FoldedLightCurve lc;
	private String waveformType;

	/**
	 * Initialize the WaveForm class.
	 *
	 * @param foldedLc     The folded light curve to be analyzed.
	 * @param waveformType The waveform analysis method to be used.
	 * @throws IllegalArgumentException If foldedLc is not an instance of
	 *                                  FoldedLightCurve.
	 */
	public WaveForm(FoldedLightCurve foldedLc, String waveformType) {
		if (!(foldedLc instanceof FoldedLightCurve)) {
			throw new IllegalArgumentException("lc must be an instance of FoldedLightCurve");
		}

		this.lc = foldedLc;
		this.waveformType = waveformType;
	}

	/**
	 * Apply a Savitzky-Golay filter to the folded light curve with non-uniform
	 * spacing.
	 *
	 * @param window  Window length of datapoints. Must be odd and smaller than the
	 *                light curve size.
	 * @param polynom The order of polynomial used. Must be smaller than the window
	 *                size.
	 * @return The smoothed folded light curve.
	 */
	public double[] unevenSavgol(int window, int polynom) {
		double[] phase = lc.getPhase();
		double[] magPhased = lc.getMagPhased();
		int N = lc.getN();

		double[] x = new double[3 * N];
		double[] y = new double[3 * N];

		// Concatenate phase and magPhased for -1, 0, +1 phases
		for (int i = 0; i < N; i++) {
			x[i] = phase[i] - 1;
			x[i + N] = phase[i];
			x[i + 2 * N] = phase[i] + 1;

			y[i] = magPhased[i];
			y[i + N] = magPhased[i];
			y[i + 2 * N] = magPhased[i];
		}

		// Apply the Savitzky-Golay filter
		double[] smoothedResults = SavitzkyGolayFilter.unevenSavgol(x, y, window, polynom);
		double[] resultArray = new double[phase.length];
		// Extract the filtered results corresponding to the original phase array
		System.arraycopy(smoothedResults, phase.length, resultArray, 0, phase.length);

		return resultArray;
	}

	/**
	 * Calculate the residual magnitude of the folded light curve using the
	 * specified waveform analysis method.
	 *
	 * @param window  Window size for the Savitzky-Golay filter.
	 * @param polynom Polynomial order for the Savitzky-Golay filter.
	 * @return The residual magnitude of the folded light curve.
	 * @throws IllegalArgumentException If the specified waveform analysis method is
	 *                                  not implemented.
	 */
	public double[] residualMagnitude(int window, int polynom) {
		double[] waveform;

		if ("uneven_savgol".equals(this.waveformType)) {
			waveform = this.unevenSavgol(window, polynom);
		} else {
			throw new IllegalArgumentException("Method _" + this.waveformType + "_ not implemented yet.");
		}

		double[] magPhased = this.lc.getMagPhased();
		double[] residual = new double[magPhased.length];
		for (int i = 0; i < magPhased.length; i++) {
			residual[i] = magPhased[i] - waveform[i];
		}
		return residual;
	}

	/**
	 * Calculate the residual magnitude of the folded light curve using default
	 * parameters.
	 *
	 * @return The residual magnitude of the folded light curve.
	 */
	public double[] residualMagnitude() {
		int window;
		int polynom = 3; // Default polynomial order

		if ("uneven_savgol".equals(this.waveformType)) {
			int wd = (int) Math.round(0.1 * this.lc.getN());
			if (wd % 2 == 0)
				wd += 1; // Ensure window is odd
			window = wd; // Use default window size calculated from the light curve
		} else {
			throw new IllegalArgumentException("Method _" + this.waveformType + "_ not implemented yet.");
		}

		return residualMagnitude(window, polynom); // Call the original method with parameters given
	}
}