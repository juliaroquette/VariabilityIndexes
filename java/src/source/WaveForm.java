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

import java.util.Arrays;

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
	private String waveformType;
	private double[] phase;
	private double[] magPhased;
	private int N;

	/**
	 * Initialize the WaveForm class using a FoldedLightCurve object.
	 *
	 * @param foldedLc     The folded light curve to be analyzed.
	 * @param waveformType The waveform analysis method to be used.
	 * @throws IllegalArgumentException If the FoldedLightCurve object is null.
	 */
	public WaveForm(FoldedLightCurve foldedLc, String waveformType) {
		if (foldedLc == null) {
			throw new IllegalArgumentException("FoldedLightCurve cannot be null");
		}

		this.waveformType = waveformType;
		this.phase = foldedLc.getPhase();
		this.magPhased = foldedLc.getMagPhased();
		this.N = this.magPhased.length;
	}

	/**
	 * Initialize the WaveForm class using phase and magnitude arrays directly.
	 *
	 * @param phase        The phase array of the folded light curve.
	 * @param magPhased    The magnitude array of the folded light curve
	 *                     corresponding to the phases.
	 * @param waveformType The waveform analysis method to be used.
	 * @throws IllegalArgumentException If either the phase or magPhased array is
	 *                                  null, or if they have different lengths.
	 */

	public WaveForm(double[] phase, double[] magPhased, String waveformType) {
		if (phase == null || magPhased == null) {
			throw new IllegalArgumentException("Phase and magPhased arrays cannot be null");
		}
		if (phase.length != magPhased.length) {
			throw new IllegalArgumentException("Phase and magPhased arrays must have the same length");
		}

		this.waveformType = waveformType;
		this.phase = Arrays.copyOf(phase, phase.length);
		this.magPhased = Arrays.copyOf(magPhased, magPhased.length);
		this.N = this.magPhased.length;
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
		// Preparation for Savitzky-Golay filter
		double[] x = new double[3 * this.N];
		double[] y = new double[3 * this.N];

		// Concatenate phase and magPhased for -1, 0, +1 phases
		for (int i = 0; i < this.N; i++) {
			x[i] = this.phase[i] - 1;
			x[i + this.N] = this.phase[i];
			x[i + 2 * this.N] = this.phase[i] + 1;

			y[i] = this.magPhased[i];
			y[i + this.N] = this.magPhased[i];
			y[i + 2 * this.N] = this.magPhased[i];
		}

		// Apply the Savitzky-Golay filter
		double[] smoothedResults = SavitzkyGolayFilter.unevenSavgol(x, y, window, polynom);
		double[] resultArray = new double[this.N];
		// Extract the filtered results corresponding to the original phase array
		System.arraycopy(smoothedResults, this.N, resultArray, 0, this.N);

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

		double[] residual = new double[this.N];
		for (int i = 0; i < this.N; i++) {
			residual[i] = this.magPhased[i] - waveform[i];
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
			int wd = (int) Math.round(0.1 * this.N);
			if (wd % 2 == 0)
				wd += 1; // Ensure window is odd
			window = wd; // Use default window size calculated from the light curve
		} else {
			throw new IllegalArgumentException("Method _" + this.waveformType + "_ not implemented yet.");
		}

		return residualMagnitude(window, polynom); // Call the original method with parameters given
	}
}