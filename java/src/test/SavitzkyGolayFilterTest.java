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

package test;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;

import source.LightCurve;
import source.FoldedLightCurve;
import source.SavitzkyGolayFilter;

/**
 * Tests the SavitzkyGolayFilter implementation for accuracy by comparing the
 * filtered light curve against expected results obtained from the equivalent
 * Python implementation.
 * 
 * This test ensures that the Java implementation of the Savitzky-Golay filter
 * produces results consistent with those from @juliaroquette's Python
 * implementation, using the same input parameters and data.
 * 
 * @author Mate Madarasz
 * @version 1.0
 */

public class SavitzkyGolayFilterTest {

	private double[] phase;
	private double[] magPhased;
	private double[] expectedResults;
	private final double delta = 1e-14;

	/**
	 * Sets up the test environment by initializing a light curve, folding it with a
	 * specified period, and preparing the phase and magnitude arrays for testing.
	 */
	@BeforeEach
	public void setUp() {
		final int N = 100;
		double[] time = new double[N];
		double[] mag = new double[N];
		double[] err = new double[N];
		boolean[] mask = new boolean[N];
		final double period = 10.0;
		final double amplitude = 1.0;

		// Initialize the light curve data
		for (int i = 0; i < N; i++) {
			time[i] = 80.0 * i / (N - 1);
			mag[i] = 0.5 * amplitude * Math.sin(2 * Math.PI * time[i] / period);
			err[i] = 0.0;
			mask[i] = true;
		}

		LightCurve lc = new LightCurve(time, mag, err, mask);
		FoldedLightCurve flc = new FoldedLightCurve(lc, period);

		// Extract the phase and magnitudes for the folded light curve
		// Note that this was already tested in FoldedLightCurveTest.java
		this.phase = flc.getPhase();
		this.magPhased = flc.getMagPhased();
		this.expectedResults = ExpectedValuesForTests.EXPECTED_UNEVEN_SAVGOL;
	}

	/**
	 * Tests the unevenSavgol method of the SavitzkyGolayFilter class to ensure it
	 * produces expected results when applied to a folded light curve.
	 */
	@Test
	public void testUnevenSavgol() {
		int N = 100; // Number of data points
		int window = 11; // Window size
		int polynom = 3; // Polynomial order

		// Prepare extended arrays for phases and magnitudes
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
		double[] actualResults = new double[phase.length];
		// Extract the filtered results corresponding to the original phase array
		System.arraycopy(smoothedResults, phase.length, actualResults, 0, phase.length);

		// Assert that the filtered results match the expected values within a small
		// tolerance
		Assertions.assertArrayEquals(expectedResults, actualResults, delta);
	}
}
