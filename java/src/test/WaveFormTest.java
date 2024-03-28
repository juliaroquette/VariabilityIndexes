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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import source.WaveForm;
import source.LightCurve;
import source.FoldedLightCurve;

/**
 *
 * Tests the WaveForm class to ensure it accurately computes the residual
 * magnitudes of a folded light curve.
 * 
 * This test ensures that the Java implementation of the WaveForm class produces
 * results consistent with those from @juliaroquette's Python implementation,
 * using the same input parameters and data.
 * 
 * @author Mate Madarasz
 * @version 1.0
 *
 */

public class WaveFormTest {
	private WaveForm waveformFromLightCurve;
	private WaveForm waveformFromParameters;
	private final double delta = 1e-14;
	private double[] expectedResiduals;

	/**
	 * Sets up the test environment by initializing a light curve, folding it with a
	 * specified period, prepares a WaveForm instance for testing.
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

		// Create WaveForm using the Folded Light Curve
		this.waveformFromLightCurve = new WaveForm(flc, "uneven_savgol");
		// Create WaveForm using the phase and magPhased arrays
		double[] phase = flc.getPhase();
		double[] magPhased = flc.getMagPhased();
		this.waveformFromParameters = new WaveForm(phase, magPhased, "uneven_savgol");

		this.expectedResiduals = ExpectedValuesForTests.EXPECTED_RESIDUALS_FOR_QINDEX;
	}

	/**
	 * Verifies correct residual magnitude calculation with specified parameters,
	 * using the waveformFromLightCurve object.
	 */
	@Test
	public void testResidualMagnitudeWithParameters() {
		int window = 11;
		int polynom = 3;

		double[] actualResiduals = waveformFromLightCurve.residualMagnitude(window, polynom);

		Assertions.assertArrayEquals(expectedResiduals, actualResiduals, delta,
				"Residual magnitudes do not match expected values with parameters.");
	}

	/**
	 * Checks residual magnitude calculation with default parameters, using the
	 * waveformFromLightCurve object.
	 */
	@Test
	public void testResidualMagnitudeWithoutParameters() {
		double[] actualResiduals = waveformFromLightCurve.residualMagnitude();

		Assertions.assertArrayEquals(expectedResiduals, actualResiduals, delta,
				"Residual magnitudes do not match expected default values.");
	}

	/**
	 * Verifies correct residual magnitude calculation with specified parameters,
	 * using the waveformFromParameters object.
	 */
	@Test
	public void testResidualMagnitudeWithParametersSecondConstructor() {
		int window = 11;
		int polynom = 3;

		double[] actualResiduals = waveformFromParameters.residualMagnitude(window, polynom);

		Assertions.assertArrayEquals(expectedResiduals, actualResiduals, delta,
				"Residual magnitudes do not match expected values with parameters.");
	}

	/**
	 * Checks residual magnitude calculation with default parameters, using the
	 * waveformFromParameters object.
	 */
	@Test
	public void testResidualMagnitudeWithoutParametersSecondConstructor() {
		double[] actualResiduals = waveformFromParameters.residualMagnitude();

		Assertions.assertArrayEquals(expectedResiduals, actualResiduals, delta,
				"Residual magnitudes do not match expected default values.");
	}
}
