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
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import source.LightCurve;
import source.VariabilityIndex;
import source.FoldedLightCurve;

/**
 * Tests for verifying the behaviour of the VariabilityIndex class. Expected
 * values were taken from @juliaroquette's Python implementation.
 * 
 * @author Mate Madarasz
 * @version 1.0
 *
 */

public class VariabilityIndexTest {
	private FoldedLightCurve flc;
	private final double delta = 1e-15;

	/**
	 * Sets up the testing environment before each test. This includes initializing
	 * a light curve and folding it with a specified period.
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

		for (int i = 0; i < N; i++) {
			time[i] = 80.0 * i / (N - 1);
			mag[i] = 0.5 * amplitude * Math.sin(2 * Math.PI * time[i] / period);
			err[i] = 0.0;
			mask[i] = true;
		}

		LightCurve lc = new LightCurve(time, mag, err, mask);
		flc = new FoldedLightCurve(lc, period);
	}

	/**
	 * Tests the calculation of the M-index using a predefined light curve.
	 */
	@Test
	public void testMIndexCalculation() {
		VariabilityIndex vi = new VariabilityIndex(flc, 10.0, false);
		double expectedMIndex = 1.4635198358404185e-15;
		assertEquals(expectedMIndex, vi.getMIndex().getValue(), delta, "The M-index calculation is incorrect.");
	}

	/**
	 * Tests the response of the MIndex constructor to an invalid percentile value.
	 */
	@Test
	public void testMIndexWithInvalidPercentile() {
		assertThrows(IllegalArgumentException.class, () -> {
			new VariabilityIndex(flc, 50.0, false);
		}, "Expected to throw IllegalArgumentException for invalid percentile.");
	}
}
