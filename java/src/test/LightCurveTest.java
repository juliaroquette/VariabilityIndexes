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
import source.LightCurve;

/**
 *
 * Tests for verifying the behaviour of the LightCurve class. This test suite
 * covers constructor validation, property calculations, and the consistency of
 * LightCurve methods with expected outcomes taken from @juliaroquette's Python
 * implementation.
 * 
 * @author Mate Madarasz
 * @version 1.0
 *
 */

public class LightCurveTest {

	private LightCurve lc;
	private final double delta = 1e-15;

	/**
	 * Sets up the testing environment before each test. Initializes a LightCurve
	 * instance with predefined parameters to be used in subsequent tests.
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

		lc = new LightCurve(time, mag, err, mask);
	}

	/**
	 * Tests the LightCurve constructor for handling arrays of different lengths.
	 * Expecting an IllegalArgumentException to be thrown.
	 */
	@Test
	public void testConstructorWithDifferentLengths() {
		double[] time = { 1.0, 2.0 };
		double[] mag = { 1.0, 2.0, 3.0 };
		double[] err = { 1.0, 2.0 };
		Assertions.assertThrows(IllegalArgumentException.class, () -> {
			new LightCurve(time, mag, err);
		});
	}

	/**
	 * Verifies that the LightCurve constructor without a mask parameter initializes
	 * correctly.
	 */
	@Test
	public void testConstructorWithoutMask() {
		double[] time = { 1.0, 2.0 };
		double[] mag = { 1.0, 2.0 };
		double[] err = { 1.0, 2.0 };
		LightCurve lc = new LightCurve(time, mag, err);
		Assertions.assertNotNull(lc);
	}

	/**
	 * Tests the LightCurve constructor with a mask parameter for correct
	 * initialization.
	 */
	@Test
	public void testConstructorWithMask() {
		double[] time = { 1.0, 2.0, 3.0 };
		double[] mag = { 1.0, 2.0, 3.0 };
		double[] err = { 1.0, 2.0, 3.0 };
		boolean[] mask = { true, false, true };
		LightCurve lc = new LightCurve(time, mag, err, mask);
		Assertions.assertNotNull(lc);
	}

	/**
	 * Tests the correct number of points in the LightCurve.
	 */
	@Test
	public void testNumberOfPoints() {
		int expectedNumberOfPoints = 100;
		Assertions.assertEquals(expectedNumberOfPoints, lc.getN());
	}

	/**
	 * Verifies the time span of the LightCurve.
	 */
	@Test
	public void testTimeSpan() {
		double expectedTimeSpan = 80.0;
		Assertions.assertEquals(expectedTimeSpan, lc.getTimeSpan(), delta);
	}

	/**
	 * Tests the standard deviation calculation of the LightCurve.
	 */
	@Test
	public void testStandardDeviation() {
		double expectedSTD = 0.3517811819867572;
		Assertions.assertEquals(expectedSTD, lc.getStd(), delta);
	}

	/**
	 * Checks the mean value of the LightCurve.
	 */
	@Test
	public void testMean() {
		double expectedMean = -1.3682954979366874e-17;
		Assertions.assertEquals(expectedMean, lc.getMean(), delta);
	}

	/**
	 * Tests for the maximum value in the LightCurve.
	 */
	@Test
	public void testMaxValue() {
		double expectedMax = 0.49993706383693753;
		Assertions.assertEquals(expectedMax, lc.getMax(), delta);
	}

	/**
	 * Tests for the minimum value in the LightCurve.
	 */
	@Test
	public void testMinValue() {
		double expectedMin = -0.49993706383693753;
		Assertions.assertEquals(expectedMin, lc.getMin(), delta);
	}

	/**
	 * Verifies the maximum time value in the LightCurve.
	 */
	@Test
	public void testTimeMax() {
		double expectedMaxTime = 80.0;
		Assertions.assertEquals(expectedMaxTime, lc.getTimeMax(), delta);
	}

	/**
	 * Verifies the minimum time value in the LightCurve.
	 */
	@Test
	public void testTimeMin() {
		double expectedMinTime = 0.0;
		Assertions.assertEquals(expectedMinTime, lc.getTimeMin(), delta);
	}

	/**
	 * Tests the weighted average calculation, expecting NaN due to zero errors.
	 */
	@Test
	public void testWeightedAverage() {
		Assertions.assertEquals(Double.NaN, lc.getWeightedAverage());
	}

	/**
	 * Tests the median value of the LightCurve.
	 */
	@Test
	public void testMedian() {
		double expectedMedian = -4.898587196589413e-16;
		Assertions.assertEquals(expectedMedian, lc.getMedian(), delta);
	}
}
