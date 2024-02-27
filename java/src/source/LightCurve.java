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

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.descriptive.rank.Min;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

/**
 *
 * Class representing a light curve, equivalent to the Python LightCurve class
 * written by @juliaroquette.
 * 
 * @author Mate Madarasz
 * @version 1.0
 *
 */

@Getter
@Setter
public class LightCurve {
	private double[] time;
	private double[] mag;
	private double[] err;
	private boolean[] mask;

	/**
	 * Initializes a new LightCurve object with time, magnitude, and error arrays,
	 * without a mask. All data points are included.
	 *
	 * @param time Array of time values.
	 * @param mag  Array of magnitude values.
	 * @param err  Array of error values.
	 * @throws IllegalArgumentException if the lengths of the time, mag, and err
	 *                                  arrays are not equal.
	 */
	public LightCurve(double[] time, double[] mag, double[] err) throws IllegalArgumentException {
		// Check if they have the same size
		validateArrayLengths(time, mag, err);
		this.time = time;
		this.mag = mag;
		this.err = err;
		// Create a mask that includes all data points.
		this.mask = new boolean[time.length];
		Arrays.fill(this.mask, true);
	}

	/**
	 * Initializes a new LightCurve object with time, magnitude, error arrays, and a
	 * mask to filter the data points.
	 *
	 * @param time Array of time values.
	 * @param mag  Array of magnitude values.
	 * @param err  Array of error values.
	 * @param mask Array of boolean values for filtering data points.
	 * @throws IllegalArgumentException if the lengths of the time, mag, err, and
	 *                                  mask arrays are not equal.
	 */
	public LightCurve(double[] time, double[] mag, double[] err, boolean[] mask) throws IllegalArgumentException {
		// Check if they have the same size.
		validateArrayLengths(time, mag, err);

		// As mask is given as well, check if it has the same size as the other arrays.
		if (mask.length != time.length) {
			throw new IllegalArgumentException("Mask array length must match time, mag, and err array lengths.");
		}

		this.time = IntStream.range(0, mask.length).filter(i -> mask[i]).mapToDouble(i -> time[i]).toArray();
		this.mag = IntStream.range(0, mask.length).filter(i -> mask[i]).mapToDouble(i -> mag[i]).toArray();
		this.err = IntStream.range(0, mask.length).filter(i -> mask[i]).mapToDouble(i -> err[i]).toArray();
		this.mask = mask;
	}

	/**
	 * Validates that the lengths of the time, mag, and err arrays are equal. Throws
	 * IllegalArgumentException if they are not.
	 *
	 * @param time Array of time values.
	 * @param mag  Array of magnitude values.
	 * @param err  Array of error values.
	 * @throws IllegalArgumentException if the array lengths are not equal.
	 */
	private void validateArrayLengths(double[] time, double[] mag, double[] err) throws IllegalArgumentException {
		if (time.length != mag.length || mag.length != err.length) {
			throw new IllegalArgumentException("Time, magnitude, and error arrays must have the same length.");
		}
	}

	/**
	 * Returns the number of data points in the light curve.
	 *
	 * @return Number of data points.
	 */
	public int getN() {
		return mag.length;
	}

	/**
	 * Returns the total time span of the light curve.
	 *
	 * @return Light curve time span.
	 */
	public double getTimeSpan() {
		return Arrays.stream(time).max().getAsDouble() - Arrays.stream(time).min().getAsDouble();
	}

	/**
	 * Returns the standard deviation of the magnitude values.
	 *
	 * @return Standard deviation.
	 */
	public double getStd() {
		double average = Arrays.stream(mag).average().orElse(Double.NaN);
		return Math.sqrt(Arrays.stream(mag).map(m -> m - average).map(m -> m * m).average().orElse(Double.NaN));
	}

	/**
	 * Returns the mean of the magnitude values.
	 *
	 * @return Mean value.
	 */
	public double getMean() {
		Mean mean = new Mean();
		return mean.evaluate(mag);
	}

	/**
	 * Returns the maximum magnitude value.
	 *
	 * @return Maximum magnitude value.
	 */
	public double getMax() {
		Max max = new Max();
		return max.evaluate(mag);
	}

	/**
	 * Returns the minimum magnitude value.
	 *
	 * @return Minimum magnitude value.
	 */
	public double getMin() {
		Min min = new Min();
		return min.evaluate(mag);
	}

	/**
	 * Returns the maximum time value.
	 *
	 * @return Maximum time value.
	 */
	public double getTimeMax() {
		Max max = new Max();
		return max.evaluate(time);
	}

	/**
	 * Returns the minimum time value.
	 *
	 * @return Minimum time value.
	 */
	public double getTimeMin() {
		Min min = new Min();
		return min.evaluate(time);
	}

	/**
	 * Returns the median of the magnitude values.
	 *
	 * @return Median value.
	 */
	public double getMedian() {
		Median median = new Median();
		return median.evaluate(mag);
	}

	/**
	 * Returns the weighted average of the magnitude values, where weights are the
	 * inverse square of the error values.
	 *
	 * @return Weighted average.
	 */
	public double getWeightedAverage() {
		double weightedSum = IntStream.range(0, mag.length).mapToDouble(i -> mag[i] / (err[i] * err[i])).sum();
		double weightSum = Arrays.stream(err).map(e -> 1 / (e * e)).sum();
		return weightedSum / weightSum;
	}
}