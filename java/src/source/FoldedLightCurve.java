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

/**
 * Represents a folded light curve, extending the functionality of the
 * LightCurve class to include folding based on a specified timescale. This Java
 * implementation is an equivalent one to a Python version created
 * by @juliaroquette, which defines a base class for handling light curves.
 * 
 * @author Mate Madarasz
 * @version 1.0
 */

@Getter
@Setter
public class FoldedLightCurve extends LightCurve {
	private double timescale;
	private double[] phase;
	private double[] magPhased;
	private double[] errPhased;

	/**
	 * Constructs a FoldedLightCurve object directly from arrays of time, magnitude,
	 * and error values, along with a specified timescale and mask.
	 * 
	 * @param time      Array of time values.
	 * @param mag       Array of magnitude values.
	 * @param err       Array of error values.
	 * @param timescale The timescale used for folding the light curve.
	 * @param mask      An optional boolean array to filter data points.
	 */
	public FoldedLightCurve(double[] time, double[] mag, double[] err, double timescale, boolean[] mask) {
		super(time, mag, err, mask);
		this.timescale = timescale;
		calculatePhaseAndSort();
	}

	/**
	 * Constructs a FoldedLightCurve object from an existing LightCurve object and a
	 * specified timescale.
	 * 
	 * @param lc        The LightCurve object to fold.
	 * @param timescale The timescale used for folding the light curve.
	 */
	public FoldedLightCurve(LightCurve lc, double timescale) {
		this(lc.getTime(), lc.getMag(), lc.getErr(), timescale, lc.getMask());
	}

	/**
	 * Calculates the phase for each observation in the light curve based on the
	 * specified timescale and sorts the light curve data by phase. This method
	 * updates the phase, magPhased, and errPhased arrays, aligning the data points
	 * according to their phase in the folding period.
	 */
	private void calculatePhaseAndSort() {
		// Calculate the phase of each time point as a fraction of the timescale,
		// resulting in values between 0 and 1.
		phase = Arrays.stream(this.getTime()).map(t -> (t % timescale) / timescale).toArray();

		// Create an array of indices to sort the phase array, and subsequently use it
		// to sort magnitude and error arrays.
		Integer[] indices = IntStream.range(0, this.getN()).boxed().toArray(Integer[]::new);
		Arrays.sort(indices, (i, j) -> Double.compare(phase[i], phase[j]));

		double[] sortedPhase = new double[this.getN()];
		double[] sortedMag = new double[this.getN()];
		double[] sortedErr = new double[this.getN()];

		for (int i = 0; i < this.getN(); i++) {
			int index = indices[i];
			sortedPhase[i] = phase[index];
			sortedMag[i] = this.getMag()[index];
			sortedErr[i] = this.getErr()[index];
		}

		this.phase = sortedPhase;
		this.magPhased = sortedMag;
		this.errPhased = sortedErr;
	}
}
