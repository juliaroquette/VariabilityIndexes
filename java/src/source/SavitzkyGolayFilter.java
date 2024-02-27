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

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Implements the Savitzky-Golay smoothing filter for data with non-uniform
 * spacing. This Java version is equivalent to the Python implementation
 * provided by @juliaroquette, designed to apply a Savitzky-Golay filter to a
 * dataset where the data points are not equally spaced.
 *
 * @author Mate Madarasz
 * @version 1.0
 */

public class SavitzkyGolayFilter {

	/**
	 * Applies a Savitzky-Golay filter to an array with non-uniform spacing.
	 *
	 * @param x       Array, must be the same length as 'y'.
	 * @param y       Array to be smoothed.
	 * @param window  The window size for the filter, must be an odd integer smaller
	 *                than 'x'.
	 * @param polynom The order of the polynomial used, must be less than the window
	 *                size.
	 * @return The smoothed 'y' values as a double array.
	 * @throws IllegalArgumentException If input parameters do not meet
	 *                                  requirements.
	 */
	public static double[] unevenSavgol(double[] x, double[] y, int window, int polynom) {
		// Validate inputs
		if (x.length != y.length) {
			throw new IllegalArgumentException("x and y must be of the same size");
		}
		if (x.length < window) {
			throw new IllegalArgumentException("The data size must be larger than the window size");
		}
		if (window % 2 == 0) {
			throw new IllegalArgumentException("The window must be an odd integer");
		}
		if (polynom >= window) {
			throw new IllegalArgumentException("polynom must be less than window");
		}

		int halfWindow = window / 2;
		polynom += 1;

		double[] ySmoothed = new double[y.length];
		for (int i = 0; i < y.length; i++) {
			ySmoothed[i] = Double.NaN;
		}

		RealMatrix A = new Array2DRowRealMatrix(window, polynom);
		for (int i = halfWindow; i < x.length - halfWindow; i++) {
			double[] t = new double[window];
			for (int j = 0; j < window; j++) {
				t[j] = x[i - halfWindow + j] - x[i];
				for (int k = 0; k < polynom; k++) {
					A.setEntry(j, k, Math.pow(t[j], k)); // Fill the matrix with powers of t
				}
			}

			RealMatrix At = A.transpose();
			RealMatrix AtA = At.multiply(A);
			RealMatrix AtAInv = MatrixUtils.inverse(AtA);
			RealMatrix coeffs = AtAInv.multiply(At);

			// Apply the coefficients to the data within the window to smooth
			double smoothedValue = 0.0;
			for (int j = 0; j < window; j++) {
				smoothedValue += coeffs.getEntry(0, j) * y[i - halfWindow + j];
			}
			ySmoothed[i] = smoothedValue;
		}

		return ySmoothed;
	}
}
