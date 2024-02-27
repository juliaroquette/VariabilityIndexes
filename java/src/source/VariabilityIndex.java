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
import java.util.Arrays;
import java.util.function.DoublePredicate;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

/**
 * Implements the Q&M variability indexes from Cody et al. 2014 as an equivalent
 * Java code to @juliaroquette's Python implementation.
 * 
 * @author Mate Madarasz
 * @version 1.0
 */

@Getter
public class VariabilityIndex {
	private LightCurve lc;
	private MIndex MIndex;
	private QIndex qIndex;

	/**
	 * Constructs a VariabilityIndex instance for the given light curve.
	 *
	 * @param lc          the light curve to analyze.
	 * @param MPercentile the percentile value used in the M-index calculation.
	 * @param MIsFlux     indicates whether flux should be considered instead of
	 *                    magnitude.
	 * @throws IllegalArgumentException if lc is null.
	 */
	public VariabilityIndex(LightCurve lc, double MPercentile, boolean MIsFlux, double timescale,
			String waveformMethod) {
		if (lc == null) {
			throw new IllegalArgumentException("lc must be an instance of LightCurve");
		}
		this.lc = lc;
		this.MIndex = new MIndex(this, MPercentile, MIsFlux);
		this.qIndex = new QIndex(this, timescale, waveformMethod);
	}

	/**
	 * Inner class to calculate the M-index, a measure of variability based on the
	 * specified percentile.
	 */
	@Getter
	public class MIndex {
		private double percentile;
		private boolean isFlux;
		private VariabilityIndex parent;

		/**
		 * Constructs an MIndex instance for the specified parent VariabilityIndex.
		 *
		 * @param parent     the parent VariabilityIndex instance.
		 * @param percentile the percentile to use in the variability calculation.
		 * @param isFlux     true if the calculation is based on flux; false for
		 *                   magnitude.
		 */
		public MIndex(VariabilityIndex parent, double percentile, boolean isFlux) {
			this.parent = parent;
			setPercentile(percentile);
			this.isFlux = isFlux;
		}

		/**
		 * Sets the percentile value for variability calculation.
		 *
		 * @param newPercentile the new percentile value to set.
		 * @throws IllegalArgumentException if the percentile value is not between 0 and
		 *                                  49.
		 */
		public void setPercentile(double newPercentile) {
			if (newPercentile > 0 && newPercentile < 49) {
				this.percentile = newPercentile;
			} else {
				throw new IllegalArgumentException("Please enter a valid percentile (between 0. and 49.)");
			}
		}

		/**
		 * Generates a DoublePredicate representing the filter condition based on the
		 * specified percentile.
		 *
		 * @return a DoublePredicate for filtering values outside the specified
		 *         percentile range.
		 */
		private DoublePredicate getPercentileMask() {
			double lowerBound = percentile(this.parent.lc.getMag(), this.percentile);
			double upperBound = percentile(this.parent.lc.getMag(), 100 - this.percentile);

			return value -> value <= lowerBound || value >= upperBound;
		}

		/**
		 * Calculates and returns the value of the M-index.
		 *
		 * @return the calculated M-index value.
		 */
		public double getValue() {
			Mean mean = new Mean();
			double[] filteredMag = Arrays.stream(this.parent.lc.getMag()).filter(getPercentileMask()).toArray();
			double meanFilteredMag = mean.evaluate(filteredMag);
			double median = this.parent.lc.getMedian();
			double std = this.parent.lc.getStd();

			return (1 - 2 * (this.isFlux ? 1 : 0)) * (meanFilteredMag - median) / std;
		}
	}

	/**
	 * Inner class to calculate the Q-index
	 */
	@Getter
	public class QIndex {
		private double timescale;
		private String waveformMethod;
		private VariabilityIndex parent;

		/**
		 * Constructs a QIndex instance for the specified parent VariabilityIndex.
		 *
		 * @param parent         the parent VariabilityIndex instance.
		 * @param timescale      the timescale to use for folding the light curve.
		 * @param waveformMethod the method to analyze the waveform.
		 */
		public QIndex(VariabilityIndex parent, double timescale, String waveformMethod) {
			this.parent = parent;
			setTimescale(timescale);
			this.waveformMethod = waveformMethod;
		}
		
		/**
	     * Sets the timescale and ensures it's a positive value.
	     *
	     * @param timescale the new timescale to set.
	     */
	    public void setTimescale(double timescale) {
	        if (timescale > 0) {
	            this.timescale = timescale;
	        } else {
	            throw new IllegalArgumentException("Please enter a valid positive timescale");
	        }
	    }
		
		/**
		 * Calculates and returns the value of the Q-index.
		 *
		 * @return the calculated Q-index value.
		 */
		public double getValue() {
			FoldedLightCurve flc = new FoldedLightCurve(lc.getTime(),
					lc.getMag(),
					lc.getErr(),
					timescale,
					lc.getMask());
			
			WaveForm waveForm = new WaveForm(flc, waveformMethod);
			double[] residuals = waveForm.residualMagnitude();

			Variance variance = new Variance();
		    Mean mean = new Mean();

		    double varResiduals = variance.evaluate(residuals);
		    double varMagPhased = variance.evaluate(flc.getMagPhased());

		    double meanErrPhased = mean.evaluate(flc.getErrPhased());
		    double meanErrPhasedSquared = meanErrPhased * meanErrPhased;

		    return (varResiduals - meanErrPhasedSquared) / (varMagPhased - meanErrPhasedSquared);
		}
	}

	/**
	 * Calculates the percentile of the given data array.
	 *
	 * @param data       the data array to calculate the percentile for.
	 * @param percentile the percentile to calculate.
	 * @return the value at the specified percentile.
	 */
	private static double percentile(double[] data, double percentile) {
		Percentile p = new Percentile().withEstimationType(Percentile.EstimationType.R_7);

		return p.evaluate(data, percentile);
	}
}
