require 'spec_helper'
require_relative '../../fixtures/noncentral_chi_squared_vectors'

describe RubyStatistics::Distribution::NonCentralChiSquared do
  let(:fixtures) { RubyStatistics::Distribution::NonCentralChiSquaredFixtures }

  # All probabilities are compared with a mixed absolute/relative
  # tolerance: the implementation targets ~1e-13 relative accuracy.
  def expect_probability_close(actual, expected)
    tolerance = [1.0e-11 * expected.abs, 1.0e-13].max
    expect(actual).to be_within(tolerance).of(expected)
  end

  describe '#initialize' do
    it 'accepts positive degrees of freedom and a non-negative noncentrality' do
      expect { described_class.new(1, 0) }.not_to raise_error
      expect { described_class.new(2.5, 3.75) }.not_to raise_error
      expect { described_class.new(10) }.not_to raise_error # lambda defaults to 0
    end

    it 'rejects non-positive degrees of freedom' do
      expect { described_class.new(0, 1) }.to raise_error(ArgumentError)
      expect { described_class.new(-2.5, 1) }.to raise_error(ArgumentError)
    end

    it 'rejects a negative noncentrality parameter' do
      expect { described_class.new(3, -0.5) }.to raise_error(ArgumentError)
    end
  end

  describe '#mean and #variance' do
    it 'matches the theoretical moments k + lambda and 2(k + 2 lambda)' do
      [[4, 0], [3.5, 2.5], [10, 500], [0.7, 1.3]].each do |k, lambda|
        dist = described_class.new(k, lambda)
        expect(dist.mean).to eq k + lambda
        expect(dist.variance).to eq 2.0 * (k + 2.0 * lambda)
      end
    end
  end

  describe 'offline reference vectors' do
    it 'matches the reference cdf, survival and log-survival values' do
      fixtures::VECTORS.each do |k, lambda, x, cdf, sf, log_sf|
        dist = described_class.new(k, lambda)

        expect_probability_close(dist.cumulative_function(x), cdf)
        expect_probability_close(dist.survival_function(x), sf)
        expect(dist.log_survival_function(x)).to be_within(1.0e-10).of(log_sf)
      end
    end

    it 'matches the reference quantiles' do
      fixtures::QUANTILES.each do |k, lambda, p, quantile|
        dist = described_class.new(k, lambda)
        result = dist.inverse_cumulative_function(p)

        expect(result).to be_within(1.0e-9 * [quantile, 1.0].max).of(quantile)
      end
    end
  end

  describe 'degenerate case lambda = 0' do
    it 'coincides with the central chi-squared distribution of the library' do
      [1, 2, 3.5, 7, 12].each do |k|
        central = RubyStatistics::Distribution::ChiSquared.new(k)
        noncentral = described_class.new(k, 0.0)

        [0.3, 1.5, 5.0, 11.0, 25.0].each do |x|
          expect(noncentral.cumulative_function(x)).to be_within(1.0e-12).of(central.cumulative_function(x))
          expect(noncentral.survival_function(x)).to be_within(1.0e-12).of(1.0 - central.cumulative_function(x))
        end
      end
    end

    it 'satisfies the closed form of the central chi-squared with 2 degrees of freedom' do
      dist = described_class.new(2, 0.0)

      [0.1, 1.0, 4.0, 12.0, 40.0].each do |x|
        expect(dist.cumulative_function(x)).to be_within(1.0e-13).of(1.0 - Math.exp(-x / 2.0))
        expect(dist.survival_function(x)).to be_within(1.0e-13).of(Math.exp(-x / 2.0))
        expect(dist.log_survival_function(x)).to be_within(1.0e-10).of(-x / 2.0)
      end
    end
  end

  describe 'mathematical identities' do
    it 'satisfies F(x; 1, lambda) = Phi(sqrt(x) - sqrt(lambda)) - Phi(-sqrt(x) - sqrt(lambda))' do
      normal = RubyStatistics::Distribution::StandardNormal.new

      [0.0, 0.5, 2.5, 9.0].each do |lambda|
        dist = described_class.new(1, lambda)
        root = Math.sqrt(lambda)

        [0.2, 1.0, 3.5, 9.0, 25.0].each do |x|
          sqrt_x = Math.sqrt(x)
          expected = normal.cumulative_function(sqrt_x - root) - normal.cumulative_function(-sqrt_x - root)

          expect(dist.cumulative_function(x)).to be_within(1.0e-12).of(expected)
        end
      end
    end

    it 'satisfies cdf + survival = 1 across the parameter range' do
      [[0.5, 0.0], [1, 0.5], [3.5, 2.5], [10, 5], [4, 1000], [50, 500]].each do |k, lambda|
        dist = described_class.new(k, lambda)

        [0.1, 1.0, 5.0, 20.0, 100.0].each do |scale|
          x = scale * (k + lambda) / 10.0
          expect(dist.cumulative_function(x) + dist.survival_function(x)).to be_within(1.0e-11).of(1.0)
        end
      end
    end

    it 'satisfies survival = exp(log_survival) away from underflow' do
      [[3.5, 2.5], [10, 5], [6, 40]].each do |k, lambda|
        dist = described_class.new(k, lambda)

        [1.0, 5.0, 20.0, 60.0].each do |x|
          sf = dist.survival_function(x)
          next if sf.zero?

          expect(Math.exp(dist.log_survival_function(x))).to be_within(1.0e-10 * sf).of(sf)
        end
      end
    end

    it 'is monotonic: the cdf never decreases and never exceeds [0, 1]' do
      [[3.5, 2.5], [4, 1000]].each do |k, lambda|
        dist = described_class.new(k, lambda)
        previous = 0.0

        30.times do |i|
          x = (i + 1) * (k + lambda) / 15.0
          cdf = dist.cumulative_function(x)

          expect(cdf).to be >= previous - 1.0e-15
          expect(cdf).to be >= 0.0
          expect(cdf).to be <= 1.0
          previous = cdf
        end
      end
    end
  end

  describe 'boundary values' do
    let(:dist) { described_class.new(3.5, 2.5) }

    it 'handles zero and negative values' do
      expect(dist.cumulative_function(0)).to eq 0.0
      expect(dist.cumulative_function(-3)).to eq 0.0
      expect(dist.survival_function(0)).to eq 1.0
      expect(dist.survival_function(-3)).to eq 1.0
      expect(dist.log_survival_function(0)).to eq 0.0
    end

    it 'handles infinite values' do
      expect(dist.cumulative_function(Float::INFINITY)).to eq 1.0
      expect(dist.survival_function(Float::INFINITY)).to eq 0.0
      expect(dist.log_survival_function(Float::INFINITY)).to eq(-Float::INFINITY)
    end
  end

  describe '#inverse_cumulative_function' do
    let(:dist) { described_class.new(6.5, 3.2) }

    it 'returns the exact endpoints for probabilities 0 and 1' do
      expect(dist.inverse_cumulative_function(0.0)).to eq 0.0
      expect(dist.inverse_cumulative_function(1.0)).to eq Float::INFINITY
    end

    it 'rejects probabilities outside [0, 1]' do
      expect { dist.inverse_cumulative_function(-0.1) }.to raise_error(ArgumentError)
      expect { dist.inverse_cumulative_function(1.1) }.to raise_error(ArgumentError)
    end

    it 'round-trips with the cdf across the probability range' do
      [1.0e-8, 1.0e-4, 0.01, 0.5, 0.9, 0.99, 1.0 - 1.0e-6].each do |p|
        x = dist.inverse_cumulative_function(p)
        expect(dist.cumulative_function(x)).to be_within(1.0e-9 * p).of(p)
      end
    end

    it 'round-trips with the survival function in the extreme upper tail' do
      [1.0e-8, 1.0e-12, 1.0e-16].each do |tail|
        # 1 - tail is not exactly representable; the effective target is
        # the double 1.0 - (1.0 - tail), which is what gets inverted.
        probability = 1.0 - tail
        effective_tail = 1.0 - probability
        x = dist.inverse_cumulative_function(probability)
        expect(dist.survival_function(x)).to be_within(1.0e-7 * effective_tail).of(effective_tail)
      end
    end

    it 'inverts probabilities many orders of magnitude below the mean' do
      [1.0e-50, 1.0e-300].each do |p|
        x = dist.inverse_cumulative_function(p)
        expect(dist.cumulative_function(x)).to be_within(1.0e-9 * p).of(p)
      end
    end

    it 'round-trips x through cdf and inverse cdf' do
      [0.5, 3.0, 10.0, 30.0].each do |x|
        p = dist.cumulative_function(x)
        expect(dist.inverse_cumulative_function(p)).to be_within(1.0e-9 * x).of(x)
      end
    end

    it 'raises a ConvergenceError when the iteration budget is exhausted' do
      expect { dist.inverse_cumulative_function(0.5, max_iterations: 1) }
        .to raise_error(described_class::ConvergenceError)
    end
  end

  describe 'explicit non-convergence status' do
    let(:dist) { described_class.new(10, 500) }

    it 'raises ConvergenceError from the float-returning methods' do
      expect { dist.cumulative_function(700, max_terms: 3) }
        .to raise_error(described_class::ConvergenceError)
      expect { dist.survival_function(700, max_terms: 3) }
        .to raise_error(described_class::ConvergenceError)
      expect { dist.log_survival_function(700, max_terms: 3) }
        .to raise_error(described_class::ConvergenceError)
    end

    it 'reports the remainder bound and the flag through the evaluation objects' do
      evaluation = dist.cumulative_evaluation(700, max_terms: 3)

      expect(evaluation.converged).to be false
      expect(evaluation.error_bound).to be > 1.0e-13
      expect(evaluation.terms).to eq 3
    end

    it 'reports a tiny remainder bound for a converged evaluation' do
      evaluation = dist.survival_evaluation(700)

      expect(evaluation.converged).to be true
      expect(evaluation.error_bound).to be < 1.0e-12
      expect(evaluation.terms).to be > 0
    end
  end

  describe 'extreme tails' do
    it 'returns 0.0 for a survival probability below the double range' do
      expect(described_class.new(4, 2).survival_function(5000)).to eq 0.0
    end

    it 'still computes the log-survival far below the double range' do
      log_sf = described_class.new(4, 2).log_survival_function(5000)

      expect(log_sf).to be_within(1.0e-10).of(-2400.2929682157934)
    end

    it 'keeps relative accuracy for a survival probability near 1e-300' do
      dist = described_class.new(2, 0.5)
      sf = dist.survival_function(700)
      log_sf = dist.log_survival_function(700)

      expect(sf).to be > 0.0
      expect(Math.log(sf)).to be_within(1.0e-9).of(log_sf)
    end
  end
end
