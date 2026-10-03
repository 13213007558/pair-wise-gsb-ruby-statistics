require 'spec_helper'

describe RubyStatistics::Distribution::MultivariateNormal do
  let(:infinity) { Float::INFINITY }

  describe '.rectangular_probability' do
    it 'multiplies independent univariate probabilities' do
      result = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[1.0, 0.0], [0.0, 4.0]],
        lower_bound: [-1.0, 0.0],
        upper_bound: [1.0, infinity],
        error_budget: 0.0001
      )

      expected = (Math.erf(1 / Math.sqrt(2)) * 0.5)
      expect(result.probability).to be_within(1e-12).of(expected)
      expect(result.algorithm).to eq('independent_product')
      expect(result).to be_exact
      expect(result.samples_used).to eq(0)
    end

    it 'computes a two-dimensional positive quadrant with the quadrant identity' do
      result = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[1.0, 0.5], [0.5, 1.0]],
        lower_bound: [0.0, 0.0],
        upper_bound: [infinity, infinity],
        error_budget: 1e-7
      )

      expect(result.probability).to be_within(1e-9).of(1.0 / 3.0)
      expect(result.algorithm).to eq('bivariate_quadrant_formula')
      expect(result.strict_error_bound).to eq(false)
      expect(result.deterministic).to eq(true)
    end

    it 'handles completely correlated variables as one random direction' do
      result = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[1.0, 1.0], [1.0, 1.0]],
        lower_bound: [-1.0, -1.0],
        upper_bound: [1.0, 1.0],
        error_budget: 0.0001
      )

      expect(result.probability).to be_within(1e-12).of(0.6826894921370859)
      expect(result.rank).to eq(1)
      expect(result.rank_deficient).to eq(true)
      expect(result.near_singular).to eq(true)
      expect(result).to be_exact
    end

    it 'respects a zero-variance coordinate inside the interval' do
      result = described_class.rectangular_probability(
        mean: [0.0, 1.0],
        covariance: [[1.0, 0.0], [0.0, 0.0]],
        lower_bound: [0.0, 1.0],
        upper_bound: [infinity, 1.0],
        error_budget: 0.0001
      )

      expect(result.probability).to be_within(1e-12).of(0.5)
      expect(result.rank).to eq(1)
    end

    it 'returns zero when a zero-variance coordinate is outside the interval' do
      result = described_class.rectangular_probability(
        mean: [0.0, 1.0],
        covariance: [[1.0, 0.0], [0.0, 0.0]],
        lower_bound: [0.0, 0.0],
        upper_bound: [infinity, 0.5],
        error_budget: 0.0001
      )

      expect(result.probability).to eq(0.0)
      expect(result).to be_exact
    end

    it 'handles a degenerate distribution concentrated at one point' do
      inside = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[0.0, 0.0], [0.0, 0.0]],
        lower_bound: [0.0, 0.0],
        upper_bound: [0.0, 0.0],
        error_budget: 0.0001
      )
      outside = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[0.0, 0.0], [0.0, 0.0]],
        lower_bound: [1.0, 0.0],
        upper_bound: [1.0, 0.0],
        error_budget: 0.0001
      )

      expect(inside.probability).to eq(1.0)
      expect(outside.probability).to eq(0.0)
      expect(inside.rank).to eq(0)
      expect(inside).to be_exact
    end

    it 'estimates a rank-deficient three-dimensional event with a seeded statistical error' do
      direction_one = [1.0, 1.0, 0.0]
      direction_two = [0.0, 1.0, 1.0]
      covariance = Array.new(3) do |i|
        Array.new(3) do |j|
          direction_one[i] * direction_one[j] + direction_two[i] * direction_two[j]
        end
      end

      result = described_class.rectangular_probability(
        mean: [0.0, 0.0, 0.0],
        covariance: covariance,
        lower_bound: [-infinity, -infinity, -infinity],
        upper_bound: [infinity, 0.0, infinity],
        error_budget: 0.04,
        confidence_level: 0.95,
        seed: 7,
        max_samples: 5000
      )

      expect(result.probability).to be_within(0.035).of(0.5)
      expect(result.estimated_error).to be <= 0.04
      expect(result.budget_exhausted).to eq(false)
      expect(result.strict_error_bound).to eq(false)
      expect(result.random_seed).to eq(7)
      expect(result.samples_used).to eq(5000)
    end

    it 'reports budget exhaustion instead of claiming a statistical bound' do
      result = described_class.rectangular_probability(
        mean: [0.0, 0.0, 0.0],
        covariance: [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]],
        lower_bound: [0.0, 0.0, 0.0],
        upper_bound: [0.1, 0.1, 0.1],
        error_budget: 0.001,
        seed: 7,
        max_samples: 100
      )

      expect(result.budget_exhausted).to eq(true)
      expect(result.strict_error_bound).to eq(false)
      expect(result.estimated_error).to be > 0.001
      expect(result.samples_used).to eq(100)
    end

    it 'is deterministic for the same seed in randomized dimensions' do
      arguments = {
        mean: [0.0, 0.0, 0.0],
        covariance: [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]],
        lower_bound: [0.0, 0.0, 0.0],
        upper_bound: [infinity, infinity, infinity],
        error_budget: 0.1,
        seed: 123,
        max_samples: 1000
      }

      expect(described_class.rectangular_probability(**arguments).probability).to eq(
        described_class.rectangular_probability(**arguments).probability
      )
    end

    it 'supports six dimensions but rejects seven' do
      six = Array.new(6, 0.0)
      result = described_class.rectangular_probability(
        mean: six,
        covariance: Array.new(6) { |i| Array.new(6) { |j| i == j ? 1.0 : 0.0 } },
        lower_bound: six,
        upper_bound: Array.new(6, infinity),
        error_budget: 0.0001
      )
      expect(result.probability).to be_within(1e-12).of(1.0 / 64.0)

      expect do
        described_class.new(Array.new(7, 0.0), Array.new(7) { Array.new(7, 0.0) })
      end.to raise_error(described_class::DimensionError)
    end

    it 'rejects a non-positive-semidefinite covariance without regularization' do
      expect do
        described_class.new([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])
      end.to raise_error(described_class::NonPositiveSemidefiniteError)
    end

    it 'rejects asymmetric covariance and invalid bounds' do
      expect do
        described_class.new([0.0, 0.0], [[1.0, 0.2], [0.3, 1.0]])
      end.to raise_error(described_class::ArgumentError)

      expect do
        described_class.rectangular_probability(
          mean: [0.0],
          covariance: [[1.0]],
          lower_bound: [1.0],
          upper_bound: [0.0],
          error_budget: 0.01
        )
      end.to raise_error(described_class::ArgumentError)
    end

    it 'requires an integer random seed' do
      expect do
        described_class.rectangular_probability(
          mean: [0.0, 0.0, 0.0],
          covariance: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
          lower_bound: [0.0, 0.0, 0.0],
          upper_bound: [1.0, 1.0, 1.0],
          error_budget: 0.01,
          seed: nil
        )
      end.to raise_error(described_class::ArgumentError)
    end

    it 'marks tiny positive variance as near singular but still valid' do
      distribution = described_class.new(
        [0.0, 0.0],
        [[1e-11, 0.0], [0.0, 1.0]]
      )

      expect(distribution.near_singular).to eq(true)
      expect(distribution.rank).to eq(2)
    end

    it 'keeps tiny positive variance full rank while marking it near singular' do
      result = described_class.rectangular_probability(
        mean: [0.0, 0.0],
        covariance: [[1e-13, 0.0], [0.0, 1.0]],
        lower_bound: [1.0, -infinity],
        upper_bound: [infinity, infinity],
        error_budget: 0.0001
      )

      expect(result.rank).to eq(2)
      expect(result.near_singular).to eq(true)
      expect(result.probability).to eq(0.0)
    end
  end
end
