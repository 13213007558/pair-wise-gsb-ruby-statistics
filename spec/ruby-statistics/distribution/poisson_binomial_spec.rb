require 'spec_helper'

describe RubyStatistics::Distribution::PoissonBinomial do
  def exhaustive_masses(probabilities)
    masses = Array.new(probabilities.size + 1, 0.0)

    (0...(2 ** probabilities.size)).each do |mask|
      mass = 1.0
      successes = 0

      probabilities.each_with_index do |probability, index|
        if mask[index].zero?
          mass *= 1.0 - probability
        else
          mass *= probability
          successes += 1
        end
      end

      masses[successes] += mass
    end

    masses
  end

  [
    [],
    [0.0],
    [1.0],
    [0.0, 1.0],
    [0.2, 0.0, 0.7, 1.0],
    [0.1, 0.3, 0.5, 0.9]
  ].each do |probabilities|
    context "with probabilities #{probabilities.inspect}" do
      let(:distribution) { described_class.new(probabilities) }
      let(:masses) { exhaustive_masses(probabilities) }

      it 'matches an exhaustive enumeration for PMF, CDF, and survival' do
        (0..probabilities.size).each do |k|
          cumulative_mass = masses[0..k].sum
          survival_mass = masses[(k + 1)..].sum

          expect(distribution.probability_mass_function(k)).to be_within(1e-12).of(masses[k])
          expect(distribution.cumulative_function(k)).to be_within(1e-12).of(cumulative_mass)
          expect(distribution.survival_function(k)).to be_within(1e-12).of(survival_mass)
          expect(Math.exp(distribution.log_probability_mass_function(k))).to be_within(1e-12).of(masses[k]) if masses[k].positive?
        end

        expect(distribution.probability_mass_function(-1)).to be_nil
        expect(distribution.probability_mass_function(probabilities.size + 1)).to be_nil
        expect(distribution.cumulative_function(-1)).to eq(0.0)
        expect(distribution.survival_function(probabilities.size)).to eq(0.0)
      end
    end
  end

  describe 'initialization' do
    it 'rejects probabilities outside the unit interval and invalid numeric values' do
      [-0.1, 1.1, nil, Float::NAN, Float::INFINITY].each do |invalid_probability|
        expect { described_class.new([invalid_probability]) }.to raise_error(ArgumentError)
      end
    end

    it 'keeps certain successes and failures as support bounds' do
      distribution = described_class.new([0.0, 0.4, 1.0, 1.0])

      expect(distribution.certain_failures).to eq(1)
      expect(distribution.certain_successes).to eq(2)
      expect(distribution.probability_mass_function(1)).to eq(0.0)
      expect(distribution.probability_mass_function(2)).to be_within(1e-12).of(0.6)
      expect(distribution.probability_mass_function(3)).to be_within(1e-12).of(0.4)
      expect(distribution.probability_mass_function(4)).to eq(0.0)
      expect(distribution.probability_mass_function(5)).to be_nil
      expect(distribution.log_probability_mass_function(1)).to eq(-Float::INFINITY)
    end
  end

  describe 'empty distribution' do
    subject(:distribution) { described_class.new([]) }

    it 'is a point mass at zero' do
      expect(distribution.probability_mass_function(0)).to eq(1.0)
      expect(distribution.cumulative_function(0)).to eq(1.0)
      expect(distribution.survival_function(0)).to eq(0.0)
      expect(distribution.inverse_cumulative_function(0.5)).to eq(0)
      expect(distribution.mean).to eq(0.0)
      expect(distribution.variance).to eq(0.0)
    end
  end

  describe '#inverse_cumulative_function' do
    subject(:distribution) { described_class.new([0.2, 0.4, 0.8]) }

    it 'returns the first integer whose cumulative probability reaches the argument' do
      expect(distribution.inverse_cumulative_function(0.0)).to eq(0)
      expect(distribution.inverse_cumulative_function(0.096)).to eq(0)
      expect(distribution.inverse_cumulative_function(0.097)).to eq(1)
      expect(distribution.inverse_cumulative_function(0.568)).to eq(1)
      expect(distribution.inverse_cumulative_function(0.569)).to eq(2)
      expect(distribution.inverse_cumulative_function(0.936)).to eq(2)
      expect(distribution.inverse_cumulative_function(0.937)).to eq(3)
      expect(distribution.inverse_cumulative_function(1.0)).to eq(3)
    end

    it 'is undefined outside the unit interval' do
      expect(distribution.inverse_cumulative_function(-0.1)).to be_nil
      expect(distribution.inverse_cumulative_function(1.1)).to be_nil
    end
  end

  describe 'numerical tails' do
    subject(:distribution) { described_class.new(Array.new(500, 1e-300)) }

    it 'computes a nonzero right tail directly even when 1 - CDF underflows' do
      expect(1.0 - distribution.cumulative_function(0)).to eq(0.0)
      expect(distribution.survival_function(0)).to be > 0.0
      expect(distribution.survival_function(0)).to be_within(1e-12).of(500 * 1e-300)
      expect(distribution.log_survival_function(0)).to be_within(1e-12).of(Math.log(500 * 1e-300))
    end
  end

  describe 'moments and normalization' do
    let(:probabilities) do
      [0.0, 1.0] + Array.new(198) { |index| ((index % 97) + 1) / 100.0 }
    end
    subject(:distribution) { described_class.new(probabilities) }

    it 'normalizes to one and matches the analytical mean and variance' do
      masses = (distribution.certain_successes..(200 - distribution.certain_failures)).map do |k|
        distribution.probability_mass_function(k)
      end
      support = (distribution.certain_successes..(200 - distribution.certain_failures)).to_a
      observed_mean = support.zip(masses).sum { |k, mass| k * mass }
      observed_variance = support.zip(masses).sum { |k, mass| ((k - observed_mean) ** 2) * mass }

      expect(masses.sum).to be_within(1e-10).of(1.0)
      expect(observed_mean).to be_within(1e-10).of(distribution.mean)
      expect(observed_variance).to be_within(1e-10).of(distribution.variance)
      expect(distribution.mean).to be_within(1e-12).of(probabilities.sum)
      expect(distribution.variance).to be_within(1e-12).of(probabilities.sum { |p| p * (1.0 - p) })
    end
  end

  describe 'equal probabilities' do
    [0.0, 0.5, 0.37, 1.0].each do |p|
      it "matches Binomial when every probability is #{p}" do
        number_of_trials = 30
        poisson_binomial = described_class.new(Array.new(number_of_trials) { p })
        binomial = RubyStatistics::Distribution::Binomial.new(number_of_trials, p)

        (0..number_of_trials).each do |k|
          expect(poisson_binomial.probability_mass_function(k))
            .to be_within(1e-12).of(binomial.probability_mass_function(k))
          expect(poisson_binomial.cumulative_function(k))
            .to be_within(1e-9).of(binomial.cumulative_function(k))
        end

        expect(poisson_binomial.mean).to be_within(1e-12).of(binomial.mean)
        expect(poisson_binomial.variance).to be_within(1e-12).of(binomial.variance)
      end
    end
  end
end
