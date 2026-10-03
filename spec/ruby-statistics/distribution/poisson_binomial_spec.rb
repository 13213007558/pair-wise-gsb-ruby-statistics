require 'spec_helper'

describe RubyStatistics::Distribution::PoissonBinomial do
  let(:probabilities) { [0.2, 0.5, 0.7] }
  let(:distribution) { described_class.new(probabilities) }

  def enumerated_pmf(probabilities)
    Array.new(probabilities.size + 1, 0.0).tap do |masses|
      [0, 1].repeated_permutation(probabilities.size).each do |outcomes|
        probability = outcomes.each_with_index.inject(1.0) do |result, (outcome, index)|
          result * (outcome == 1 ? probabilities[index] : 1.0 - probabilities[index])
        end

        masses[outcomes.sum] += probability
      end
    end
  end

  describe '#initialize' do
    it 'accepts an array of probabilities' do
      expect(described_class.new(probabilities).probabilities).to eq(probabilities.map(&:to_f))
    end

    it 'accepts probabilities as separate arguments' do
      expect(described_class.new(*probabilities).probabilities).to eq(probabilities.map(&:to_f))
    end

    it 'rejects probabilities outside the unit interval' do
      expect { described_class.new([0.5, 1.1]) }.to raise_error(ArgumentError)
      expect { described_class.new([-0.1, 0.5]) }.to raise_error(ArgumentError)
    end

    it 'rejects non-finite probabilities' do
      expect { described_class.new([Float::NAN]) }.to raise_error(ArgumentError)
      expect { described_class.new([Float::INFINITY]) }.to raise_error(ArgumentError)
    end
  end

  describe 'empty probability set' do
    let(:empty_distribution) { described_class.new([]) }

    it 'places all mass at zero' do
      expect(empty_distribution.probability_mass_function(0)).to eq(1.0)
      expect(empty_distribution.log_probability_mass_function(0)).to eq(0.0)
      expect(empty_distribution.cumulative_function(0)).to eq(1.0)
      expect(empty_distribution.survival_function(0)).to eq(0.0)
      expect(empty_distribution.mean).to eq(0.0)
      expect(empty_distribution.variance).to eq(0.0)
    end

    it 'maps every quantile to zero' do
      expect(empty_distribution.quantile(0)).to eq(0)
      expect(empty_distribution.quantile(0.5)).to eq(0)
      expect(empty_distribution.quantile(1)).to eq(0)
    end
  end

  describe 'deterministic probabilities' do
    let(:deterministic_distribution) { described_class.new(0, 1, 0, 1) }

    it 'keeps zero and one as a deterministic shift' do
      expect(deterministic_distribution.mean).to eq(2.0)
      expect(deterministic_distribution.variance).to eq(0.0)
      expect(deterministic_distribution.probability_mass_function(2)).to eq(1.0)
      [0, 1, 3, 4].each { |k| expect(deterministic_distribution.probability_mass_function(k)).to eq(0.0) }
    end

    it 'calculates cumulative and survival probabilities directly' do
      expect(deterministic_distribution.cumulative_function(1)).to eq(0.0)
      expect(deterministic_distribution.cumulative_function(2)).to eq(1.0)
      expect(deterministic_distribution.survival_function(1)).to eq(1.0)
      expect(deterministic_distribution.survival_function(2)).to eq(0.0)
    end

    it 'finds the first integer whose cumulative probability reaches the quantile' do
      expect(deterministic_distribution.quantile(0)).to eq(0)
      expect(deterministic_distribution.quantile(0.000001)).to eq(2)
      expect(deterministic_distribution.quantile(Float::MIN)).to eq(2)
      expect(deterministic_distribution.quantile(1)).to eq(4)
    end
  end

  describe '#probability_mass_function' do
    it 'is not defined outside the support' do
      expect(distribution.probability_mass_function(-1)).to be_nil
      expect(distribution.probability_mass_function(4)).to be_nil
    end

    it 'matches complete enumeration for heterogeneous probabilities' do
      enumerated_pmf(probabilities).each_with_index do |probability, k|
        expect(distribution.probability_mass_function(k)).to be_within(1e-15).of(probability)
      end
    end

    it 'matches a Binomial distribution when all probabilities are equal' do
      n = 9
      p = 0.37
      binomial = RubyStatistics::Distribution::Binomial.new(n, p)
      poisson_binomial = described_class.new(Array.new(n, p))

      (0..n).each do |k|
        expect(poisson_binomial.probability_mass_function(k))
          .to be_within(1e-14).of(binomial.probability_mass_function(k))
      end
    end
  end

  describe '#cumulative_function and #survival_function' do
    it 'is defined with the limiting values outside the support' do
      expect(distribution.cumulative_function(-1)).to eq(0.0)
      expect(distribution.cumulative_function(3)).to eq(1.0)
      expect(distribution.survival_function(-1)).to eq(1.0)
      expect(distribution.survival_function(3)).to eq(0.0)
    end

    it 'matches enumerated prefix and suffix sums' do
      masses = enumerated_pmf(probabilities)

      (0..probabilities.size).each do |k|
        expected_cdf = masses[0..k].sum
        expected_survival = masses[(k + 1)..].sum

        expect(distribution.cumulative_function(k)).to be_within(1e-15).of(expected_cdf)
        expect(distribution.survival_function(k)).to be_within(1e-15).of(expected_survival)
        expect(distribution.log_cumulative_function(k))
          .to be_within(1e-14).of(Math.log(expected_cdf))
      end
    end

    it 'matches a Binomial CDF when all probabilities are equal' do
      n = 9
      p = 0.37
      binomial = RubyStatistics::Distribution::Binomial.new(n, p)
      poisson_binomial = described_class.new(Array.new(n, p))

      (0..n).each do |k|
        expect(poisson_binomial.cumulative_function(k))
          .to be_within(1e-12).of(binomial.cumulative_function(k))
      end
    end

    it 'keeps a directly accumulated tiny right tail through the log probability' do
      tiny_probabilities = [1e-110, 1e-111, 1e-112]
      tiny_distribution = described_class.new(tiny_probabilities)
      expected_log_probability = tiny_probabilities.sum { |value| Math.log(value) }

      expect(tiny_distribution.cumulative_function(2)).to eq(1.0)
      expect(1.0 - tiny_distribution.cumulative_function(2)).to eq(0.0)
      expect(tiny_distribution.survival_function(2)).to eq(0.0)
      expect(tiny_distribution.log_survival_function(2))
        .to be_within(1e-12).of(expected_log_probability)
    end

    it 'returns log probabilities below the positive Float underflow bound' do
      tiny_probability = Math.exp(-300)
      tiny_distribution = described_class.new(Array.new(3, tiny_probability))

      expect(tiny_distribution.probability_mass_function(3)).to eq(0.0)
      expect(tiny_distribution.log_probability_mass_function(3)).to be_within(1e-12).of(-900)
    end
  end

  describe '#quantile' do
    it 'is the smallest integer whose CDF reaches the requested probability' do
      expect(distribution.quantile(0.0)).to eq(0)
      expect(distribution.quantile(0.12)).to eq(0)
      expect(distribution.quantile(0.13)).to eq(1)
      expect(distribution.quantile(0.55)).to eq(1)
      expect(distribution.quantile(0.56)).to eq(2)
      expect(distribution.quantile(0.93)).to eq(2)
      expect(distribution.quantile(0.94)).to eq(3)
      expect(distribution.quantile(1.0)).to eq(3)
    end

    it 'rejects probabilities outside [0, 1]' do
      expect { distribution.quantile(-0.1) }.to raise_error(ArgumentError)
      expect { distribution.quantile(1.1) }.to raise_error(ArgumentError)
    end

    it 'matches the Binomial inverse CDF when all probabilities are equal' do
      n = 9
      p = 0.37
      binomial = RubyStatistics::Distribution::Binomial.new(n, p)
      poisson_binomial = described_class.new(Array.new(n, p))
      binomial_quantile = (0..n).find { |k| binomial.cumulative_function(k) >= p }

      expect(poisson_binomial.quantile(p)).to eq(binomial_quantile)
    end
  end

  describe 'larger distributions' do
    let(:larger_probabilities) do
      [0.0, 1.0] + Array.new(180) { |index| 0.05 + (index % 20) * 0.045 }
    end
    let(:larger_distribution) { described_class.new(larger_probabilities) }

    it 'preserves total mass, mean and variance' do
      masses = (0..larger_distribution.number_of_trials).map do |k|
        larger_distribution.probability_mass_function(k)
      end

      expect(masses.sum).to be_within(1e-12).of(1.0)
      expect(masses.each_with_index.sum { |probability, k| probability * k })
        .to be_within(1e-10).of(larger_distribution.mean)
      variance = masses.each_with_index.sum do |probability, k|
        probability * (k - larger_distribution.mean) ** 2
      end
      expect(variance).to be_within(1e-10).of(larger_distribution.variance)
    end

    it 'has complementary directly calculated CDF and survival values' do
      (0..larger_distribution.number_of_trials).each do |k|
        expect(larger_distribution.cumulative_function(k) + larger_distribution.survival_function(k))
          .to be_within(1e-12).of(1.0)
      end
    end
  end
end
