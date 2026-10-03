require 'spec_helper'

describe RubyStatistics::StatisticalTest::KolmogorovSmirnovTest do
  describe 'KSTest' do
    it 'defines KSTest as an alias for the KolmogorovSmirnovTest class name' do
      expect(RubyStatistics::StatisticalTest::KSTest).to be_a(Class)
      expect(RubyStatistics::StatisticalTest::KSTest).to eq described_class
    end
  end

  describe '.two_samples' do
    it 'computes a two-sided Kolmogorov-Smirnov test for two samples, following example ONE' do
      # Data extracted from http://www.stats.ox.ac.uk/~massa/Lecture%2013.pdf
      # calculation in R:
      # > ks.test(c(1.2,1.4,1.9,3.7,4.4,4.8,9.7,17.3,21.1,28.4), c(5.6,6.5,6.6,6.9,9.2,10.4,10.6,19.3))
      #
      #   Two-sample Kolmogorov-Smirnov test
      #
      #   D = 0.6, p-value = 0.04987
      #   alternative hypothesis: two-sided

      group_one = [1.2, 1.4, 1.9, 3.7, 4.4, 4.8, 9.7, 17.3, 21.1, 28.4]
      group_two = [5.6, 6.5, 6.6, 6.9, 9.2, 10.4, 10.6, 19.3]

      # alpha, by default, is 0.05
      result = described_class.two_samples(group_one: group_one, group_two: group_two)

      expect(result[:null]).to be false
      expect(result[:alternative]).to be true
      expect(result[:d_max]).to eq 0.6
    end

    it 'computes a two-sided Kolmogorov-Smirnov test for two samples, following example TWO' do
      # Calculation in R
      # Two-sample Kolmogorov-Smirnov test
      #
      # D = 1, p-value = 6.657e-08
      # alternative hypothesis: two-sided

      men = RubyStatistics::Distribution::Normal.new(3.0, 1.0).random(elements: 10, seed: 100)
      women = RubyStatistics::Distribution::Weibull.new(2.0, 3.0).random(elements: 20, seed: 100)

      # alpha, by default, is 0.05
      result = described_class.two_samples(group_one: men, group_two: women)

      expect(result[:null]).to be false
      expect(result[:alternative]).to be true
      expect(result[:d_max]).to eq 1.0
    end

    it 'computes a two-sided Kolmogorov-Smirnov test for two samples, following example THREE' do
      # Calculation in R
      # Two-sample Kolmogorov-Smirnov test
      #
      # D = 0.4, p-value = 0.873
      # alternative hypothesis: two-sided

      men = RubyStatistics::Distribution::StandardNormal.new.random(elements: 500, seed: 10)
      women = RubyStatistics::Distribution::StandardNormal.new.random(elements: 50, seed: 40)

      # alpha, by default, is 0.05
      result = described_class.two_samples(group_one: men, group_two: women)

      expect(result[:null]).to be true
      expect(result[:alternative]).to be false
      expect(result[:d_max]).to eq 0.12
    end

    it 'keeps the asymptotic critical-value result when exact mode is disabled' do
      result = described_class.two_samples(group_one: [1, 2], group_two: [3, 4])

      expect(result).not_to have_key(:p_value)
      expect(result).not_to have_key(:completed)
      expect(result).to have_key(:d_critical)
    end

    it 'delegates to the exact permutation test when requested' do
      result = described_class.two_samples(group_one: [1, 1], group_two: [2, 2], exact: true)

      expect(result[:exact]).to be true
      expect(result[:completed]).to be true
      expect(result[:p_value_exact]).to eq Rational(2, 6)
    end
  end

  describe '.exact_two_samples' do
    it 'matches complete label enumeration for six to ten observations with ties' do
      alternatives = %i[two_sided greater less]

      (6..10).each do |sample_size|
        (1...sample_size).each do |group_one_size|
          values = [1, 1, 2, 2, 2, 3, 3, 4, 4, 4].first(sample_size)
          group_one = values.first(group_one_size)
          group_two = values.last(sample_size - group_one_size)

          alternatives.each do |alternative|
            enumeration = described_class.exact_two_samples(
              group_one: group_one,
              group_two: group_two,
              alternative: alternative,
              algorithm: :enumeration
            )
            dynamic_programming = described_class.exact_two_samples(
              group_one: group_one,
              group_two: group_two,
              alternative: alternative,
              algorithm: :dynamic_programming
            )

            expect(dynamic_programming[:completed]).to be true
            expect(dynamic_programming[:d_max]).to eq enumeration[:d_max]
            expect(dynamic_programming[:favorable_assignments]).to eq enumeration[:favorable_assignments]
            expect(dynamic_programming[:total_assignments]).to eq enumeration[:total_assignments]
            expect(dynamic_programming[:p_value_exact]).to eq enumeration[:p_value_exact]
          end
        end
      end
    end

    it 'matches the known exact probability for a larger continuous sample' do
      group_one = [1.2, 1.4, 1.9, 3.7, 4.4, 4.8, 9.7, 17.3, 21.1, 28.4]
      group_two = [5.6, 6.5, 6.6, 6.9, 9.2, 10.4, 10.6, 19.3]

      result = described_class.exact_two_samples(group_one: group_one, group_two: group_two)

      expect(result[:completed]).to be true
      expect(result[:d_max]).to eq Rational(3, 5)
      expect(result[:favorable_assignments]).to eq 2182
      expect(result[:total_assignments]).to eq 43_758
      expect(result[:p_value_exact]).to eq Rational(1091, 21_879)
    end

    it 'counts tied values as whole blocks without counting internal orders' do
      result = described_class.exact_two_samples(
        group_one: [1, 1, 2, 2],
        group_two: [2, 2, 3, 3]
      )

      expect(result[:d_max]).to eq Rational(1, 2)
      expect(result[:favorable_assignments]).to eq 46
      expect(result[:total_assignments]).to eq 70
      expect(result[:p_value_exact]).to eq Rational(46, 70)
    end

    it 'uses the inclusive tail convention' do
      result = described_class.exact_two_samples(
        group_one: [1, 1, 1],
        group_two: [1, 2, 2]
      )

      expect(result[:d_max]).to eq Rational(2, 3)
      expect(result[:favorable_assignments]).to eq 8
      expect(result[:total_assignments]).to eq 20
      expect(result[:p_value_exact]).to eq Rational(2, 5)
    end

    it 'returns one for samples with identical observed values' do
      result = described_class.exact_two_samples(
        group_one: [7, 7, 7, 7],
        group_two: [7, 7]
      )

      expect(result[:d_max]).to eq 0
      expect(result[:p_value_exact]).to eq 1
      expect(result[:favorable_assignments]).to eq result[:total_assignments]
    end

    it 'handles extremely unbalanced sample sizes' do
      group_one = Array.new(199, 1)
      group_two = [2]

      result = described_class.exact_two_samples(
        group_one: group_one,
        group_two: group_two,
        alternative: :greater
      )

      expect(result[:d_max]).to eq 1
      expect(result[:favorable_assignments]).to eq 1
      expect(result[:total_assignments]).to eq 200
      expect(result[:p_value_exact]).to eq Rational(1, 200)
    end

    it 'keeps the one-sided result when exchanging samples and reversing the direction' do
      group_one = Array.new(20, 1)
      group_two = Array.new(20, 2)

      greater = described_class.exact_two_samples(
        group_one: group_one,
        group_two: group_two,
        alternative: :greater
      )
      swapped_less = described_class.exact_two_samples(
        group_one: group_two,
        group_two: group_one,
        alternative: :less
      )
      two_sided = described_class.exact_two_samples(group_one: group_two, group_two: group_one)

      expect(swapped_less[:p_value_exact]).to eq greater[:p_value_exact]
      expect(two_sided[:d_max]).to eq greater[:d_max]
      expect(two_sided[:p_value_exact]).to eq Rational(2, greater[:total_assignments])
    end

    it 'handles larger tied samples' do
      result = described_class.exact_two_samples(
        group_one: Array.new(50, 1),
        group_two: Array.new(50, 2)
      )

      expect(result[:completed]).to be true
      expect(result[:d_max]).to eq 1
      expect(result[:favorable_assignments]).to eq 2
      expect(result[:total_assignments]).to eq described_class.send(:binomial_coefficient, 100, 50)
    end

    it 'returns an incomplete exact result instead of using the asymptotic approximation' do
      result = described_class.exact_two_samples(
        group_one: Array.new(10) { |index| index },
        group_two: Array.new(10) { |index| index + 10 },
        state_budget: 5
      )

      expect(result[:exact]).to be true
      expect(result[:completed]).to be false
      expect(result[:state_budget]).to eq 5
      expect(result[:states_required]).to eq 6
      expect(result[:reason]).to include('state_budget')
      expect(result).not_to have_key(:p_value)
      expect(result).not_to have_key(:p_value_exact)
      expect(result).not_to have_key(:d_critical)
    end

    it 'rejects empty samples and invalid arguments' do
      expect do
        described_class.exact_two_samples(group_one: [], group_two: [1])
      end.to raise_error(ArgumentError, /at least one observation/)

      expect do
        described_class.exact_two_samples(group_one: [1], group_two: [2], state_budget: 0)
      end.to raise_error(ArgumentError, /state_budget/)

      expect do
        described_class.exact_two_samples(group_one: [1], group_two: [2], alternative: :both)
      end.to raise_error(ArgumentError, /alternative/)
    end

    it 'accepts R-style alternative names' do
      result = described_class.exact_two_samples(
        group_one: [1, 1],
        group_two: [2, 2],
        alternative: 'two.sided'
      )

      expect(result[:alternative_hypothesis]).to eq :two_sided
      expect(result[:p_value_exact]).to eq Rational(1, 3)
    end
  end
end
