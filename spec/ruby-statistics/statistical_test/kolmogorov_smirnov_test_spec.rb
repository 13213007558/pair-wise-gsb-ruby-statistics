require 'spec_helper'

describe RubyStatistics::StatisticalTest::KolmogorovSmirnovTest do
  describe 'KSTest' do
    it 'defines KSTest as an alias for the KolmogorovSmirnovTest class name' do
      expect(RubyStatistics::StatisticalTest::KSTest).to be_a(Class)
      expect(RubyStatistics::StatisticalTest::KSTest).to eq described_class
    end
  end

  def choose(total, selected)
    selected = [selected, total - selected].min
    (1..selected).reduce(1) do |result, value|
      result * (total - selected + value) / value
    end
  end

  def ks_statistic(group_one, group_two, alternative)
    size_one = group_one.size
    size_two = group_two.size
    counts = Hash.new { |hash, value| hash[value] = [0, 0] }

    group_one.each { |value| counts[value][0] += 1 }
    group_two.each { |value| counts[value][1] += 1 }

    cumulative_one = 0
    cumulative_two = 0
    positive = 0
    negative = 0

    counts.keys.sort.each do |value|
      cumulative_one += counts[value][0]
      cumulative_two += counts[value][1]
      difference = cumulative_one * size_two - cumulative_two * size_one
      positive = [positive, difference].max
      negative = [negative, -difference].max
    end

    values = {
      greater: positive.to_r / (size_one * size_two),
      less: negative.to_r / (size_one * size_two),
      two_sided: [positive, negative].max.to_r / (size_one * size_two)
    }

    values[alternative]
  end

  def exact_oracle(values, size_one, alternative)
    observed = ks_statistic(values.first(size_one), values.drop(size_one), alternative)
    tail = (0...values.size).to_a.combination(size_one).count do |indices|
      selected = indices.to_h { |index| [index, true] }
      group_one = values.each_index.filter_map { |index| values[index] if selected[index] }
      group_two = values.each_index.filter_map { |index| values[index] unless selected[index] }

      ks_statistic(group_one, group_two, alternative) >= observed
    end

    tail.to_r / choose(values.size, size_one)
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

    it 'keeps the default asymptotic critical-value interface' do
      result = described_class.two_samples(
        group_one: [1, 1, 2],
        group_two: [2, 3, 3]
      )

      expect(result).not_to have_key(:p_value)
      expect(result).not_to have_key(:completed)
      expect(result[:exact]).to be_nil
    end
  end

  describe '.two_samples with exact: true' do
    it 'matches exhaustive label allocations for six through ten observations' do
      alternatives = [:two_sided, :greater, :less]

      6.upto(10) do |sample_size|
        patterns = [
          Array.new(sample_size, 0),
          [0] * (sample_size / 2) + [1] * (sample_size - sample_size / 2),
          (1..sample_size).map { |value| value / 2 },
          (1..sample_size).map { |value| value % 3 },
          (1..sample_size).to_a
        ]
        random = Random.new(900 + sample_size)
        12.times { patterns << Array.new(sample_size) { random.rand(1..(sample_size / 2 + 1)) } }

        patterns.uniq.each do |values|
          1.upto(sample_size - 1) do |size_one|
            alternatives.each do |alternative|
              result = described_class.two_samples(
                group_one: values.first(size_one),
                group_two: values.drop(size_one),
                exact: true,
                alternative: alternative
              )

              expect(result[:completed]).to be(true)
              expect(result[:exact]).to be(true)
              expect(result[:p_value]).to eq exact_oracle(values, size_one, alternative)
            end
          end
        end
      end
    end

    it 'uses an inclusive tail for a two-sided statistic with tied blocks' do
      group_one = [1, 1, 2, 2]
      group_two = [2, 2, 3, 3]

      result = described_class.two_samples(
        group_one: group_one,
        group_two: group_two,
        exact: true
      )

      expect(result[:d_statistic]).to eq 1/2r
      expect(result[:p_value]).to eq 46/70r
    end

    it 'matches an exhaustive oracle for a larger tied sample' do
      values = [0] * 6 + [1] * 6 + [2] * 6
      size_one = 9

      [:two_sided, :greater, :less].each do |alternative|
        result = described_class.two_samples(
          group_one: values.first(size_one),
          group_two: values.drop(size_one),
          exact: true,
          alternative: alternative
        )

        expect(result[:p_value]).to eq exact_oracle(values, size_one, alternative)
      end
    end

    it 'matches an exhaustive oracle for a larger continuous sample' do
      values = (1..20).to_a
      size_one = 10
      result = described_class.two_samples(
        group_one: values.first(size_one),
        group_two: values.drop(size_one),
        exact: true
      )

      expect(result[:p_value]).to eq exact_oracle(values, size_one, :two_sided)
    end

    it 'returns one for all equal observations' do
      result = described_class.two_samples(
        group_one: [1, 1, 1],
        group_two: [1, 1, 1, 1],
        exact: true
      )

      expect(result[:d_statistic]).to eq 0r
      expect(result[:p_value]).to eq 1r
      expect(result[:completed]).to be(true)
    end

    it 'handles complete separation with very unequal sample sizes' do
      group_one = [1]
      group_two = [2, 2, 3, 3, 4]

      result = described_class.two_samples(
        group_one: group_one,
        group_two: group_two,
        exact: true
      )

      expect(result[:d_statistic]).to eq 1r
      expect(result[:p_value]).to eq 2/6r
    end

    it 'gives the same two-sided exact p-value after exchanging samples' do
      group_one = [0, 0, 1, 1, 1]
      group_two = [1, 1, 2, 2, 2, 2]

      result = described_class.two_samples(
        group_one: group_one,
        group_two: group_two,
        exact: true
      )
      exchanged = described_class.two_samples(
        group_one: group_two,
        group_two: group_one,
        exact: true
      )

      expect(exchanged[:d_statistic]).to eq result[:d_statistic]
      expect(exchanged[:p_value]).to eq result[:p_value]
    end

    it 'exchanges greater and less when samples are swapped' do
      group_one = [1, 1, 1, 2]
      group_two = [2, 2, 3, 3, 3]

      greater = described_class.two_samples(
        group_one: group_one,
        group_two: group_two,
        exact: true,
        alternative: :greater
      )
      swapped_less = described_class.two_samples(
        group_one: group_two,
        group_two: group_one,
        exact: true,
        alternative: :less
      )

      expect(swapped_less[:p_value]).to eq greater[:p_value]
      expect(swapped_less[:d_statistic]).to eq greater[:d_statistic]
    end

    it 'reports incompletion and no asymptotic p-value when the state budget is exhausted' do
      result = described_class.two_samples(
        group_one: Array.new(100, 0),
        group_two: (1..100).to_a,
        exact: true,
        state_budget: 2
      )

      expect(result[:completed]).to be(false)
      expect(result[:exact]).to be(true)
      expect(result[:p_value]).to be_nil
      expect(result[:null]).to be_nil
      expect(result[:alternative]).to be_nil
      expect(result[:required_states]).to be > result[:state_budget]
      expect(result[:reason]).to include('state')
    end

    it 'rejects an unsupported alternative' do
      expect do
        described_class.two_samples(
          group_one: [1],
          group_two: [2],
          exact: true,
          alternative: :left
        )
      end.to raise_error(ArgumentError, /unsupported alternative/)
    end
  end
end
