require 'spec_helper'

describe RubyStatistics::StatisticalTest::WilcoxonSignedRankTest do
  let(:test_class) { described_class.new }

  # Independent reference implementation: ranks |d| with midranks, applies the
  # zero-difference rule and enumerates all 2^n sign assignments one by one.
  # The implementation under test uses dynamic programming instead, so this
  # cross-checks the DP against explicit enumeration.
  def reference_midranks(values)
    values.map do |value|
      less = values.count { |other| other < value }
      equal = values.count { |other| other == value }
      less + (equal + 1) / 2.0
    end
  end

  def reference_items(differences, zero_method)
    if zero_method == :pratt
      ranks = reference_midranks(differences.map(&:abs))
      differences.zip(ranks).reject { |difference, _| difference.zero? }
    else
      kept = differences.reject(&:zero?)
      ranks = reference_midranks(kept.map(&:abs))
      kept.zip(ranks)
    end
  end

  def brute_force(differences, zero_method)
    items = reference_items(differences, zero_method)
    ranks = items.map { |_, rank| rank }
    observed = items.select { |difference, _| difference.positive? }
                    .reduce(0.0) { |memo, (_, rank)| memo + rank }

    totals = Hash.new(0)
    assignments = 2**ranks.size
    (0...assignments).each do |mask|
      sum = 0.0
      ranks.each_with_index { |rank, index| sum += rank if mask[index] == 1 }
      totals[sum] += 1
    end

    less = totals.select { |sum, _| sum <= observed }.values.sum / assignments.to_f
    greater = totals.select { |sum, _| sum >= observed }.values.sum / assignments.to_f

    { statistic: observed,
      less: less,
      greater: greater,
      two: [1.0, 2 * [less, greater].min].min }
  end

  describe 'input validation' do
    it 'rejects paired samples of different sizes' do
      expect { test_class.perform(0.05, :two_tail, [1, 2], [1]) }
        .to raise_error(ArgumentError, /same size/)
    end

    it 'rejects empty samples' do
      expect { test_class.perform(0.05, :two_tail, []) }
        .to raise_error(ArgumentError, /at least one/)
    end

    it 'rejects unknown tails, zero methods and methods' do
      expect { test_class.perform(0.05, :one_tail, [1, 2]) }
        .to raise_error(ArgumentError, /tails/)
      expect { test_class.perform(0.05, :two_tail, [1, 2], nil, zero_method: :drop) }
        .to raise_error(ArgumentError, /zero_method/)
      expect { test_class.perform(0.05, :two_tail, [1, 2], nil, method: :bootstrap) }
        .to raise_error(ArgumentError, /method/)
    end
  end

  # Example extracted from R's wilcox.test documentation: paired samples with
  # no ties and no zero differences, V = 40, exact two-sided p = 0.03906.
  describe '#perform' do
    let(:before) { [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30] }
    let(:after) { [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29] }

    it 'computes the paired differences of the same subjects' do
      result = test_class.perform(0.05, :two_tail, before, after)

      expect(result[:n]).to eq 9
      expect(result[:statistic]).to eq 40.0
      expect(result[:exact]).to be true
      expect(result[:p_value]).to be_within(1e-12).of(0.0390625)
      to_check = 10.0 / 512
      expect(result[:p_value_greater]).to be_within(1e-12).of(to_check)
      expect(result[:p_value_less]).to be_within(1e-12).of(505.0 / 512)
      expect(result[:null]).to be false
      expect(result[:alternative]).to be true
    end

    it 'gives exact one-sided p-values' do
      all_positive = test_class.perform(0.05, :two_tail, before)

      expect(all_positive[:statistic]).to eq 45.0
      expect(all_positive[:p_value]).to be_within(1e-12).of(2.0 / 512)

      right = test_class.perform(0.05, :right_tail, before)
      left = test_class.perform(0.05, :left_tail, before)

      expect(right[:p_value]).to be_within(1e-12).of(1.0 / 512)
      expect(left[:p_value]).to eq 1.0
    end

    it 'shifts the differences by mu before testing' do
      shifted = before.map { |value| value - 1.0 }

      direct = test_class.perform(0.05, :two_tail, shifted)
      with_mu = test_class.perform(0.05, :two_tail, before, nil, mu: 1.0)

      expect(with_mu[:statistic]).to eq direct[:statistic]
      expect(with_mu[:p_value]).to eq direct[:p_value]
    end
  end

  describe 'ties in absolute differences' do
    # |d| = [1, 1, 1, 2] gets midranks [2, 2, 2, 4]; W+ = 8. The tie-aware
    # exact distribution gives p = 0.5, while the no-ties sign-rank table
    # would give 2 * 3/16 = 0.375 for the same statistic.
    it 'builds the exact distribution from the actual tied ranks' do
      result = test_class.perform(0.05, :two_tail, [1, 1, -1, 2])

      expect(result[:statistic]).to eq 8.0
      expect(result[:p_value]).to eq 0.5
      expect(result[:p_value]).not_to eq 0.375
    end

    it 'handles half-integer midranks by scaling them to integers' do
      # |d| = [1, 2, 2, 3] gets midranks [1, 2.5, 2.5, 4].
      differences = [1, 2, 2, -3]
      result = test_class.perform(0.05, :two_tail, differences)
      expected = brute_force(differences, :wilcoxon)

      expect(result[:statistic]).to eq 6.0
      expect(result[:p_value]).to eq expected[:two]
      expect(result[:p_value_less]).to eq expected[:less]
      expect(result[:p_value_greater]).to eq expected[:greater]
    end
  end

  describe 'zero-difference handling' do
    it 'discards zero differences with zero_method: :wilcoxon' do
      result = test_class.perform(0.05, :two_tail, [0, 5, -1], nil, zero_method: :wilcoxon)

      expect(result[:n]).to eq 2
      expect(result[:statistic]).to eq 2.0
    end

    it 'keeps zero differences in the ranking with zero_method: :pratt' do
      result = test_class.perform(0.05, :two_tail, [0, 5, -1], nil, zero_method: :pratt)

      # The zero difference takes rank 1, so the remaining ranks shift up.
      expect(result[:n]).to eq 2
      expect(result[:statistic]).to eq 3.0
    end

    it 'applies the same zero convention to the estimate' do
      wilcoxon = test_class.perform(0.05, :two_tail, [0, 2, 4], nil, zero_method: :wilcoxon)
      pratt = test_class.perform(0.05, :two_tail, [0, 2, 4], nil, zero_method: :pratt)

      # Walsh averages of [2, 4] vs [0, 2, 4].
      expect(wilcoxon[:estimate]).to eq 3.0
      expect(pratt[:estimate]).to eq 2.0
    end

    context 'when every difference is zero' do
      it 'returns p-value 1, statistic 0 and estimate 0 for both zero methods' do
        %i[wilcoxon pratt].each do |zero_method|
          result = test_class.perform(0.05, :two_tail, [0, 0, 0], nil, zero_method: zero_method)

          expect(result[:n]).to eq 0
          expect(result[:statistic]).to eq 0.0
          expect(result[:p_value]).to eq 1.0
          expect(result[:estimate]).to eq 0.0
          expect(result[:z]).to be_nil
        end
      end

      it 'accepts the whole real line when the sample is too small to reject' do
        result = test_class.perform(0.05, :two_tail, [0, 0, 0])

        expect(result[:acceptance_set]).to eq [
          { lower: -Float::INFINITY, upper: Float::INFINITY,
            include_lower: false, include_upper: false }
        ]
        expect(result[:confidence_interval]).to eq [-Float::INFINITY, Float::INFINITY]
      end

      it 'shrinks the acceptance set to the single point mu = 0' do
        result = test_class.perform(0.05, :two_tail, Array.new(10, 0))

        expect(result[:acceptance_set]).to eq [
          { lower: 0.0, upper: 0.0, include_lower: true, include_upper: true }
        ]
        expect(result[:confidence_interval]).to eq [0.0, 0.0]
      end
    end
  end

  describe 'paired positive/negative differences' do
    it 'balances W+ and W- for symmetric differences' do
      result = test_class.perform(0.05, :two_tail, [2, -2, 5, -5, 1, -1])

      expect(result[:statistic]).to eq 10.5
      expect(result[:w_minus]).to eq 10.5
      expect(result[:p_value]).to eq 1.0
    end
  end

  describe 'input permutation' do
    it 'is invariant under reordering of the pairs' do
      before = [3, 0, 5, 2, 2, 7, 1, 4]
      after = [1, 0, 2, 3, 2, 5, 4, 4]
      order = [2, 0, 7, 4, 1, 6, 3, 5]

      base = test_class.perform(0.05, :two_tail, before, after, zero_method: :pratt)
      shuffled = test_class.perform(0.05, :two_tail,
                                    before.values_at(*order), after.values_at(*order),
                                    zero_method: :pratt)

      expect(shuffled[:statistic]).to eq base[:statistic]
      expect(shuffled[:p_value]).to eq base[:p_value]
      expect(shuffled[:estimate]).to eq base[:estimate]
      expect(shuffled[:acceptance_set]).to eq base[:acceptance_set]
      expect(shuffled[:confidence_interval]).to eq base[:confidence_interval]
    end
  end

  describe 'exact p-values against full sign enumeration' do
    datasets = [
      [1.5, -2.3, 0.7, 4.1, -0.2],
      [1, 1, -1, 2],
      [0, 5, -1, 3],
      [2, -2, 3, -3],
      [0, 0, 0],
      [1, 2, 2, -3],
      [1, 2, 2, 2, 3, -1, -2, 0]
    ]

    datasets.each do |differences|
      %i[wilcoxon pratt].each do |zero_method|
        it "matches brute force for #{differences.inspect} with #{zero_method}" do
          expected = brute_force(differences, zero_method)

          %i[two_tail left_tail right_tail].each do |tails|
            result = test_class.perform(0.05, tails, differences, nil, zero_method: zero_method)

            key = tails == :two_tail ? :two : (tails == :left_tail ? :less : :greater)
            expect(result[:statistic]).to eq expected[:statistic]
            expect(result[:p_value_less]).to eq expected[:less]
            expect(result[:p_value_greater]).to eq expected[:greater]
            expect(result[:p_value]).to eq expected[key]
          end
        end
      end
    end
  end

  describe 'Hodges-Lehmann estimate' do
    it 'is the median of the Walsh averages' do
      result = test_class.perform(0.05, :two_tail, [1, 3, 5])

      # Walsh averages of [1, 3, 5]: 1, 2, 3, 3, 4, 5.
      expect(result[:estimate]).to eq 3.0
    end
  end

  describe 'acceptance set and confidence interval' do
    let(:sample) { [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30] }
    let(:walsh) do
      sample.each_with_index
            .flat_map { |value, i| (i...sample.size).map { |j| (value + sample[j]) / 2.0 } }
            .sort
    end

    it 'recovers the classical exact interval for the no-ties case' do
      result = test_class.perform(0.05, :two_tail, sample)

      # For n = 9 the two-sided 5% critical value is c = 5, so the classical
      # interval is [W_(6), W_(40)] of the 45 sorted Walsh averages.
      expect(result[:confidence_interval]).to eq [walsh[5], walsh[39]]
      expect(result[:estimate]).to eq walsh[22]
      expect(result[:acceptance_set].size).to eq 1
      expect(result[:acceptance_gaps]).to be false
      expect(result[:acceptance_exact]).to be true
    end

    it 'inverts the same test: mu inside is accepted, mu outside is rejected' do
      result = test_class.perform(0.05, :two_tail, sample)
      lower, upper = result[:confidence_interval]

      inside = test_class.perform(0.05, :two_tail, sample, nil, mu: (lower + upper) / 2.0)
      below = test_class.perform(0.05, :two_tail, sample, nil, mu: lower - 1.0)
      above = test_class.perform(0.05, :two_tail, sample, nil, mu: upper + 1.0)

      expect(inside[:null]).to be true
      expect(below[:alternative]).to be true
      expect(above[:alternative]).to be true
    end

    it 'keeps every accepted component inside the interval envelope' do
      result = test_class.perform(0.05, :two_tail, [1, 1, -1, 2, 0, 5, -3], nil, zero_method: :pratt)
      lower, upper = result[:confidence_interval]

      expect(result[:acceptance_set]).not_to be_empty
      result[:acceptance_set].each do |component|
        expect(component[:lower]).to be >= lower
        expect(component[:upper]).to be <= upper
      end
    end

    # Ties make p(mu) non-monotone: the region (1.75, 2.0) is rejected while
    # [1.0, 1.75] and the single point 2.0 are accepted. The interval envelope
    # [1.0, 2.0] therefore contains the rejected gap, which is exactly why it
    # is conservative (true coverage >= nominal confidence level).
    it 'reports rejected gaps inside the conservative interval envelope' do
      sample = [-1.0, 0.5, 2.0, 1.5, 1.0, 4.0, 1.5, 1.5, 2.0]
      result = test_class.perform(0.2, :two_tail, sample)

      expect(result[:acceptance_gaps]).to be true
      expect(result[:acceptance_set]).to eq [
        { lower: 1.0, upper: 1.75, include_lower: true, include_upper: true },
        { lower: 2.0, upper: 2.0, include_lower: true, include_upper: true }
      ]
      expect(result[:confidence_interval]).to eq [1.0, 2.0]

      inside_gap = test_class.perform(0.2, :two_tail, sample, nil, mu: 1.8)
      expect(inside_gap[:alternative]).to be true
      expect(result[:confidence_interval].first).to be < 1.8
      expect(result[:confidence_interval].last).to be > 1.8
    end
  end

  describe 'normal approximation' do
    let(:sample) { (1..60).map { |i| (i * 7 % 13) - 5.5 } }

    it 'is used on demand and reports a z statistic' do
      result = test_class.perform(0.05, :two_tail, sample, nil, method: :normal)

      expect(result[:exact]).to be false
      expect(result[:z]).not_to be_nil
      expect(result[:p_value]).to be_between(0.0, 1.0)
    end

    it 'is selected automatically for large samples' do
      result = test_class.perform(0.05, :two_tail, sample)

      expect(result[:exact]).to be false
      expect(result[:acceptance_exact]).to be false
    end

    it 'stays close to the exact p-value for moderate samples' do
      moderate = (1..20).map { |i| (i * 7 % 13) - 5.5 }

      exact = test_class.perform(0.05, :two_tail, moderate, nil, method: :exact)
      approx = test_class.perform(0.05, :two_tail, moderate, nil, method: :normal)

      expect(approx[:p_value]).to be_within(0.02).of(exact[:p_value])
    end
  end
end
