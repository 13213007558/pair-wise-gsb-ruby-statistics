require 'spec_helper'

describe RubyStatistics::StatisticalTest::WilcoxonSignedRankTest do
  let(:test_class) { described_class.new }

  def brute_force_p_values(differences, zero_method)
    values = differences.map(&:to_r)
    ranked_values = zero_method == :wilcox ? values.reject(&:zero?) : values
    assignments = 1 << ranked_values.size

    absolute_values = ranked_values.map(&:abs)
    ranks = Array.new(ranked_values.size)
    ordered = absolute_values.each_with_index.sort_by { |value, _index| value }
    position = 0

    while position < ordered.size
      value = ordered[position].first
      last_position = position
      last_position += 1 while last_position + 1 < ordered.size && ordered[last_position + 1].first == value

      average_rank = ((position + 1)..(last_position + 1)).sum / (last_position - position + 1).to_r
      (position..last_position).each { |offset| ranks[ordered[offset].last] = average_rank }
      position = last_position + 1
    end

    nonzero_positions = ranked_values.each_index.reject { |index| ranked_values[index].zero? }
    w_plus_observed = nonzero_positions.sum(0r) do |index|
      ranked_values[index].positive? ? ranks[index] : 0r
    end
    total_rank = nonzero_positions.sum(0r) { |index| ranks[index] }
    tails = Hash.new(0)

    (0...assignments).each do |mask|
      w_plus = nonzero_positions.sum(0r) do |index|
        mask[index] == 1 ? ranks[index] : 0r
      end

      tails[:greater] += 1 if w_plus >= w_plus_observed
      tails[:less] += 1 if w_plus <= w_plus_observed
      tails[:two_sided] += 1 if w_plus >= [w_plus_observed, total_rank - w_plus_observed].max ||
                                  w_plus <= [w_plus_observed, total_rank - w_plus_observed].min
    end

    {
      greater: Rational(tails[:greater], assignments),
      less: Rational(tails[:less], assignments),
      two_sided: Rational(tails[:two_sided], assignments),
      w_plus: w_plus_observed
    }
  end

  def interval_contains?(intervals, value)
    intervals.any? do |interval|
      lower_ok = interval[:lower_inclusive] ? value >= interval[:lower] : value > interval[:lower]
      upper_ok = interval[:upper_inclusive] ? value <= interval[:upper] : value < interval[:upper]
      lower_ok && upper_ok
    end
  end

  it 'has a short alias' do
    expect(described_class).to eq RubyStatistics::StatisticalTest::WilcoxonSignedRank
  end

  it 'can be invoked on the class' do
    result = described_class.perform(0.05, :two_tail, [1, 2, 3])

    expect(result[:p_value]).to eq 1/4r
  end

  it 'accepts the zero-difference rule as a positional option' do
    result = test_class.perform(0.05, :two_tail, [0, 1, 2, 3], nil, :pratt)

    expect(result[:zero_method]).to eq :pratt
    expect(result[:estimate]).to eq result[:point_estimate]
    expect(result[:acceptance_region]).to eq result[:acceptance_set]
  end

  describe '#signed_ranks' do
    it 'discards zero differences by default' do
      result = test_class.signed_ranks([0, -3, 3, 1])

      expect(result).to eq [-5/2r, 5/2r, 1r]
    end

    it 'uses the Pratt ranks for zero differences and tied absolute values' do
      result = test_class.signed_ranks([0, -3, 3, 1], :pratt)

      expect(result).to eq [0r, -7/2r, 7/2r, 2r]
    end
  end

  describe '#perform' do
    it 'computes exact one-sided and two-sided p-values for paired samples' do
      before = [48, 22, 34, 29, 20, 39]
      after = [42, 25, 27, 26, 20, 35]

      result = test_class.perform(0.05, :two_tail, before, after)

      expect(result[:w_plus]).to eq 27/2r
      expect(result[:w_minus]).to eq 3/2r
      expect(result[:p_value]).to eq 3/16r
      expect(result[:greater_p_value]).to eq 3/32r
      expect(result[:less_p_value]).to eq 31/32r
      expect(result[:exact]).to be true
    end

    it 'uses paired differences as first minus second' do
      result = test_class.perform(0.05, :greater, [7, 2], [3, 4])

      expect(result[:w_plus]).to eq 2r
      expect(result[:p_value]).to eq 1/2r
    end

    it 'supports explicit one-tail directions' do
      greater = test_class.perform(0.05, :one_tail, [4, 1], direction: :greater)
      less = test_class.perform(0.05, :one_tail, [4, 1], direction: :less)

      expect(greater[:alternative_hypothesis]).to eq :greater
      expect(less[:alternative_hypothesis]).to eq :less
      expect(greater[:p_value]).to eq 1/4r
      expect(less[:p_value]).to eq 1r
    end

    it 'infers the one-tail direction from the observed differences' do
      positive = test_class.perform(0.05, :one_tail, [4, 1])
      negative = test_class.perform(0.05, :one_tail, [-4, -1])

      expect(positive[:alternative_hypothesis]).to eq :greater
      expect(positive[:p_value]).to eq 1/4r
      expect(negative[:alternative_hypothesis]).to eq :less
      expect(negative[:p_value]).to eq 1/4r
    end

    it 'keeps tied absolute-value ranks as half-integers' do
      result = test_class.perform(0.05, :two_tail, [-3, 3, 1])

      expect(result[:signed_ranks]).to eq [-5/2r, 5/2r, 1r]
      expect(result[:w_plus]).to eq 7/2r
      expect(result[:p_value]).to eq 1r
    end

    it 'matches complete sign enumeration with non-zero absolute-rank ties' do
      differences = [-5, 5, 3, 1, -1, 9, -3]
      expected = brute_force_p_values(differences, :wilcox)
      result = test_class.perform(0.05, :two_tail, differences)

      expect(result[:w_plus]).to eq expected[:w_plus]
      expect(result[:p_value]).to eq expected[:two_sided]
      expect(result[:greater_p_value]).to eq expected[:greater]
      expect(result[:less_p_value]).to eq expected[:less]
    end

    it 'matches complete sign enumeration for every small sign pattern' do
      magnitudes = [1, 2, 2, 4]

      (0...(1 << magnitudes.size)).each do |mask|
        differences = magnitudes.map.with_index do |magnitude, index|
          mask[index] == 1 ? magnitude : -magnitude
        end
        expected = brute_force_p_values(differences, :wilcox)
        result = test_class.perform(0.05, :two_tail, differences)

        expect(result[:w_plus]).to eq expected[:w_plus]
        expect(result[:greater_p_value]).to eq expected[:greater]
        expect(result[:less_p_value]).to eq expected[:less]
        expect(result[:two_sided_p_value]).to eq expected[:two_sided]
      end
    end

    it 'matches complete sign enumeration with zero differences under both rules' do
      magnitudes = [1, 2, 2, 4]

      [:wilcox, :pratt].each do |zero_method|
        (0...(1 << magnitudes.size)).each do |mask|
          differences = [0, 0]
          magnitudes.each_with_index do |magnitude, index|
            differences << (mask[index] == 1 ? magnitude : -magnitude)
          end
          expected = brute_force_p_values(differences, zero_method)
          result = test_class.perform(0.05, :two_tail, differences, zero_method: zero_method)

          expect(result[:w_plus]).to eq expected[:w_plus]
          expect(result[:p_value]).to eq expected[:two_sided]
          expect(result[:greater_p_value]).to eq expected[:greater]
          expect(result[:less_p_value]).to eq expected[:less]
        end
      end
    end

    it 'matches complete sign enumeration when zero differences are discarded' do
      differences = [0, -5, 5, 3, 1, -1, 0, 9, -3]
      expected = brute_force_p_values(differences, :wilcox)
      result = test_class.perform(0.05, :two_tail, differences, zero_method: :discard)

      expect(result[:nonzero_n]).to eq 7
      expect(result[:zero_n]).to eq 2
      expect(result[:p_value]).to eq expected[:two_sided]
      expect(result[:greater_p_value]).to eq expected[:greater]
      expect(result[:less_p_value]).to eq expected[:less]
    end

    it 'matches complete Pratt sign enumeration over all observations' do
      differences = [0, -5, 5, 3, 1, -1, 0, 9, -3]
      expected = brute_force_p_values(differences, :pratt)
      result = test_class.perform(0.05, :two_tail, differences, zero_method: :pratt)

      expect(result[:nonzero_n]).to eq 7
      expect(result[:zero_n]).to eq 2
      expect(result[:w_plus]).to eq expected[:w_plus]
      expect(result[:p_value]).to eq expected[:two_sided]
      expect(result[:greater_p_value]).to eq expected[:greater]
      expect(result[:less_p_value]).to eq expected[:less]
    end

    it 'is invariant when all paired rows are reordered' do
      before = [48, 22, 34, 29, 20, 39]
      after = [42, 25, 27, 26, 20, 35]
      order = [3, 0, 5, 1, 4, 2]

      original = test_class.perform(0.05, :two_tail, before, after)
      reordered = test_class.perform(
        0.05, :two_tail,
        order.map { |index| before[index] },
        order.map { |index| after[index] }
      )

      expect(reordered.values_at(:w_plus, :w_minus, :p_value, :point_estimate))
        .to eq original.values_at(:w_plus, :w_minus, :p_value, :point_estimate)
    end

    it 'does not incorrectly pair independently shuffled groups' do
      before = [48, 22, 34]
      after = [42, 25, 27]

      expect(test_class.perform(0.05, :two_tail, before, after)[:w_plus])
        .not_to eq test_class.perform(0.05, :two_tail, before, after.reverse)[:w_plus]
    end

    it 'uses a consistent zero convention when every difference is zero' do
      result = test_class.perform(0.05, :two_tail, [0, 0, 0], zero_method: :pratt)

      expect(result[:w_plus]).to eq 0r
      expect(result[:w_minus]).to eq 0r
      expect(result[:p_value]).to eq 1r
      expect(result[:point_estimate]).to eq 0r
      expect(result[:confidence_interval]).to eq(
        lower: 0r,
        upper: 0r,
        lower_inclusive: true,
        upper_inclusive: true,
        envelope_contains_rejected_gaps: false,
        conservative: true,
        exact: true,
        confidence_level: 0.95
      )
    end

    it 'returns no point estimate if discarding all zero differences' do
      result = test_class.perform(0.05, :two_tail, [0, 0, 0])

      expect(result[:nonzero_n]).to eq 0
      expect(result[:p_value]).to eq 1r
      expect(result[:point_estimate]).to be_nil
      expect(result[:confidence_interval]).to be_nil
      expect(result[:acceptance_set]).to eq []
    end

    it 'inverts the exact test to obtain the point estimate and acceptance set' do
      differences = [-7, -5, -3, 1, 2, 8, 12, 16]
      result = test_class.perform(1/32r, :two_tail, differences)

      expect(result[:point_estimate]).to eq 5/2r
      expect(result[:acceptance_set]).to eq [
        { lower: -5r, upper: 12r, lower_inclusive: true, upper_inclusive: true }
      ]
      expect(result[:confidence_interval]).to include(
        lower: -5r,
        upper: 12r,
        envelope_contains_rejected_gaps: false,
        conservative: true
      )
    end

    it 'uses an exact level small enough to produce a finite location interval' do
      differences = [1, 2, 3, 4, 5]
      result = test_class.perform(1/8r, :two_tail, differences)

      expect(result[:point_estimate]).to eq 3r
      expect(interval_contains?(result[:acceptance_set], 1r)).to be false
      expect(interval_contains?(result[:acceptance_set], 149/100r)).to be false
      expect(interval_contains?(result[:acceptance_set], 3/2r - 1/1000r)).to be false
      expect(interval_contains?(result[:acceptance_set], 3/2r)).to be true
      expect(interval_contains?(result[:acceptance_set], 4r)).to be true
      expect(interval_contains?(result[:acceptance_set], 9/2r)).to be true
      expect(interval_contains?(result[:acceptance_set], 9/2r + 1/1000r)).to be false
      expect(result[:confidence_interval]).to include(
        lower: 3/2r,
        upper: 9/2r,
        envelope_contains_rejected_gaps: false
      )
    end

    it 'marks an envelope that spans rejected gaps as conservative' do
      differences = [1, 3, 8, -3, -6, -5, 4, 4, 4, -6]
      result = test_class.perform(0.05, :two_tail, differences)
      acceptance_set = result[:acceptance_set]
      envelope = result[:confidence_interval]

      expect(acceptance_set.size).to be > 1
      expect(envelope[:envelope_contains_rejected_gaps]).to be true
      expect(envelope[:conservative]).to be true
      expect(interval_contains?(acceptance_set, -4r)).to be true
      expect(interval_contains?(acceptance_set, -3r)).to be false
      expect(interval_contains?(acceptance_set, envelope[:lower])).to be true
      expect(interval_contains?(acceptance_set, envelope[:upper])).to be true
    end

    it 'inverts directional tests into one-sided location sets' do
      differences = [1, 2, 3, 4, 5, 6]
      greater = test_class.perform(0.05, :greater, differences)
      less = test_class.perform(0.05, :less, differences)

      expect(greater[:acceptance_set]).to eq [
        { lower: 2r, upper: Float::INFINITY, lower_inclusive: true, upper_inclusive: false }
      ]
      expect(less[:acceptance_set]).to eq [
        { lower: -Float::INFINITY, upper: 5r, lower_inclusive: false, upper_inclusive: true }
      ]
    end

    it 'validates paired sample sizes' do
      expect { test_class.perform(0.05, :two_tail, [1, 2], [1]) }
        .to raise_error ArgumentError, 'both paired samples must have the same size'
    end
  end
end
