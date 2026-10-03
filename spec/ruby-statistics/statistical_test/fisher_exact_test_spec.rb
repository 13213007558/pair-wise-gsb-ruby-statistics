require 'spec_helper'

describe RubyStatistics::StatisticalTest::FisherExactTest do
  describe '.observed_probability' do
    # Hand-computed: margins (10, 14) x (12, 12), N = 24.
    # P(table) = C(12,1) * C(12,9) / C(24,10) = 2640 / 1961256.
    it 'matches the hand-computed hypergeometric probability for a 2x2 table' do
      table = [[1, 9], [11, 3]]

      expect(described_class.observed_probability(table))
        .to be_within(1e-9).of(2640.0 / 1_961_256)
    end

    # Hand-computed: margins (4, 4) x (4, 4), N = 8.
    # P(table) = C(4,3) * C(4,1) / C(8,4) = 16 / 70.
    it 'matches the hand-computed probability for Fisher tea tasting table' do
      table = [[3, 1], [1, 3]]

      expect(described_class.observed_probability(table))
        .to be_within(1e-9).of(16.0 / 70)
    end
  end

  describe '.exact_test' do
    # Hand-computed: tables with margins (10, 14) x (12, 12) have weights
    # C(12,k) * C(12,10-k) for k in 0..10. The observed weight is 2640 and
    # only k in {0, 1, 9, 10} have weight <= 2640, so the two-sided p-value
    # is (66 + 2640 + 2640 + 66) / 1961256 = 5412 / 1961256.
    it 'matches the hand-computed two-sided p-value for a 2x2 table' do
      result = described_class.exact_test([[1, 9], [11, 3]])

      expect(result[:complete]).to be true
      expect(result[:observed_probability]).to be_within(1e-9).of(2640.0 / 1_961_256)
      expect(result[:p_value]).to be_within(1e-9).of(5412.0 / 1_961_256)
      expect(result[:total_mass]).to be_within(1e-9).of(1.0)
    end

    # Hand-computed: weights C(4,k) * C(4,4-k) for k in 0..4 are
    # 1, 16, 36, 16, 1 over C(8,4) = 70. The observed weight is 16, so the
    # two-sided p-value is (1 + 16 + 16 + 1) / 70 = 34 / 70.
    it 'matches the hand-computed two-sided p-value for the tea tasting table' do
      result = described_class.exact_test([[3, 1], [1, 3]])

      expect(result[:complete]).to be true
      expect(result[:p_value]).to be_within(1e-9).of(34.0 / 70)
    end

    it 'is invariant under row and column permutations' do
      table = [[2, 1, 0], [0, 1, 2], [1, 0, 1]]
      reference = described_class.exact_test(table)

      row_permuted = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
      column_permuted = [[1, 0, 2], [1, 2, 0], [0, 1, 1]]

      [row_permuted, column_permuted].each do |permuted|
        result = described_class.exact_test(permuted)

        expect(result[:p_value]).to be_within(1e-12).of(reference[:p_value])
        expect(result[:observed_probability])
          .to be_within(1e-12).of(reference[:observed_probability])
      end
    end

    it 'is invariant under transposition' do
      table = [[1, 2, 0, 3], [2, 0, 1, 1], [0, 1, 1, 0]]
      reference = described_class.exact_test(table)
      transposed = described_class.exact_test(table.transpose)

      expect(transposed[:p_value]).to be_within(1e-12).of(reference[:p_value])
      expect(transposed[:observed_probability])
        .to be_within(1e-12).of(reference[:observed_probability])
    end

    it 'enumerates the full conditional space with total probability one (2x2)' do
      result = described_class.exact_test([[1, 2], [2, 1]])

      expect(result[:complete]).to be true
      expect(result[:states]).to eq 4
      expect(result[:total_mass]).to be_within(1e-12).of(1.0)
      # Every table is at most as likely as the observed one here.
      expect(result[:p_value]).to be_within(1e-12).of(1.0)
    end

    it 'enumerates the full conditional space with total probability one (2x3)' do
      result = described_class.exact_test([[1, 2, 0], [2, 1, 1]])

      expect(result[:complete]).to be true
      expect(result[:total_mass]).to be_within(1e-12).of(1.0)
    end

    it 'enumerates the full conditional space with total probability one (3x3)' do
      result = described_class.exact_test([[2, 0, 1], [1, 1, 0], [0, 1, 1]])

      expect(result[:complete]).to be true
      expect(result[:total_mass]).to be_within(1e-12).of(1.0)
      expect(result[:accumulated_mass]).to be_within(1e-12).of(result[:p_value])
    end

    it 'drops all-zero rows and columns while keeping the result mapping' do
      table = [[1, 0, 9, 0], [0, 0, 0, 0], [11, 0, 3, 0]]
      result = described_class.exact_test(table)
      reference = described_class.exact_test([[1, 9], [11, 3]])

      expect(result[:reduced_table]).to eq [[1, 9], [11, 3]]
      expect(result[:nonzero_rows]).to eq [0, 2]
      expect(result[:nonzero_columns]).to eq [0, 2]
      expect(result[:row_margins]).to eq [10, 0, 14]
      expect(result[:column_margins]).to eq [12, 0, 12, 0]
      expect(result[:p_value]).to be_within(1e-12).of(reference[:p_value])
      expect(result[:observed_probability])
        .to be_within(1e-12).of(reference[:observed_probability])
    end

    it 'reports an incomplete search without a fake final p-value' do
      table = [[5, 5], [5, 5]]
      result = described_class.exact_test(table, max_states: 3)

      expect(result[:complete]).to be false
      expect(result[:p_value]).to be_nil
      expect(result[:states]).to eq 3
      expect(result[:accumulated_mass]).to be > 0.0
      expect(result[:total_mass]).to be < 1.0

      full = described_class.exact_test(table)
      expect(full[:complete]).to be true
      expect(result[:accumulated_mass]).to be <= full[:p_value]
    end

    it 'rejects tables with negative entries' do
      expect { described_class.exact_test([[1, -2], [3, 4]]) }
        .to raise_error(ArgumentError, /negative/)
    end

    it 'rejects tables with non-integer entries' do
      expect { described_class.exact_test([[1, 2.5], [3, 4]]) }
        .to raise_error(ArgumentError, /not an integer/)
    end

    it 'rejects ragged tables' do
      expect { described_class.exact_test([[1, 2], [3]]) }
        .to raise_error(ArgumentError, /same length/)
    end

    it 'rejects empty tables' do
      expect { described_class.exact_test([]) }.to raise_error(ArgumentError)
      expect { described_class.exact_test([[], []]) }.to raise_error(ArgumentError)
    end

    it 'rejects degenerate margins with a zero total count' do
      expect { described_class.exact_test([[0, 0], [0, 0]]) }
        .to raise_error(ArgumentError, /total count is zero/)
    end

    it 'rejects non positive max_states' do
      expect { described_class.exact_test([[1, 2], [3, 4]], max_states: 0) }
        .to raise_error(ArgumentError, /max_states/)
    end
  end
end
