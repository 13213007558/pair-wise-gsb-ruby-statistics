require 'spec_helper'

describe RubyStatistics::StatisticalTest::FisherExactTest do
  describe '.perform' do
    it 'calculates a hand-computed 2x2 observed probability and two-sided p-value' do
      result = described_class.perform([[3, 1], [1, 3]], max_states: Float::INFINITY)

      expect(result[:observed_probability]).to eq(Rational(8, 35))
      expect(result[:p_value]).to eq(Rational(17, 35))
      expect(result[:two_sided_p_value]).to eq(Rational(17, 35))
      expect(result[:accumulated_probability]).to eq(1r)
      expect(result[:visited_states]).to eq(5)
      expect(result[:total_states]).to eq(5)
      expect(result[:status]).to eq(:complete)
    end

    it 'uses probability no greater than the observed table for the two-sided tail' do
      result = described_class.perform([[0, 4], [3, 3]], max_states: Float::INFINITY)

      expect(result[:observed_probability]).to eq(Rational(1, 6))
      expect(result[:p_value]).to eq(Rational(1, 5))
    end

    it 'is invariant under row and column permutations' do
      table = [[0, 4], [3, 3]]
      row_permuted = [[3, 3], [0, 4]]
      column_permuted = [[4, 0], [3, 3]]
      expected = described_class.perform(table, max_states: Float::INFINITY)

      expect(described_class.perform(row_permuted, max_states: Float::INFINITY)).to include(
        observed_probability: expected[:observed_probability],
        p_value: expected[:p_value],
        total_states: expected[:total_states]
      )
      expect(described_class.perform(column_permuted, max_states: Float::INFINITY)).to include(
        observed_probability: expected[:observed_probability],
        p_value: expected[:p_value],
        total_states: expected[:total_states]
      )
    end

    it 'is invariant under transposition' do
      table = [[0, 4], [3, 3]]
      transposed = [[0, 3], [4, 3]]
      expected = described_class.perform(table, max_states: Float::INFINITY)

      expect(described_class.perform(transposed, max_states: Float::INFINITY)).to include(
        observed_probability: expected[:observed_probability],
        p_value: expected[:p_value],
        total_states: expected[:total_states]
      )
    end

    it 'removes all-zero rows and columns while retaining their original index mapping' do
      table = [
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0],
        [0, 1, 0]
      ]

      result = described_class.perform(table, max_states: Float::INFINITY)

      expect(result[:reduced_table]).to eq([[1, 0, 1], [0, 1, 0]])
      expect(result[:row_indices]).to eq([1, 3])
      expect(result[:column_indices]).to eq([0, 1, 2])
      expect(result[:observed_probability]).to eq(Rational(1, 3))
      expect(result[:p_value]).to eq(1r)
    end

    it 'accepts an all-zero table as a single empty-margin conditional state' do
      result = described_class.perform([[0, 0], [0, 0]], max_states: Float::INFINITY)

      expect(result[:reduced_table]).to eq([])
      expect(result[:row_indices]).to eq([])
      expect(result[:column_indices]).to eq([])
      expect(result[:observed_probability]).to eq(1r)
      expect(result[:p_value]).to eq(1r)
      expect(result[:status]).to eq(:complete)
    end

    it 'reports an incomplete search without presenting a final p-value' do
      result = described_class.perform([[3, 1], [1, 3]], max_states: 2)

      expect(result[:status]).to eq(:incomplete)
      expect(result[:complete]).to be(false)
      expect(result[:p_value]).to be_nil
      expect(result[:two_sided_p_value]).to be_nil
      expect(result[:visited_states]).to eq(2)
      expect(result[:total_states]).to be_nil
      expect(result[:accumulated_probability]).to eq(Rational(17, 70))
      expect(result[:accumulated_probability]).to be < 1r
    end

    it 'is complete when the state limit is exactly the number of fixed-margin tables' do
      result = described_class.perform([[3, 1], [1, 3]], max_states: 5)

      expect(result[:status]).to eq(:complete)
      expect(result[:p_value]).to eq(Rational(17, 35))
      expect(result[:accumulated_probability]).to eq(1r)
    end

    it 'includes the observed table as the first searched state when the limit is one' do
      result = described_class.perform([[3, 1], [1, 3]], max_states: 1)

      expect(result[:status]).to eq(:incomplete)
      expect(result[:p_value]).to be_nil
      expect(result[:visited_states]).to eq(1)
      expect(result[:accumulated_probability]).to eq(Rational(8, 35))
    end

    it 'rejects invalid margins and non-integer or negative values' do
      expect { described_class.perform([[-1, 2], [2, 1]]) }.to raise_error(ArgumentError)
      expect { described_class.perform([[1, 2], [2]]) }.to raise_error(ArgumentError)
      expect { described_class.perform([[], []]) }.to raise_error(ArgumentError)
      expect { described_class.perform([]) }.to raise_error(ArgumentError)
      expect { described_class.perform([[1.0, 2], [2, 1]]) }.to raise_error(ArgumentError)
      expect { described_class.perform([[1, 2], [2, 1]], max_states: 0) }.to raise_error(ArgumentError)
      expect { described_class.perform([[1, 2], [2, 1]], max_states: -1) }.to raise_error(ArgumentError)
    end

    it 'uses a documented non-negative floating-point comparison tolerance' do
      exact = described_class.perform([[3, 1], [1, 3]], max_states: Float::INFINITY)
      tolerant = described_class.perform([[3, 1], [1, 3]], max_states: Float::INFINITY, tolerance: 1e-12)

      expect(exact[:comparison_tolerance]).to eq(0.0)
      expect(tolerant[:comparison_tolerance]).to eq(1e-12)
      expect(tolerant[:p_value]).to eq(exact[:p_value])
      expect { described_class.perform([[3, 1], [1, 3]], tolerance: -1e-12) }.to raise_error(ArgumentError)
    end
  end

  describe '.feasible_tables' do
    it 'enumerates every fixed-margin table whose probabilities sum to one' do
      observed = [[1, 0, 1], [0, 1, 0]]
      candidates = described_class.feasible_tables(observed).to_a
      all_tables = candidates + [observed]

      expect(candidates).to contain_exactly(
        [[1, 1, 0], [0, 0, 1]],
        [[0, 1, 1], [1, 0, 0]]
      )

      total_probability = all_tables.sum do |table|
        described_class.perform(table, max_states: Float::INFINITY)[:observed_probability]
      end

      expect(total_probability).to eq(1r)
    end

    it 'full-enumerates a small 3x4 table after zero margins are removed' do
      observed = [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1]
      ]
      result = described_class.perform(observed, max_states: Float::INFINITY)
      reduced = result[:reduced_table]
      all_probabilities = described_class.feasible_tables(reduced).map do |table|
        described_class.perform(table, max_states: Float::INFINITY)[:observed_probability]
      end
      all_probabilities << result[:observed_probability]

      expect(result[:row_indices]).to eq([1, 2, 3])
      expect(result[:column_indices]).to eq([0, 1, 2, 3])
      expect(result[:total_states]).to eq(33)
      expect(all_probabilities.sum).to eq(1r)
    end
  end
end
