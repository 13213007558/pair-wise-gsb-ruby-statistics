require 'spec_helper'

describe RubyStatistics::MinimumCovarianceDeterminant do
  def multiply(matrix, vector)
    matrix.map do |row|
      row.each_with_index.sum { |value, index| value * vector[index] }
    end
  end

  def scatter_after_transform(scatter, transform)
    size = transform.size
    Array.new(size) do |row|
      Array.new(size) do |column|
        size.times.sum do |i|
          size.times.sum do |j|
            transform[row][i] * scatter[i][j] * transform[column][j]
          end
        end
      end
    end
  end

  describe '#initialize' do
    it 'enumerates the minimum-determinant h-subset for small two-dimensional data' do
      data = [[0, 0], [0, 1], [1, 0], [1, 1], [10, 10]]
      result = described_class.fit(data, h: 4, reweight: true)

      expect(result.verified_global_optimum?).to be(true)
      expect(result.search_candidate?).to be(false)
      expect(result.support).to eq([0, 1, 2, 3])
      expect(result.center).to contain_exactly(0.5, 0.5)
      expect(result.scatter).to eq([[0.25, 0.0], [0.0, 0.25]])
      expect(result.determinant).to be_within(1e-12).of(0.0625)
      expect(result.reweighted_support).to eq([0, 1, 2, 3])
      expect(result.reweighted_weights).to eq([1.0, 1.0, 1.0, 1.0, 0.0])
    end

    it 'keeps the same raw support under an invertible affine transform and transforms estimates equivariantly' do
      data = [[0, 0], [0, 1], [1, 0], [1, 1], [8, 7]]
      transform = [[2.0, -1.0], [1.0, 3.0]]
      shift = [4.0, -5.0]
      transformed = data.map do |point|
        multiply(transform, point).each_with_index.map { |value, index| value + shift[index] }
      end

      original = described_class.fit(data, h: 4, reweight: true)
      affine = described_class.fit(transformed, h: 4, reweight: true)

      expected_center = multiply(transform, original.center).each_with_index.map { |value, index| value + shift[index] }
      expected_scatter = scatter_after_transform(original.scatter, transform)
      expected_determinant = (2.0 * 3.0 - (-1.0) * 1.0) ** 2 * original.determinant

      expect(affine.support).to eq(original.support)
      expect(affine.center).to eq(expected_center)
      expect(affine.scatter).to match_matrix(expected_scatter)
      expect(affine.determinant).to be_within(1e-10).of(expected_determinant)
      expect(affine.reweighted_weights).to eq(original.reweighted_weights)
      reweighted_center = multiply(transform, original.reweighted_center)
        .each_with_index.map { |value, index| value + shift[index] }
      expect(affine.reweighted_center).to eq(reweighted_center)
      expect(affine.reweighted_scatter).to match_matrix(scatter_after_transform(original.reweighted_scatter, transform))
    end

    it 'does not change the selected support when an excluded outlier is moved farther away' do
      close = described_class.fit([[0, 0], [0, 1], [1, 0], [1, 1], [8, 7]], h: 4)
      far = described_class.fit([[0, 0], [0, 1], [1, 0], [1, 1], [80, 70]], h: 4)

      expect(far.support).to eq(close.support)
      expect(far.center).to eq(close.center)
      expect(far.scatter).to match_matrix(close.scatter)
    end

    it 'reports a singular fit when all observations lie in a lower-dimensional affine subspace' do
      data = [[0, 0], [1, 1], [2, 2], [3, 3], [100, 100]]
      result = described_class.fit(data, h: 3, reweight: true)

      expect(result.verified_global_optimum?).to be(true)
      expect(result.rank).to eq(1)
      expect(result.singular?).to be(true)
      expect(result.support).to eq([0, 1, 2])
      expect(result.determinant).to eq(0.0)
      expect(result.generalized_determinant).to be > 0.0
      expect(result.scatter.map { |row| row.map(&:zero?) }.flatten.uniq).to eq([false])
      expect(result.reweighted_weights).to eq([1.0, 1.0, 1.0, 0.0, 0.0])
      expect(result.reweighted_support).to eq([0, 1, 2])
      expect(result.reweighted_generalized_determinant).to be > 0.0
      expect(result.warnings).to include('data lie in an affine subspace of rank 1; reported scatter is singular')
    end

    it 'reconstructs singular scatter for a non-axis-aligned affine subspace' do
      data = [[1, 2], [3, 1], [5, 0], [9, -2], [100, -47.5]]
      result = described_class.fit(data, h: 3)

      expect(result.rank).to eq(1)
      expect(result.singular?).to be(true)
      expect(result.determinant).to eq(0.0)
      expect(result.generalized_determinant).to be > 0.0
      expect(result.scatter).to match_matrix([
        [2.666666666666667, -1.3333333333333333],
        [-1.3333333333333333, 0.6666666666666667]
      ])
    end

    it 'retains duplicate observations and does not synthesize noise' do
      data = [[0, 0], [0, 0], [1, 1], [2, 2], [100, 100]]
      result = described_class.fit(data, h: 3)

      expect(result.support).to eq([0, 1, 2])
      expect(result.rank).to eq(1)
      expect(result.singular?).to be(true)
      expect(result.warnings).to include('duplicate observations are retained without adding noise')
    end

    it 'does not let a singular duplicate subset win the determinant objective in full-rank data' do
      data = [[0, 0], [0, 0], [1, 0], [0, 1]]
      result = described_class.fit(data, h: 3)

      expect(result.verified_global_optimum?).to be(true)
      expect(result.rank).to eq(2)
      expect(result.singular?).to be(false)
      expect(result.support).to eq([0, 2, 3])
      expect(result.determinant).to be > 0.0
    end

    it 'returns a zero scatter when all observations are identical' do
      result = described_class.fit(Array.new(4, [2.0, -3.0]), h: 1, reweight: true)

      expect(result.verified_global_optimum?).to be(true)
      expect(result.rank).to eq(0)
      expect(result.singular?).to be(true)
      expect(result.center).to eq([2.0, -3.0])
      expect(result.scatter).to eq(Array.new(2) { Array.new(2, 0.0) })
      expect(result.reweighted_weights).to eq([1.0, 1.0, 1.0, 1.0])
      expect(result.warnings).to include('all observations are identical; scatter is a zero matrix')
    end

    it 'raises instead of regularizing when h cannot span the observed affine space' do
      data = [[0, 0], [1, 1], [2, 2], [3, 3]]
      expect { described_class.fit(data, h: 1) }
        .to raise_error(ArgumentError, 'h must be at least 2 to span the data')
    end

    it 'raises for h outside the sample range' do
      expect { described_class.fit([[0, 0], [1, 1]], h: 3) }
        .to raise_error(ArgumentError, 'h must be between 1 and the number of observations')
    end

    it 'raises for non-integer h instead of silently truncating it' do
      expect { described_class.fit([[0, 0], [1, 1], [2, 2]], h: 1.5) }
        .to raise_error(ArgumentError, 'h must be an integer')
    end

    it 'marks fixed-seed C-step results as unverified search candidates' do
      data = (0...30).map do |index|
        [Math.sin(index) * 2.0, Math.cos(index * 1.7) * 3.0]
      end + [[60.0, 80.0]]

      result = described_class.fit(data, starts: 5, seed: 42)
      repeated = described_class.fit(data, starts: 5, seed: 42)

      expect(result.search_candidate?).to be(true)
      expect(result.verified_global_optimum?).to be(false)
      expect(result.support).to eq(repeated.support)
      expect(result.center).to eq(repeated.center)
    end

    it 'still labels a result as a search candidate even when deterministic starts find a stable optimum' do
      data = (0...20).map { |index| [index.to_f, Math.sin(index)] } + [[50.0, 50.0]]
      result = described_class.fit(data, h: 11, starts: 1, seed: 1)

      expect(result.search_candidate?).to be(true)
      expect(result.verified_global_optimum?).to be(false)
      expect(result.h).to eq(11)
    end

    it 'supports an explicitly requested post-fit reweighting step' do
      fit = described_class.fit([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10]], h: 4)
      expect(fit.reweighted?).to be(false)

      reweighted = fit.reweight
      expect(reweighted.reweighted?).to be(true)
      expect(reweighted.support).to eq(fit.support)
      expect(reweighted.reweighted_weights).to eq([1.0, 1.0, 1.0, 1.0, 0.0])
      expect(fit.reweighted_weights).to be_nil
    end
  end

  matcher :match_matrix do |expected|
    supports_block_expectations

    match do |actual|
      actual.size == expected.size &&
        actual.each_with_index.all? do |row, row_index|
          row.each_with_index.all? do |value, column_index|
            (value - expected[row_index][column_index]).abs < 1e-12
          end
        end
    end
  end
end
