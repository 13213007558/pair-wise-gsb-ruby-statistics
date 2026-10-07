require 'spec_helper'

describe RubyStatistics::MinimumCovarianceDeterminant do
  let(:inliers) do
    [
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0],
      [2.0, 1.0],
      [2.0, 0.0]
    ]
  end
  let(:outliers) { [[10.0, 10.0], [-10.0, 8.0]] }
  let(:data) { inliers + outliers }

  def transform(point)
    x, y = point
    [2.0 * x + y + 3.0, -x + y - 5.0]
  end

  def transform_scatter(matrix)
    linear = [[2.0, 1.0], [-1.0, 1.0]]
    linear.map do |row|
      Array.new(2) do |column|
        2.times.sum do |i|
          2.times.sum do |j|
            row[i] * matrix[i][j] * linear[column][j]
          end
        end
      end
    end
  end

  describe '.fit' do
    it 'finds the exact minimum-covariance subset in two dimensions' do
      result = described_class.fit(data, reweight: true)

      expect(result.h).to eq(5)
      expect(result.support.size).to eq(5)
      expect(result.support - (0...inliers.size).to_a).to be_empty
      expect(result.verified_global_optimum?).to be(true)
      expect(result.mode).to eq(:exact_enumeration)
      expect(result.weights).to eq([1, 1, 1, 1, 1, 1, 0, 0])
      expect(result.reweighted?).to be(true)
      expect(result.center).to eq([0.8, 0.6])
      expect(result.scatter[0][0]).to be_within(1e-12).of(0.56)
      expect(result.scatter[0][1]).to be_within(1e-12).of(0.12)
      expect(result.scatter[1][0]).to be_within(1e-12).of(0.12)
      expect(result.scatter[1][1]).to be_within(1e-12).of(0.24)
      expect(result.reweighted_center).to eq([1.0, 0.5])
      expect(result.reweighted_scatter).to eq([
        [0.6666666666666666, 0.0],
        [0.0, 0.25]
      ])
    end

    it 'does not replace raw estimates unless reweighting is requested' do
      result = described_class.fit(data)

      expect(result.reweighted?).to be(false)
      expect(result.reweighted_center).to be_nil
      expect(result.reweighted_scatter).to be_nil
      expect(result.weights).to eq([1, 1, 1, 1, 1, 1, 0, 0])
      expect(result.threshold).to be_within(1e-10).of(7.377758908227866)
    end

    it 'is affine equivariant under an invertible transformation' do
      original = described_class.fit(data, reweight: true)
      transformed_data = data.map { |point| transform(point) }
      transformed = described_class.fit(transformed_data, reweight: true)
      transformed_center = transform(original.center)
      transformed_reweighted_center = transform(original.reweighted_center)
      expected_scatter = transform_scatter(original.scatter)
      expected_reweighted = transform_scatter(original.reweighted_scatter)

      expect(transformed.support).to eq(original.support)
      expect(transformed.center).to eq(transformed_center)
      expect(transformed.reweighted_center).to eq(transformed_reweighted_center)
      expect(transformed.weights).to eq(original.weights)
      expect(transformed.determinant).to be_within(1e-10).of(9.0 * original.determinant)
      transformed.scatter.each_with_index do |row, row_index|
        row.each_with_index do |value, column_index|
          expect(value).to be_within(1e-10).of(expected_scatter[row_index][column_index])
        end
      end
      transformed.reweighted_scatter.each_with_index do |row, row_index|
        row.each_with_index do |value, column_index|
          expect(value).to be_within(1e-10).of(expected_reweighted[row_index][column_index])
        end
      end
    end

    it 'explains moving an outlier farther away without changing the raw subset' do
      near_outlier = data.dup
      near_outlier[-1] = [12.0, 9.0]
      far_outlier = data.dup
      far_outlier[-1] = [120.0, 90.0]

      near_result = described_class.fit(near_outlier)
      far_result = described_class.fit(far_outlier)
      near_distances = described_class.squared_mahalanobis_distances(
        near_outlier,
        center: near_result.center,
        scatter: near_result.scatter
      )
      far_distances = described_class.squared_mahalanobis_distances(
        far_outlier,
        center: far_result.center,
        scatter: far_result.scatter
      )

      expect(far_result.support).to eq(near_result.support)
      expect(far_result.center).to eq(near_result.center)
      expect(far_result.scatter).to eq(near_result.scatter)
      expect(far_result.weights).to eq(near_result.weights)
      expect(far_distances.last).to be > near_distances.last
    end

    it 'returns a deterministic search candidate without claiming verified optimality' do
      first = described_class.fit(data, exact: false, starts: 25, seed: 12_345)
      second = described_class.fit(data, exact: false, starts: 25, seed: 12_345)

      expect(first.search_candidate?).to be(true)
      expect(first.verified_global_optimum?).to be(false)
      expect(first.mode).to eq(:multi_start_c_step)
      expect(first.starts).to eq(25)
      expect(first.iterations).to be > 0
      expect(first.singular_subsets).to be_an(Integer)
      expect(first.support).to eq(second.support)
      expect(first.center).to eq(second.center)
      expect(first.scatter).to eq(second.scatter)
      expect(first.support - (0...inliers.size).to_a).to be_empty
    end

    it 'skips singular subsets while retaining duplicate observations' do
      duplicated = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [10.0, 10.0]
      ]
      result = described_class.fit(duplicated)

      expect(result.support.size).to eq(4)
      expect(result.singular_subsets).to be > 0
      expect(result.determinant).to be > 0
      expect(result.weights).to all(eq(1).or eq(0))
    end

    it 'can start C-steps from a deterministic full-rank subset with duplicates' do
      duplicated = [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [10.0, 10.0]
      ]
      result = described_class.fit(duplicated, exact: false, starts: 10, seed: 7)

      expect(result.search_candidate?).to be(true)
      expect(result.starts).to eq(10)
      expect(result.determinant).to be > 0
    end

    it 'can choose a different anchor when the first observation is duplicated' do
      duplicated_anchor = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
      ]
      result = described_class.fit(duplicated_anchor, exact: false, starts: 5, seed: 11)

      expect(result.search_candidate?).to be(true)
      expect(result.determinant).to be > 0
    end

    it 'raises a rank error for data in a lower-dimensional subspace' do
      collinear = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]

      expect { described_class.fit(collinear) }.to raise_error(
        described_class::SingularCovarianceError,
        /lower-dimensional subspace/
      )
    end

    it 'rejects h values that are too small or too large' do
      expect { described_class.fit(data, h: 2) }.to raise_error(ArgumentError, /between 5 and 8/)
      expect { described_class.fit(data, h: 9) }.to raise_error(ArgumentError, /between 5 and 8/)
    end

    it 'does not silently reweight when the threshold selects no observations' do
      result = described_class.fit(data)

      expect {
        described_class.reweight(
          data,
          center: result.center,
          scatter: result.scatter,
          threshold: 0.0
        )
      }.to raise_error(ArgumentError, /selected no observations/)
    end

    it 'rejects non-finite observations instead of perturbing them' do
      expect { described_class.fit([[0, 0], [1, 0], [0, 1], [Float::NAN, 1]]) }.to raise_error(
        ArgumentError,
        /finite values/
      )
    end
  end
end
