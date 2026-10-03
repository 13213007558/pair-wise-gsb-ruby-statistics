require 'spec_helper'

describe RubyStatistics::Distribution::BoundedKernelDensity do
  let(:lower) { 0.0 }
  let(:upper) { 1.0 }
  let(:data) { [0.2, 0.4, 0.5, 0.7, 0.9] }
  let(:bandwidth) { 0.15 }

  # Slow, independent reference implementation: every evaluation rebuilds
  # the mirror images from scratch and sums them term by term. The
  # optimised object precomputes the per-observation images once; these
  # helpers exist to verify that the optimised path returns the same
  # numbers as the straightforward mirror summation.
  def slow_images(x, lower, upper, reflections)
    length = upper - lower
    (-reflections..reflections).flat_map do |k|
      [x + 2.0 * k * length, 2.0 * lower - x + 2.0 * k * length]
    end
  end

  def slow_pdf(z)
    Math.exp(-0.5 * z * z) / Math.sqrt(2.0 * Math::PI)
  end

  def slow_cdf(z)
    0.5 * (1.0 + Math.erf(z / Math.sqrt(2.0)))
  end

  def slow_norm(data, lower, upper, bandwidth, reflections)
    data.sum do |xi|
      slow_images(xi, lower, upper, reflections).sum do |y|
        slow_cdf((upper - y) / bandwidth) - slow_cdf((lower - y) / bandwidth)
      end
    end
  end

  def slow_density(data, lower, upper, bandwidth, reflections, x)
    kernel = data.sum do |xi|
      slow_images(xi, lower, upper, reflections).sum do |y|
        slow_pdf((x - y) / bandwidth) / bandwidth
      end
    end
    kernel / slow_norm(data, lower, upper, bandwidth, reflections)
  end

  def slow_cumulative(data, lower, upper, bandwidth, reflections, x)
    mass = data.sum do |xi|
      slow_images(xi, lower, upper, reflections).sum do |y|
        slow_cdf((x - y) / bandwidth) - slow_cdf((lower - y) / bandwidth)
      end
    end
    mass / slow_norm(data, lower, upper, bandwidth, reflections)
  end

  def slow_loo(data, lower, upper, bandwidth, reflections)
    data.each_with_index.sum do |x, i|
      rest = data.each_with_index.reject { |_, j| j == i }.map(&:first)
      numerator = rest.sum do |xj|
        slow_images(xj, lower, upper, reflections).sum { |y| slow_pdf((x - y) / bandwidth) / bandwidth }
      end
      denominator = rest.sum do |xj|
        slow_images(xj, lower, upper, reflections).sum do |y|
          slow_cdf((upper - y) / bandwidth) - slow_cdf((lower - y) / bandwidth)
        end
      end
      Math.log(numerator / denominator)
    end
  end

  def simpson(kde, from, to, steps = 2_000)
    h = (to - from) / steps
    (0..steps).sum do |i|
      weight = i.zero? || i == steps ? 1 : (i.odd? ? 4 : 2)
      weight * kde.density_function(from + i * h)
    end * h / 3.0
  end

  describe 'validation' do
    it 'rejects an inverted interval' do
      expect { described_class.new(data, lower: 1.0, upper: 0.0, bandwidth: 0.1) }
        .to raise_error(ArgumentError)
    end

    it 'rejects empty data' do
      expect { described_class.new([], lower: 0.0, upper: 1.0, bandwidth: 0.1) }
        .to raise_error(ArgumentError)
    end

    it 'rejects observations outside the interval' do
      expect { described_class.new([0.5, 1.5], lower: 0.0, upper: 1.0, bandwidth: 0.1) }
        .to raise_error(ArgumentError, /outside/)
    end

    it 'rejects a non-positive bandwidth' do
      expect { described_class.new(data, lower: 0.0, upper: 1.0, bandwidth: 0.0) }
        .to raise_error(ArgumentError)
      expect { described_class.new(data, lower: 0.0, upper: 1.0, bandwidth: -0.5) }
        .to raise_error(ArgumentError)
    end

    it 'requires a bandwidth or the :loo selector' do
      expect { described_class.new(data, lower: 0.0, upper: 1.0) }
        .to raise_error(ArgumentError)
    end

    it 'requires a search interval when bandwidth is :loo' do
      expect { described_class.new(data, lower: 0.0, upper: 1.0, bandwidth: :loo) }
        .to raise_error(ArgumentError)
    end
  end

  describe '#density_function' do
    it 'is zero outside the interval' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.density_function(-0.5)).to eq 0.0
      expect(kde.density_function(1.5)).to eq 0.0
    end

    it 'matches the slow mirror summation on a small sample' do
      small = [0.15, 0.35, 0.35, 0.8]
      kde = described_class.new(small, lower: lower, upper: upper, bandwidth: 0.1)

      [0.0, 0.13, 0.35, 0.5, 0.77, 1.0].each do |x|
        expected = slow_density(small, lower, upper, 0.1, kde.reflections, x)
        expect(kde.density_function(x)).to be_within(1e-12).of(expected)
      end
    end

    it 'integrates to one over the whole interval' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(simpson(kde, lower, upper)).to be_within(1e-8).of(1.0)
    end
  end

  describe '#cumulative_function' do
    it 'is exactly zero below and exactly one above the interval' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.cumulative_function(-1.0)).to eq 0.0
      expect(kde.cumulative_function(2.0)).to eq 1.0
    end

    it 'is zero at the lower bound and one at the upper bound' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.cumulative_function(lower)).to be_within(1e-15).of(0.0)
      expect(kde.cumulative_function(upper)).to be_within(1e-15).of(1.0)
    end

    it 'is non-decreasing' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)
      values = (0..100).map { |i| kde.cumulative_function(i / 100.0) }

      expect(values.each_cons(2).all? { |a, b| b >= a }).to be true
    end

    it 'matches the slow mirror summation on a small sample' do
      small = [0.15, 0.35, 0.35, 0.8]
      kde = described_class.new(small, lower: lower, upper: upper, bandwidth: 0.1)

      [0.0, 0.2, 0.35, 0.6, 0.95, 1.0].each do |x|
        expected = slow_cumulative(small, lower, upper, 0.1, kde.reflections, x)
        expect(kde.cumulative_function(x)).to be_within(1e-12).of(expected)
      end
    end
  end

  describe '#probability' do
    it 'matches the difference of the cumulative function' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.probability(0.2, 0.6))
        .to eq(kde.cumulative_function(0.6) - kde.cumulative_function(0.2))
    end

    it 'matches numerical integration of the density' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.probability(0.1, 0.8)).to be_within(1e-6).of(simpson(kde, 0.1, 0.8))
    end

    it 'clips the query interval to the support' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.probability(-5.0, 5.0)).to be_within(1e-15).of(1.0)
      expect(kde.probability(-5.0, -1.0)).to eq 0.0
      expect(kde.probability(2.0, 3.0)).to eq 0.0
    end

    it 'is zero for an empty or inverted query interval' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.probability(0.5, 0.5)).to eq 0.0
      expect(kde.probability(0.7, 0.3)).to eq 0.0
    end
  end

  describe '#truncation_error' do
    it 'stays below the requested tolerance' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)
      error = kde.truncation_error

      expect(error[:omitted_density]).to be <= described_class::DEFAULT_TOLERANCE
      expect(error[:omitted_mass]).to be <= described_class::DEFAULT_TOLERANCE * kde.length
      expect(error[:normalization]).to eq kde.normalization
      expect(error[:density]).to be >= error[:omitted_density]
      expect(error[:cumulative]).to be >= 0.0
    end

    it 'shrinks when more reflections are used' do
      small = described_class.truncation_density_bound(0.5, 1.0, 1)
      larger = described_class.truncation_density_bound(0.5, 1.0, 3)

      expect(larger).to be < small
    end
  end

  describe '.reflection_count' do
    it 'raises when the tolerance cannot be reached' do
      expect { described_class.reflection_count(100.0, 1.0, 1e-300) }
        .to raise_error(ArgumentError, /tolerance/)
    end

    it 'respects an explicitly given number of reflections' do
      kde = described_class.new(data, lower: lower, upper: upper,
                                bandwidth: bandwidth, reflections: 4)

      expect(kde.reflections).to eq 4
    end
  end

  describe '.leave_one_out_log_likelihood' do
    it 'matches the slow leave-one-out summation on a small sample' do
      small = [0.15, 0.35, 0.35, 0.8]

      [0.05, 0.1, 0.3].each do |h|
        expected = slow_loo(small, lower, upper, h, 2)
        actual = described_class.leave_one_out_log_likelihood(
          small, lower: lower, upper: upper, bandwidth: h, reflections: 2
        )

        expect(actual).to be_within(1e-10).of(expected)
      end
    end
  end

  describe '.select_bandwidth' do
    let(:clustered) { [0.42, 0.46, 0.48, 0.5, 0.52, 0.54, 0.58] }

    it 'rejects a non-positive or inverted search interval' do
      expect do
        described_class.select_bandwidth(data, lower: lower, upper: upper, interval: [0.0, 0.5])
      end.to raise_error(ArgumentError)
      expect do
        described_class.select_bandwidth(data, lower: lower, upper: upper, interval: [0.5, 0.1])
      end.to raise_error(ArgumentError)
    end

    it 'requires at least two observations' do
      expect do
        described_class.select_bandwidth([0.5], lower: lower, upper: upper, interval: [0.01, 0.5])
      end.to raise_error(ArgumentError)
    end

    it 'finds an interior optimum for clustered data' do
      selection = described_class.select_bandwidth(
        clustered, lower: lower, upper: upper, interval: [0.01, 0.5]
      )

      expect(selection.bandwidth).to be_between(0.01, 0.5).exclusive
      expect(selection.interior_optimum?).to be true
      expect(selection.warnings).to be_empty
    end

    it 'maximises the leave-one-out log-likelihood at the selected bandwidth' do
      selection = described_class.select_bandwidth(
        clustered, lower: lower, upper: upper, interval: [0.01, 0.5]
      )

      [0.02, 0.04, 0.08, 0.15, 0.3].each do |h|
        other = described_class.leave_one_out_log_likelihood(
          clustered, lower: lower, upper: upper, bandwidth: h
        )
        expect(selection.log_likelihood).to be >= other
      end
    end

    it 'reports an optimum at the lower edge when all observations coincide' do
      selection = described_class.select_bandwidth(
        [0.5, 0.5, 0.5, 0.5], lower: lower, upper: upper, interval: [0.01, 0.5]
      )

      expect(selection.at_lower_bound?).to be true
      expect(selection.interior_optimum?).to be false
      expect(selection.warnings.join).to match(/lower edge/)
      expect(selection.warnings.join).to match(/duplicate/)
      expect(selection.duplicates).to eq(0.5 => 4)
    end

    it 'reports an optimum at the upper edge for spread out data' do
      selection = described_class.select_bandwidth(
        data, lower: lower, upper: upper, interval: [0.02, 0.5]
      )

      expect(selection.at_upper_bound?).to be true
      expect(selection.interior_optimum?).to be false
      expect(selection.warnings.join).to match(/upper edge/)
    end

    it 'reports observations lying exactly on the boundaries' do
      selection = described_class.select_bandwidth(
        [0.0, 0.25, 0.5, 1.0], lower: lower, upper: upper, interval: [0.05, 0.5]
      )

      expect(selection.boundary_points).to contain_exactly(0.0, 1.0)
      expect(selection.warnings.join).to match(/boundary/)
    end
  end

  describe 'bandwidth: :loo' do
    it 'builds a density from the selected bandwidth' do
      clustered = [0.42, 0.46, 0.48, 0.5, 0.52, 0.54, 0.58]
      kde = described_class.new(
        clustered, lower: lower, upper: upper,
        bandwidth: :loo, bandwidth_interval: [0.01, 0.5]
      )

      expect(kde.bandwidth).to eq kde.bandwidth_selection.bandwidth
      expect(simpson(kde, lower, upper)).to be_within(1e-8).of(1.0)
    end
  end

  describe '#diagnostics' do
    it 'reports duplicates, boundary points and warnings' do
      kde = described_class.new([0.0, 0.5, 0.5, 1.0], lower: lower, upper: upper,
                                bandwidth: :loo, bandwidth_interval: [0.05, 0.5])
      diagnostics = kde.diagnostics

      expect(diagnostics[:sample_size]).to eq 4
      expect(diagnostics[:duplicates]).to eq(0.5 => 2)
      expect(diagnostics[:boundary_points]).to contain_exactly(0.0, 1.0)
      expect(diagnostics[:warnings].join).to match(/duplicate/)
      expect(diagnostics[:warnings].join).to match(/boundary/)
      expect(diagnostics[:bandwidth_selection]).to eq kde.bandwidth_selection
    end

    it 'has no warnings for well separated interior data with a fixed bandwidth' do
      kde = described_class.new(data, lower: lower, upper: upper, bandwidth: bandwidth)

      expect(kde.warnings).to be_empty
      expect(kde.diagnostics[:duplicates]).to be_empty
      expect(kde.diagnostics[:boundary_points]).to be_empty
    end
  end
end
