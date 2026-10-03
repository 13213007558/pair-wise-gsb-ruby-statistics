require 'spec_helper'

describe RubyStatistics::Distribution::BoundedKDE do
  let(:samples) { [0.1, 0.35, 0.7, 0.9] }
  let(:kde) { described_class.new(samples: samples, lower_bound: 0.0, upper_bound: 1.0, bandwidth: 0.1) }

  def reflected_gaussian_density(samples, point, bandwidth)
    samples.sum do |sample|
      (0..200).sum do |image_index|
        positive_center = 2.0 * image_index + sample
        negative_center = 2.0 * image_index - sample
        positive_density = Math.exp(-((point - positive_center) / bandwidth)**2 / 2.0) /
                           (bandwidth * Math.sqrt(2.0 * Math::PI))
        negative_density = Math.exp(-((point - negative_center) / bandwidth)**2 / 2.0) /
                           (bandwidth * Math.sqrt(2.0 * Math::PI))
        positive_density + negative_density
      end
    end / samples.size
  end

  def slow_leave_one_out_log_likelihood(samples, bandwidth, image_count = 200)
    densities = samples.each_with_index.map do |point, excluded_index|
      remaining = samples.dup
      remaining.delete_at(excluded_index)
      reflected_gaussian_density(remaining, point, bandwidth)
    end

    densities.sum { |density| Math.log(density) } / samples.size
  end

  describe '#density' do
    it 'matches the reflected Gaussian image sum inside the interval' do
      [0.0, 0.2, 0.5, 0.8, 1.0].each do |point|
        expected = reflected_gaussian_density(samples, point, 0.1)

        expect(kde.density(point)).to be_within(1e-12).of(expected)
      end
    end

    it 'returns zero outside the support interval' do
      expect(kde.density(-0.1)).to eq 0.0
      expect(kde.density(1.1)).to eq 0.0
    end

    it 'exposes a truncation error bound that is non-negative and conservative' do
      result = kde.density(0.5, error: true)
      expected = reflected_gaussian_density(samples, 0.5, 0.1)

      expect(result[:value]).to be_within(1e-12).of(expected)
      expect(result[:truncation_error_bound]).to be >= 0.0
    end

    it 'integrates to one over the bounded interval' do
      steps = 20_000
      integral = (0..steps).sum do |step|
        point = step.to_f / steps
        weight = step.zero? || step == steps ? 0.5 : 1.0
        weight * kde.density(point) / steps
      end

      expect(integral).to be_within(1e-5).of(1.0)
    end

    it 'applies the interval scaling to non-unit intervals' do
      scaled = described_class.new(
        samples: samples.map { |value| 2.0 + 3.0 * value },
        lower_bound: 2.0,
        upper_bound: 5.0,
        bandwidth: 0.3
      )

      expect(scaled.density(2.6)).to be_within(1e-12).of(kde.density(0.2) / 3.0)
    end
  end

  describe '#cumulative_function' do
    it 'is fixed outside and at the interval boundaries' do
      expect(kde.cumulative_function(-1.0)).to eq 0.0
      expect(kde.cumulative_function(0.0)).to eq 0.0
      expect(kde.cumulative_function(1.0)).to eq 1.0
      expect(kde.cumulative_function(2.0)).to eq 1.0
    end

    it 'matches a numerical density integral and stays ordered' do
      steps = 20_000
      values = (0..steps).map { |step| kde.cumulative_function(step.to_f / steps) }

      expect(values.each_cons(2).all? { |left, right| right >= left }).to be true
      expect(kde.cumulative_function(0.7)).to be_within(1e-4).of(
        (1...(0.7 * steps).to_i).sum { |step| kde.density(step.to_f / steps) / steps }
      )
    end

    it 'returns a zero error bound at exactly fixed boundary values' do
      expect(kde.cumulative_function(-1, error: true)).to eq(value: 0.0, truncation_error_bound: 0.0)
      expect(kde.cumulative_function(1, error: true)).to eq(value: 1.0, truncation_error_bound: 0.0)
    end
  end

  describe '#interval_probability' do
    it 'equals one for the complete support interval' do
      expect(kde.interval_probability(0.0, 1.0)).to eq 1.0
    end

    it 'clamps requested intervals to support and agrees with the CDF' do
      expect(kde.interval_probability(-2.0, 0.4)).to be_within(1e-12).of(kde.cumulative_function(0.4))
      expect(kde.interval_probability(0.4, 3.0)).to be_within(1e-12).of(1.0 - kde.cumulative_function(0.4))
      expect(kde.interval_probability(1.0, 2.0)).to eq 0.0
    end

    it 'reports a looser but valid bound with fewer images' do
      precise = described_class.new(samples: samples, lower_bound: 0.0, upper_bound: 1.0,
                                    bandwidth: 0.1, mirror_tolerance: 1e-14)
      loose = described_class.new(samples: samples, lower_bound: 0.0, upper_bound: 1.0,
                                  bandwidth: 0.1, mirror_tolerance: 3.0, max_mirror_terms: 0)
      loose_result = loose.interval_probability(0.1, 0.8, error: true)

      expect(loose_result[:value]).to be_within(loose_result[:truncation_error_bound]).of(
        precise.interval_probability(0.1, 0.8)
      )
      expect(loose_result[:truncation_error_bound]).to be > 0.0
    end

    it 'rejects reversed intervals' do
      expect { kde.interval_probability(0.8, 0.1) }.to raise_error(ArgumentError)
    end
  end

  describe 'bandwidth selection' do
    it 'validates the samples and bandwidth bounds' do
      expect do
        described_class.new(samples: [0.1, 1.1], lower_bound: 0.0, upper_bound: 1.0, bandwidth: 0.1)
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(samples: samples, lower_bound: 0.0, upper_bound: 1.0,
                            bandwidth_bounds: [0.2, 0.1])
      end.to raise_error(ArgumentError)
    end

    it 'optimizes leave-one-out likelihood and checks it with a slow image sum' do
      fit_samples = [0.35, 0.45, 0.55, 0.65]
      fitted = described_class.fit(
        samples: fit_samples,
        lower_bound: 0.0,
        upper_bound: 1.0,
        bandwidth_bounds: [0.02, 0.3],
        grid_size: 28,
        mirror_tolerance: 1e-13
      )
      fast_result = fitted.leave_one_out_log_likelihood(fitted.bandwidth, error: true)
      slow_result = slow_leave_one_out_log_likelihood(fit_samples, fitted.bandwidth)

      expect(fitted.bandwidth).to be > 0.02
      expect(fitted.bandwidth).to be < 0.3
      expect(fitted.fit_report[:global_maximum_guaranteed]).to be false
      expect(fitted.fit_report[:unique_maximum_guaranteed]).to be false
      expect(fast_result[:log_likelihood]).to be_within(1e-10).of(slow_result)
      expect(fast_result[:log_likelihood_error_bound]).to be >= 0.0
    end

    it 'removes the held-out observation together with all of its images' do
      duplicate_samples = [0.4, 0.5, 0.5, 0.6]
      fitted = described_class.new(
        samples: duplicate_samples,
        lower_bound: 0.0,
        upper_bound: 1.0,
        bandwidth_bounds: [0.04, 0.2],
        grid_size: 16
      )
      result = fitted.leave_one_out_log_likelihood(fitted.bandwidth, error: true)
      slow_result = slow_leave_one_out_log_likelihood(duplicate_samples, fitted.bandwidth)

      expect(result[:log_likelihood]).to be_within(1e-10).of(slow_result)
    end

    it 'reports repeated values, endpoint observations, and bandwidth near a search bound' do
      endpoint_samples = [0.0, 0.25, 0.25, 1.0]
      fitted = described_class.fit(
        samples: endpoint_samples,
        lower_bound: 0.0,
        upper_bound: 1.0,
        bandwidth_bounds: [0.05, 0.5],
        grid_size: 9
      )

      expect(fitted.fit_report[:warnings]).to include(
        :duplicate_observations,
        :observations_at_lower_bound,
        :observations_at_upper_bound,
        :bandwidth_near_upper_search_bound
      )
      expect(fitted.fit_report[:duplicate_observations]).to eq(0.25 => 2)
    end

    it 'requires at least two samples for leave-one-out selection' do
      expect do
        described_class.fit(samples: [0.2], lower_bound: 0.0, upper_bound: 1.0,
                            bandwidth_bounds: [0.05, 0.5])
      end.to raise_error(ArgumentError)
    end
  end

  describe 'error propagation' do
    it 'propagates omitted image density error through leave-one-out log density' do
      fitted = described_class.new(
        samples: [0.2, 0.5, 0.8],
        lower_bound: 0.0,
        upper_bound: 1.0,
        bandwidth_bounds: [0.08, 0.3],
        mirror_tolerance: 1e-8,
        grid_size: 8
      )
      result = fitted.leave_one_out_log_likelihood(0.1, error: true)
      precise = described_class.new(
        samples: [0.2, 0.5, 0.8],
        lower_bound: 0.0,
        upper_bound: 1.0,
        bandwidth: 0.1,
        mirror_tolerance: 1e-14
      ).leave_one_out_log_likelihood(0.1)

      expect((result[:log_likelihood] - precise).abs).to be <= result[:log_likelihood_error_bound]
      expect(fitted.fit_report[:maximum_search_log_likelihood_error_bound]).to be > 0.0
    end

    it 'raises instead of silently exceeding the requested image-count limit' do
      expect do
        described_class.new(
          samples: [0.2, 0.8],
          lower_bound: 0.0,
          upper_bound: 1.0,
          bandwidth: 100_000.0,
          mirror_tolerance: 1e-12,
          max_mirror_terms: 10
        ).density(0.5)
      end.to raise_error(RuntimeError, /mirrored-tail error/)
    end
  end
end
