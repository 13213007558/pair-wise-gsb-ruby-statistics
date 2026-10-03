# frozen_string_literal: true

require 'spec_helper'

describe RubyStatistics::Distribution::GeneralizedPareto do
  let(:positive_shape_excesses) do
    [0.0253, 0.0526, 0.0845, 0.1278, 0.2007,
     0.3235, 0.5108, 0.9163, 1.6094, 3.0]
  end

  let(:near_zero_shape_excesses) do
    [0.0500, 0.1054, 0.1625, 0.2231, 0.2877,
     0.3567, 0.4308, 0.5108, 0.5978, 0.6931]
  end

  let(:negative_shape_excesses) do
    (1..20).map do |index|
      probability = 1.0 - (index.to_f / 41.0)
      (1.0 - probability**0.2) / 0.2
    end
  end

  describe '.fit' do
    it 'fits a positive shape and reports a profile-likelihood return-level interval' do
      result = described_class.fit(
        excesses: positive_shape_excesses,
        threshold: 10.0,
        exceedance_probability: 0.2,
        return_period: 20.0
      )

      expect(result[:status]).to eq(:ok)
      expect(result[:shape]).to be_within(0.02).of(0.73)
      expect(result[:scale]).to be_within(0.02).of(0.31)
      expect(result[:termination_reason]).to eq(:relative_convergence)
      expect(result[:diagnostics][:support_satisfied]).to be(true)
      expect(result[:return_level]).to be_within(0.01).of(10.74)
      expect(result[:confidence_interval][:status]).to eq(:converged)
      expect(result[:confidence_interval][:lower]).to be < result[:return_level]
      expect(result[:confidence_interval][:upper]).to be > result[:return_level]

      result[:return_level_diagnostics].values_at(:lower_endpoint, :upper_endpoint).each do |endpoint|
        expect(endpoint[:likelihood_drop]).to be_within(0.01).of(3.8414588)
        expect(endpoint[:termination_reason]).to eq(:profile_root_relative_convergence)
      end
    end

    it 'uses the continuous exponential limit for a near-zero shape' do
      result = described_class.fit(
        excesses: near_zero_shape_excesses,
        threshold: 10.0,
        exceedance_probability: 0.2,
        return_period: 20.0
      )

      expect(result[:status]).to eq(:ok)
      expect(result[:shape].abs).to be < 1.0e-3
      expect(result[:scale]).to be_within(0.01).of(near_zero_shape_excesses.sum / 10.0)
      expect(result[:return_level]).to be_within(0.02).of(10.47)
      expect(result[:confidence_interval][:status]).to eq(:converged)

      result[:return_level_diagnostics].values_at(:lower_endpoint, :upper_endpoint).each do |endpoint|
        expect(endpoint[:likelihood_drop]).to be_within(0.01).of(3.8414588)
      end
    end

    it 'optimizes the remaining parameters at a fixed target for a negative shape' do
      target = described_class.quantile_for(-0.2, 1.0, 4.0)
      result = described_class.profile_return_level(
        excesses: negative_shape_excesses,
        target: target,
        shape: -0.2,
        scale: 1.0,
        shape_bounds: [-0.201, -0.199]
      )

      expect(result[:status]).to eq(:ok)
      expect(result[:shape]).to be_within(0.05).of(-0.2)
      expect(result[:scale]).to be_within(0.02).of(1.0)
      expect(result[:diagnostics][:actual_target]).to be_within(1.0e-12).of(target)
      expect(result[:diagnostics][:support_satisfied]).to be(true)
      expect(result[:termination_reason]).to eq(:relative_convergence)
    end

    it 'falls back to the exponential-limit fit for tied bounded-tail observations' do
      result = described_class.fit(
        excesses: Array.new(5, 0.5),
        threshold: 10.0,
        exceedance_probability: 0.2,
        return_period: 20.0
      )

      expect(result[:status]).to eq(:ok)
      expect(result[:shape].abs).to be < 1.0e-3
      expect(result[:confidence_interval][:status]).to eq(:converged)
    end

    it 'reports insufficient data when there are fewer than two positive excesses' do
      result = described_class.fit(
        excesses: [0.5],
        threshold: 10.0,
        exceedance_probability: 0.2
      )

      expect(result[:status]).to eq(:data_insufficient)
      expect(result[:termination_reason]).to eq(:need_at_least_two_positive_excesses)
    end

    it 'treats zero excess as insufficient tail data' do
      result = described_class.fit(
        excesses: [0.0, 0.5],
        threshold: 10.0,
        exceedance_probability: 0.2
      )

      expect(result[:status]).to eq(:data_insufficient)
      expect(result[:termination_reason]).to eq(:excess_must_be_strictly_positive)
    end

    it 'rejects excesses outside the non-negative support before optimization' do
      result = described_class.fit(
        excesses: [0.1, -0.2],
        threshold: 10.0,
        exceedance_probability: 0.2
      )

      expect(result[:status]).to eq(:data_insufficient)
      expect(result[:termination_reason]).to eq(:non_numeric_or_negative_excess)
    end

    it 'rejects an effective return period shorter than the threshold exceedance interval' do
      result = described_class.fit(
        excesses: positive_shape_excesses,
        threshold: 10.0,
        exceedance_probability: 0.1,
        return_period: 5.0
      )

      expect(result[:status]).to eq(:data_insufficient)
      expect(result[:termination_reason]).to eq(:return_period_must_exceed_inverse_exceedance_probability)
    end

    it 'does not report convergence when a fixed-target profile starts outside support' do
      target = described_class.quantile_for(-0.2, 1.0, 4.0)
      result = described_class.profile_return_level(
        excesses: negative_shape_excesses,
        target: target,
        shape: -0.2,
        scale: 0.01
      )

      expect(result[:status]).to eq(:no_finite_optimum)
      expect(result[:termination_reason]).to eq(:profile_start_point_outside_support)
    end
  end

  describe '#density_function and #cumulative_function' do
    it 'keeps scale positive and evaluates the negative-shape closed support' do
      distribution = described_class.new(shape: -0.2, scale: 2.0)

      expect(distribution.scale).to be_positive
      expect(distribution.density_function(0.0)).to be_within(1.0e-12).of(0.5)
      expect(distribution.density_function(5.0)).to be_within(1.0e-12).of(0.03125)
      expect(distribution.density_function(12.0)).to be_zero
      expect(distribution.cumulative_function(5.0)).to be_within(1.0e-12).of(0.96875)
      expect(distribution.cumulative_function(10.0)).to eq(1.0)
      expect(distribution.support_endpoint).to eq(10.0)
    end

    it 'uses the exponential formula continuously at shape zero' do
      distribution = described_class.new(shape: 0.0, scale: 2.0)

      expect(distribution.density_function(2.0)).to be_within(1.0e-12).of(Math.exp(-1.0) / 2.0)
      expect(distribution.cumulative_function(2.0)).to be_within(1.0e-12).of(1.0 - Math.exp(-1.0))
      expect(distribution.support_endpoint).to eq(Float::INFINITY)
    end
  end
end
