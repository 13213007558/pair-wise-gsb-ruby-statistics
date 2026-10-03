# frozen_string_literal: true

require 'spec_helper'

describe RubyStatistics::Distribution::GeneralizedPareto do
  let(:threshold) { 10.0 }
  let(:exceedance_proportion) { 0.1 }
  let(:return_period) { 20.0 }

  let(:positive_excesses) do
    [
      0.21550666648593264, 5.643887639504047, 6.747949879445038,
      0.7862534214197452, 0.42631347197069, 0.17497544477002602,
      1.6445818943461314, 0.13194725497972692, 0.7874081481665383,
      1.3002119985027862, 2.690995057140558, 1.552369819069816,
      0.8243662613362126, 1.1113455788108486, 1.777987045603501,
      0.3225969831814637, 0.9689388489702639, 0.597857957574377,
      0.20458080158275704, 9.47690540695166
    ]
  end

  let(:near_zero_excesses) do
    [
      0.32922833003897795, 4.294145443917945, 1.8268313614813427,
      1.0018752314971535, 3.070422326110317, 1.7947665216822926,
      0.8456434155960696, 0.6852961968585974, 0.5943787893700562,
      1.511605708242085, 1.6498468234507369, 0.7583983920763694,
      1.4422888702217025, 0.222320101945887, 0.33225146929374666,
      0.2403369810327357, 0.16813288110933985, 0.15263611802073643,
      0.043810641301531235, 0.2681118259518856
    ]
  end

  let(:negative_excesses) do
    [
      0.05596309086002016, 1.1786056605392061, 0.4046779599097734,
      1.713259693530512, 0.09193400296279775, 0.44659935008563334,
      0.3926892558251238, 0.5670143013391404
    ]
  end

  describe '.fit' do
    it 'fits a positive-shape excess tail with an interior MLE' do
      result = described_class.fit(
        excesses: positive_excesses,
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.converged).to eq(true)
      expect(result.status).to eq(:converged)
      expect(result.shape).to be_within(0.03).of(0.367)
      expect(result.scale).to be_within(0.05).of(1.226)
      expect(result.scale).to be_positive
      expect(result.negative_log_likelihood).to be_within(0.02).of(31.4214)
      expect(result.log_likelihood).to eq(-result.negative_log_likelihood)
      expect(result.termination_reason).to eq(:bounded_profile_mle)
    end

    it 'uses the continuous exponential limit near zero shape' do
      result = described_class.fit(
        excesses: near_zero_excesses,
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.converged).to eq(true)
      expect(result.shape.abs).to be < 0.001
      expect(result.scale).to be_within(0.01).of(1.0616)
      expect(result.scale).to be_positive
      expect(result.negative_log_likelihood).to be_within(0.02).of(21.1959)
    end

    it 'fits a negative-shape excess tail without crossing support' do
      result = described_class.fit(
        excesses: negative_excesses,
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.converged).to eq(true)
      expect(result.shape).to be_within(0.04).of(-0.29)
      expect(result.shape).to be > -1.0
      expect(result.scale).to be_within(0.05).of(0.804)
      expect(result.scale).to be_positive
      expect(result.negative_log_likelihood).to be_within(0.02).of(3.8480)
      expect(result.termination_reason).to eq(:bounded_profile_mle)

      negative_excesses.each do |excess|
        expect(1.0 + result.shape * excess / result.scale).to be_positive
      end
    end

    it 'does not report convergence when the optimum is at the closed support boundary' do
      result = described_class.fit(
        excesses: [0.2, 0.4, 0.6, 0.8],
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.converged).to eq(false)
      expect(result.status).to eq(:no_finite_optimum)
      expect(result.termination_reason).to eq(:lower_support_boundary)
      expect(result.negative_log_likelihood).to be_within(1.0e-10).of(
        4 * Math.log(0.8)
      )
    end

    it 'reports insufficient data for fewer than two excesses' do
      result = described_class.fit(
        excesses: [1.0],
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.converged).to eq(false)
      expect(result.status).to eq(:data_insufficient)
      expect(result.termination_reason).to eq(:fewer_than_two_excesses)
    end

    it 'rejects zero, negative, or non-finite excesses' do
      result = described_class.fit(
        excesses: [0.0, 1.0],
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )

      expect(result.status).to eq(:invalid_input)
      expect(result.termination_reason).to eq(:excess_must_be_finite_positive)
    end

    it 'rejects an exceedance proportion outside (0, 1]' do
      result = described_class.fit(
        excesses: positive_excesses,
        threshold: threshold,
        exceedance_proportion: 1.5
      )

      expect(result.status).to eq(:invalid_input)
      expect(result.termination_reason).to eq(:exceedance_proportion_out_of_range)
    end
  end

  describe '#return_level_profile_interval' do
    shared_examples 'a profiled return-level interval' do
      it 'finds both endpoints by reoptimizing at fixed return levels' do
        fit = described_class.fit(
          excesses: excesses,
          threshold: threshold,
          exceedance_proportion: exceedance_proportion
        )
        interval = fit.return_level_profile_interval(
          return_period: return_period
        )

        expect(interval.converged).to eq(true)
        expect(interval.status).to eq(:converged)
        expect(interval.lower_return_level).to be < interval.return_level
        expect(interval.return_level).to be < interval.upper_return_level
        expect(interval.critical_value).to be_within(1.0e-8).of(3.841458826883408)
        expect(interval.target_negative_log_likelihood).to be_within(1.0e-10).of(
          fit.negative_log_likelihood + interval.critical_value / 2.0
        )
        expect(interval.actual_return_level).to eq(interval.return_level)
        expect(interval.target_log_likelihood).to eq(
          -interval.target_negative_log_likelihood
        )
        expect(interval.return_level).to be > threshold
        expect(interval.diagnostics[:exceedance_multiplier]).to eq(
          1.0 / (exceedance_proportion * return_period)
        )

        [interval.lower_endpoint, interval.upper_endpoint].each do |endpoint|
          expect(endpoint.termination_reason).to eq(:target_reached)
          expect(endpoint.deviation.abs).to be < 1.0e-8
          expect(endpoint.negative_log_likelihood).to be_within(1.0e-8).of(
            interval.target_negative_log_likelihood
          )
          expect(endpoint.scale).to be_positive
          excesses.each do |excess|
            expect(1.0 + endpoint.shape * excess / endpoint.scale).to be_positive
          end
        end
      end
    end

    context 'with positive shape' do
      let(:excesses) { positive_excesses }
      it_behaves_like 'a profiled return-level interval'
    end

    context 'with near-zero shape' do
      let(:excesses) { near_zero_excesses }
      it_behaves_like 'a profiled return-level interval'
    end

    context 'with negative shape' do
      let(:excesses) { negative_excesses }
      it_behaves_like 'a profiled return-level interval'
    end

    it 'rejects a return period shorter than 1 / exceedance proportion' do
      fit = described_class.fit(
        excesses: positive_excesses,
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )
      interval = fit.return_level_profile_interval(return_period: 5.0)

      expect(interval.converged).to eq(false)
      expect(interval.status).to eq(:invalid_input)
      expect(interval.termination_reason).to eq(:return_period_too_small)
    end

    it 'reports invalid numeric interval inputs without raising' do
      fit = described_class.fit(
        excesses: positive_excesses,
        threshold: threshold,
        exceedance_proportion: exceedance_proportion
      )
      interval = fit.return_level_profile_interval(
        return_period: 'not-a-number',
        confidence_level: 0.95
      )

      expect(interval.converged).to eq(false)
      expect(interval.status).to eq(:invalid_input)
      expect(interval.termination_reason).to eq(:invalid_numeric_input)
    end
  end

  describe '#density_function and #cumulative_function' do
    it 'uses the exponential limit at zero shape and enforces scale positivity' do
      expect { described_class.new(shape: 0.0, scale: -1.0) }
        .to raise_error(ArgumentError, /scale must be positive/)

      distribution = described_class.new(shape: 0.0, scale: 2.0)
      expect(distribution.density_function(0.0)).to be_within(1.0e-10).of(0.5)
      expect(distribution.cumulative_function(0.0)).to eq(0.0)
      expect(distribution.density_function(-1.0)).to eq(0.0)
    end

    it 'enforces the finite upper support for negative shape' do
      distribution = described_class.new(shape: -0.5, scale: 1.0)
      expect(distribution.density_function(1.9)).to be_positive
      expect(distribution.density_function(2.0)).to eq(0.0)
      expect(distribution.cumulative_function(2.0)).to eq(1.0)
    end
  end
end
