# frozen_string_literal: true

require 'spec_helper'
require 'bigdecimal'

describe RubyStatistics::ConfidenceSequence::BernoulliConfidenceSequence do
  let(:alpha) { 0.05 }

  def factorial(number)
    (1..number).reduce(1, :*)
  end

  # High precision root of the uniform Beta(1, 1) mixture equation:
  #
  #   s! f! / (n + 1)! * p**(-s) * (1-p)**(-f) = 1 / alpha
  #
  # evaluated entirely with BigDecimal (no Float, no transcendental functions)
  # so small-count endpoints can be checked against an independent formula.
  def big_decimal_bounds(successes, failures, alpha, precision)
    BigDecimal.limit(precision)
    n = successes + failures
    # Build the rational constant exactly: (n + 1)! / (s! f! alpha).
    threshold = (BigDecimal(factorial(n + 1)) /
                 (BigDecimal(factorial(successes)) * BigDecimal(factorial(failures)) *
                  BigDecimal(alpha.to_s)))
    one = BigDecimal('1')

    power = lambda do |base, exponent|
      if exponent.zero?
        one
      elsif exponent.negative?
        one / (base ** exponent.abs)
      else
        base ** exponent
      end
    end

    residual = lambda do |point|
      s_power = power.call(point, -successes)
      f_power = power.call(one - point, -failures)
      s_power * f_power - threshold
    end

    # On the left of the mean the residual goes + (boundary 0) -> - (mean);
    # on the right it goes - (mean) -> + (boundary 1). Bracket accordingly.
    bisect_left = lambda do |low, high|
      200.times do
        middle = (low + high) / 2
        break middle if middle == low || middle == high

        if residual.call(middle) > 0
          low = middle
        else
          high = middle
        end
      end
      (low + high) / 2
    end

    bisect_right = lambda do |low, high|
      200.times do
        middle = (low + high) / 2
        break middle if middle == low || middle == high

        if residual.call(middle) > 0
          high = middle
        else
          low = middle
        end
      end
      (low + high) / 2
    end

    lower = successes.zero? ? BigDecimal('0') : bisect_left.call(BigDecimal('0'), BigDecimal(successes) / n)
    upper = failures.zero? ? BigDecimal('1') : bisect_right.call(BigDecimal(successes) / n, BigDecimal('1'))
    [lower, upper]
  end

  describe '#initialize' do
    it 'rejects alpha outside the open unit interval' do
      expect { described_class.new(alpha: 0) }.to raise_error(ArgumentError)
      expect { described_class.new(alpha: 1) }.to raise_error(ArgumentError)
      expect { described_class.new(alpha: -0.1) }.to raise_error(ArgumentError)
      expect { described_class.new(alpha: 1.1) }.to raise_error(ArgumentError)
    end

    it 'rejects invalid Beta mixtures' do
      expect { described_class.new(alpha: alpha, prior: []) }.to raise_error(ArgumentError)
      expect { described_class.new(alpha: alpha, prior: 'nope') }.to raise_error(ArgumentError)
      expect do
        described_class.new(alpha: alpha, prior: [{ alpha_prior: 0, beta_prior: 1 }])
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(alpha: alpha, prior: [{ alpha_prior: 1, beta_prior: 0 }])
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(alpha: alpha, prior: [{ alpha_prior: 1, beta_prior: 1, weight: -0.5 }])
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(alpha: alpha,
                            prior: [{ alpha_prior: 1, beta_prior: 1, weight: 0 }])
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(alpha: alpha, prior: [{ alpha_prior: 'x', beta_prior: 1 }])
      end.to raise_error(ArgumentError)
      expect { described_class.new(alpha: alpha, prior: [nil]) }.to raise_error(ArgumentError)
    end

    it 'normalizes mixture weights exactly once at construction' do
      cs = described_class.new(alpha: alpha,
                               prior: [
                                 { alpha_prior: 1, beta_prior: 1, weight: 3 },
                                 { alpha_prior: 2, beta_prior: 2, weight: 1 }
                               ])
      weights = cs.components.map(&:weight)
      expect(weights).to eq [0.75, 0.25]
      expect(weights.sum).to eq 1.0
    end
  end

  describe 'configuration' do
    it 'exposes alpha and the pre-specified prior through #config' do
      cs = described_class.new(alpha: alpha)

      expect(cs.config).to eq(
        alpha: 0.05,
        prior: [{ weight: 1.0, alpha_prior: 1.0, beta_prior: 1.0 }]
      )
    end

    it 'never changes after data is observed' do
      cs = described_class.new(alpha: alpha)
      config_before = cs.config

      25.times { cs.success! }
      17.times { cs.failure! }
      cs.result

      expect(cs.config).to eq config_before
      expect(cs.alpha).to eq 0.05
      expect(cs.components).to be_frozen
      expect { cs.components.first.weight = 2.0 }.to raise_error(FrozenError)
    end
  end

  describe '#observe!' do
    it 'updates only the necessary counts and returns the interval per refresh' do
      cs = described_class.new(alpha: alpha)

      expect(cs.successes).to eq 0
      expect(cs.failures).to eq 0
      expect(cs.sample_count).to eq 0
      expect(cs.result.to_h.slice(:lower, :upper, :sample_count)).to eq(
        lower: 0.0, upper: 1.0, sample_count: 0
      )

      cs.observe!(1)
      expect(cs.successes).to eq 1
      expect(cs.failures).to eq 0

      cs.observe!(0)
      expect(cs.successes).to eq 1
      expect(cs.failures).to eq 1
      expect(cs.sample_count).to eq 2

      cs.success!(3)
      cs.failure!(2)
      expect(cs.successes).to eq 4
      expect(cs.failures).to eq 3
      expect(cs.sample_count).to eq 7
      expect(cs.observe!(1)).to be(cs)
    end

    it 'rejects outcomes that are not success/failure' do
      cs = described_class.new(alpha: alpha)
      expect { cs.observe!(2) }.to raise_error(ArgumentError)
      expect { cs.observe!(-1) }.to raise_error(ArgumentError)
      expect { cs.observe!(1, -1) }.to raise_error(ArgumentError)
    end
  end

  describe '#log_e_process' do
    it 'is negative infinity for the empty prefix' do
      cs = described_class.new(alpha: alpha)
      expect(cs.log_e_process(0.5)).to eq(-Float::INFINITY)
    end

    it 'handles the boundary points p = 0 and p = 1 explicitly' do
      cs = described_class.new(alpha: alpha)
      3.times { cs.success! }
      2.times { cs.failure! }

      expect(cs.log_e_process(0.0)).to eq Float::INFINITY
      expect(cs.log_e_process(1.0)).to eq Float::INFINITY
      expect { cs.log_e_process(-0.1) }.to raise_error(ArgumentError)
      expect { cs.log_e_process(1.1) }.to raise_error(ArgumentError)
    end

    it 'stays finite at p = 0 with only failures and at p = 1 with only successes' do
      failures_only = described_class.new(alpha: alpha)
      4.times { failures_only.failure! }
      expect(failures_only.log_e_process(0.0)).to be_within(1e-12).of(Math.log(1.0 / 5))
      expect(failures_only.log_e_process(1.0)).to eq Float::INFINITY

      successes_only = described_class.new(alpha: alpha)
      4.times { successes_only.success! }
      expect(successes_only.log_e_process(1.0)).to be_within(1e-12).of(Math.log(1.0 / 5))
      expect(successes_only.log_e_process(0.0)).to eq Float::INFINITY
    end
  end

  describe '#bounds' do
    it 'contains the empirical mean and keeps bounds ordered within [0, 1]' do
      cs = described_class.new(alpha: alpha)

      [[0, 1], [1, 0], [1, 1], [2, 1], [10, 7], [100, 37], [500, 500]].each do |successes, failures|
        lower, upper = cs.bounds(successes: successes, failures: failures)
        mean = successes.to_f / (successes + failures)

        expect(0.0..1.0).to cover(lower)
        expect(0.0..1.0).to cover(upper)
        expect(lower).to be <= mean
        expect(mean).to be <= upper
        expect(lower).to be <= upper
      end
    end

    it 'uses the all-success / all-failure closed form' do
      cs = described_class.new(alpha: alpha)

      (1..12).each do |number|
        lower, upper = cs.bounds(successes: number, failures: 0)
        expect(lower).to be_within(1e-12).of((alpha / (number + 1)) ** (1.0 / number))
        expect(upper).to eq 1.0

        lower, upper = cs.bounds(successes: 0, failures: number)
        expect(lower).to eq 0.0
        expect(upper).to be_within(1e-12).of(1.0 - (alpha / (number + 1)) ** (1.0 / number))
      end
    end

    it 'produces the exact threshold at both interior endpoints' do
      cs = described_class.new(alpha: alpha)

      [[3, 2], [10, 7], [50, 13], [200, 140]].each do |successes, failures|
        lower, upper = cs.bounds(successes: successes, failures: failures)

        expect(cs.log_e_process(lower, successes: successes, failures: failures))
          .to be_within(1e-9).of(Math.log(1.0 / alpha))
        expect(cs.log_e_process(upper, successes: successes, failures: failures))
          .to be_within(1e-9).of(Math.log(1.0 / alpha))
      end
    end

    it 'matches the high precision direct formula for small counts' do
      cs = described_class.new(alpha: alpha)

      [[1, 1], [2, 1], [1, 2], [3, 2], [4, 4], [5, 2], [2, 5], [8, 3], [3, 8], [12, 7]].each do |successes, failures|
        lower, upper = cs.bounds(successes: successes, failures: failures)
        precise_lower, precise_upper = big_decimal_bounds(successes, failures, alpha, 60)

        expect((lower - precise_lower.to_f).abs).to be <= 1e-11
        expect((upper - precise_upper.to_f).abs).to be <= 1e-11
      end
    end

    it 'supports Beta mixtures and stays numerically stable for very long runs' do
      cs = described_class.new(alpha: alpha,
                               prior: [
                                 { alpha_prior: 1, beta_prior: 1, weight: 1 },
                                 { alpha_prior: 0.5, beta_prior: 0.5, weight: 3 }
                               ])
      10.times { cs.success! }
      6.times { cs.failure! }

      lower, upper = cs.bounds
      expect(lower).to be_between(0.0, 1.0)
      expect(upper).to be_between(0.0, 1.0)
      expect(lower).to be < 10.0 / 16
      expect(upper).to be > 10.0 / 16

      long_run = described_class.new(alpha: alpha)
      long_run.success!(10_000)
      lower, upper = long_run.bounds
      expected = (alpha / 10_001.0) ** (1.0 / 10_000)
      expect(lower).to be_within(1e-9).of(expected)
      expect(upper).to eq 1.0
      expect(long_run.bounds.all?(&:finite?)).to be true
    end
  end

  describe '.interval' do
    it 'refreshes the interval for each prefix instead of reusing a fixed one' do
      sequence = [1, 1, 0, 1, 0, 1, 1, 0]
      cs = described_class.new(alpha: alpha)
      previous_upper = 1.0
      refreshed = 0

      sequence.each do |outcome|
        cs.observe!(outcome)
        lower, upper = cs.bounds
        snapshot = described_class.interval(successes: cs.successes,
                                            failures: cs.failures,
                                            alpha: alpha)
        expect(snapshot.lower).to be_within(1e-15).of(lower)
        expect(snapshot.upper).to be_within(1e-15).of(upper)
        expect(snapshot.sample_count).to eq cs.sample_count
        expect(upper).to be < previous_upper if [cs.successes, cs.failures].min > 0
        refreshed += 1
      end

      expect(refreshed).to eq sequence.size
    end
  end

  describe 'pathwise coverage (fixed seed, supplementary evidence)' do
    # The guarantee is mathematical (Ville + the supermartingale), and unit
    # tests of endpoints against the exact formula are the correctness check.
    # This simulation is only supplementary evidence; thresholds are loose so
    # a fixed seed cannot make the suite fail at random.
    it 'covers the true probability simultaneously along whole paths' do
      cs = described_class.new(alpha: alpha)
      horizons = 200
      paths_per_probability = 300
      probabilities = [0.2, 0.5, 0.8]

      # Intervals depend only on (successes, failures), so each pair is solved
      # once regardless of how many simulated paths visit it.
      interval_cache = Hash.new do |cache, pair|
        cache[pair] = cs.bounds(successes: pair.first, failures: pair.last)
      end

      random = Random.new(20_261_003)
      violations = 0
      final_contained = 0
      final_widths = []

      probabilities.each do |probability|
        paths_per_probability.times do
          successes = 0
          failures = 0
          path_violated = false

          horizons.times do
            if random.rand < probability
              successes += 1
            else
              failures += 1
            end

            lower, upper = interval_cache[[successes, failures]]
            path_violated ||= probability < lower || probability > upper
          end

          lower, upper = interval_cache[[successes, failures]]
          violations += 1 if path_violated
          final_contained += 1 if lower <= probability && probability <= upper
          final_widths << upper - lower
        end
      end

      total_paths = probabilities.size * paths_per_probability

      # Ville promises at most alpha = 5% of paths ever break their sequence.
      expect(violations.to_f / total_paths).to be <= 0.12
      expect(final_contained.to_f / total_paths).to be >= 0.92
      expect(final_widths.sum / final_widths.size).to be <= 0.35
    end

    it 'never violates at the boundary probabilities 0 and 1' do
      cs = described_class.new(alpha: alpha)

      all_failures = described_class.new(alpha: alpha)
      500.times { all_failures.failure! }
      lower, upper = all_failures.bounds
      expect(lower).to eq 0.0
      expect(0.0 <= upper).to be true

      all_successes = described_class.new(alpha: alpha)
      500.times { all_successes.success! }
      lower, upper = all_successes.bounds
      expect(lower <= 1.0).to be true
      expect(upper).to eq 1.0
    end
  end
end
