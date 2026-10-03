require 'spec_helper'
require 'bigdecimal'
require 'bigdecimal/math'

describe RubyStatistics::BernoulliConfidenceSequence do
  # High-precision reference computations using BigDecimal. These are direct
  # formulas for small counts: the uniform Beta(1,1) mixture marginal of
  # (S, F) is B(S+1, F+1)/B(1,1) = S! F! / (S+F+1)! and the acceptance set is
  # the inversion of likelihood * (1/alpha) = marginal, i.e. for an endpoint p,
  #
  #   p**S (1-p)**F = alpha * S! F! / (n+1)!
  #
  # Endpoints are then located by high-precision bisection on this equation.
  module HighPrecisionReference
    include BigMath

    PRECISION = 60

    def log_gamma(value)
      BigDecimal(Math.lgamma(Float(value)).first.to_s)
    end

    # log of the uniform mixture marginal S! F! / (n+1)!
    def log_marginal(successes, failures)
      log_gamma(successes + 1) + log_gamma(failures + 1) -
        log_gamma(successes + failures + 2)
    end

    # log of log p**S (1-p)**F, p given as BigDecimal in (0, 1).
    def log_likelihood(p, successes, failures)
      successes * BigMath.log(p, PRECISION) +
        failures * BigMath.log(1 - p, PRECISION)
    end

    def log_alpha
      BigMath.log(BigDecimal('0.05'), PRECISION)
    end

    def root_on(successes, failures, range_low, range_high, side)
      target = log_marginal(successes, failures) + log_alpha
      140.times do
        middle = (range_low + range_high) / 2
        if side == :lower
          if log_likelihood(middle, successes, failures) < target
            range_low = middle
          else
            range_high = middle
          end
        else
          if log_likelihood(middle, successes, failures) < target
            range_high = middle
          else
            range_low = middle
          end
        end
      end
      ((range_low + range_high) / 2).to_f
    end

    def uniform_interval(successes, failures)
      n = successes + failures
      lower = successes.zero? ? 0.0 : root_on(
        successes, failures,
        BigDecimal('1e-60'),
        BigDecimal(successes.to_r.fdiv(n).to_s, PRECISION),
        :lower
      )
      upper = failures.zero? ? 1.0 : root_on(
        successes, failures,
        BigDecimal(successes.to_r.fdiv(n).to_s, PRECISION),
        1 - BigDecimal('1e-60'),
        :upper
      )
      [lower, upper]
    end
  end

  include HighPrecisionReference

  describe 'configuration' do
    it 'uses the uniform Beta(1,1) mixture at alpha 0.05 by default' do
      cs = described_class.new

      expect(cs.alpha).to eq 0.05
      expect(cs.mixture_prior).to eq([{ weight: 1.0, alpha: 1.0, beta: 1.0 }])
    end

    it 'accepts a pre-specified multi-component Beta mixture and normalizes weights' do
      cs = described_class.new(
        alpha: 0.1,
        mixture_prior: [
          { weight: 1, alpha: 0.5, beta: 0.5 },
          { weight: 3, alpha: 1, beta: 1 }
        ]
      )

      expect(cs.mixture_prior.map { |c| c[:weight] }).to eq [0.25, 0.75]
      expect(cs.alpha).to eq 0.1
    end

    it 'rejects invalid alpha or mixture components' do
      expect { described_class.new(alpha: 0) }.to raise_error(ArgumentError)
      expect { described_class.new(alpha: 1) }.to raise_error(ArgumentError)
      expect { described_class.new(mixture_prior: []) }.to raise_error(ArgumentError)
      expect do
        described_class.new(mixture_prior: [{ weight: 0, alpha: 1, beta: 1 }])
      end.to raise_error(ArgumentError)
      expect do
        described_class.new(mixture_prior: [{ weight: 1, alpha: 0, beta: 1 }])
      end.to raise_error(ArgumentError)
    end

    it 'freezes the configuration so it cannot silently change after data arrive' do
      cs = described_class.new
      cs.observe(1)

      expect(cs.mixture_prior).to be_frozen
      expect { cs.mixture_prior[0][:alpha] = 5 }.to raise_error(FrozenError)
      expect { cs.mixture_prior << { weight: 1, alpha: 1, beta: 1 } }
        .to raise_error(FrozenError)
      expect(cs).not_to respond_to(:alpha=)
      expect(cs).not_to respond_to(:mixture_prior=)
    end
  end

  describe '#observe / #update' do
    it 'stores only the running success and failure counts, no raw history' do
      cs = described_class.new
      cs.observe(1).observe(true)
      cs.observe(0)
      cs.update(false)
      cs.observe([1, 0])

      expect(cs.successes).to eq 3
      expect(cs.failures).to eq 3
      expect(cs.n).to eq 6
      expect(cs.count).to eq 6
      expect(cs.observed?).to be true
      instance_variables = cs.instance_variables.map(&:to_s)
      expect(instance_variables.grep(/sample|observation|history|data|sequence/i)).to be_empty
    end

    it 'rejects observations outside the Bernoulli support and empty batches' do
      cs = described_class.new
      expect { cs.observe(2) }.to raise_error(ArgumentError)
      expect { cs.observe(nil) }.to raise_error(ArgumentError)
      expect { cs.observe([]) }.to raise_error(ArgumentError)
    end

    it 'refreshes the interval after every new success or failure rather than reusing a fixed one' do
      cs = described_class.new
      previous = cs.interval
      expect(previous).to eq [0.0, 1.0]

      cs.observe(1)
      after_success = cs.interval
      expect(after_success).not_to eq previous

      cs.observe(0)
      after_failure = cs.interval
      expect(after_failure).not_to eq after_success
    end
  end

  describe '#interval with the uniform mixture against direct formulas' do
    endpoint_cases = [
      [1, 0], [2, 0], [3, 0], [5, 0],
      [0, 1], [0, 2], [0, 3], [0, 5],
      [1, 1], [1, 2], [2, 1], [2, 2], [3, 2], [2, 5], [4, 4]
    ]

    endpoint_cases.each do |successes, failures|
      it "matches the high-precision direct formula for S=#{successes}, F=#{failures}" do
        cs = described_class.new
        cs.observe(Array.new(successes, 1) + Array.new(failures, 0))

        lower, upper = cs.interval
        exact_lower, exact_upper = uniform_interval(successes, failures)

        if exact_lower.zero?
          expect(lower).to eq 0.0
        else
          expect((lower - exact_lower).abs).to be < 1e-12
        end
        if exact_upper == 1.0
          expect(upper).to eq 1.0
        else
          expect((upper - exact_upper).abs).to be < 1e-12
        end
      end
    end

    it 'gives the exact closed-form endpoints for all-success prefixes' do
      cs = described_class.new
      cs.observe(1)
      lower, upper = cs.interval
      expect(lower).to be_within(1e-15).of(0.025)
      expect(upper).to eq 1.0

      cs.observe(1)
      expect(cs.interval.first).to be_within(1e-15).of(Math.sqrt(0.05 / 3.0))

      cs.observe(1)
      expect(cs.interval.first).to be_within(1e-15).of((0.05 / 4.0)**(1.0 / 3.0))
    end

    it 'gives the exact closed-form endpoints for all-failure prefixes' do
      cs = described_class.new
      cs.observe(0)
      lower, upper = cs.interval
      expect(lower).to eq 0.0
      expect(upper).to be_within(1e-15).of(0.975)

      cs.observe(0)
      expect(cs.interval.last).to be_within(1e-15).of(1 - Math.sqrt(0.05 / 3.0))

      cs.observe(0)
      expect(cs.interval.last).to be_within(1e-15).of(1 - (0.05 / 4.0)**(1.0 / 3.0))
    end

    it 'handles p = 0 and p = 1 without NaN or infinities in the interval' do
      cs = described_class.new
      cs.observe([1, 1, 0, 1, 0])

      [cs.interval, cs.result.interval].each do |(lower, upper)|
        expect(lower).to be_finite
        expect(upper).to be_finite
        expect(0.0 <= lower).to be true
        expect(upper <= 1.0).to be true
      end
      expect(cs.accept?(0.0)).to be false
      expect(cs.accept?(1.0)).to be false
      expect(cs.log_supermartingale(0.0)).to eq Float::INFINITY
      expect(cs.log_supermartingale(1.0)).to eq Float::INFINITY
    end

    it 'always accepts the observed proportion and rejects points outside the endpoints' do
      prng = Random.new(777)
      cs = described_class.new
      30.times { cs.observe(prng.rand < 0.4 ? 1 : 0) }

      lower, upper = cs.interval
      expect(cs.accept?(cs.successes.to_r / cs.n)).to be true
      expect(cs.accept?(lower)).to be true
      expect(cs.accept?(upper)).to be true
      expect(cs.accept?([lower / 2, 0.0].max)).to eq(cs.accept?(0.0))
      expect(cs.accept?([(upper + 1.0) / 2, 1.0].min)).to eq(cs.accept?(1.0))
      expect(lower).to be < upper
    end
  end

  describe '#result' do
    it 'returns the interval, sample size and the fixed configuration' do
      cs = described_class.new(alpha: 0.01)
      cs.observe([1, 0, 1, 1])
      result = cs.result

      expect(result.n).to eq 4
      expect(result.successes).to eq 3
      expect(result.failures).to eq 1
      expect(result.alpha).to eq 0.01
      expect(result.mixture_prior).to eq cs.mixture_prior
      expect(result.interval).to eq cs.interval
      expect(result.to_h).to include(
        n: 4, successes: 3, failures: 1, alpha: 0.01
      )
    end

    it 'returns a frozen snapshot that cannot be mutated later' do
      cs = described_class.new
      cs.observe(1)
      result = cs.result
      expect(result).to be_frozen
      expect { result.n = 99 }.to raise_error(FrozenError)
    end
  end

  describe 'coverage guarantee' do
    # The mathematical guarantee is the uniform (simultaneous) statement
    # P_p(p in C_n for EVERY n) >= 1 - alpha from Ville's inequality applied
    # to the likelihood-ratio supermartingale. This fixed-seed experiment is
    # supplementary empirical evidence only; it is not the correctness proof
    # and, being deterministic, cannot fail randomly between runs.
    FIXED_SEED = 20261003
    N = 100
    PATHS = 2000
    TRUE_PROBABILITIES = [0.05, 0.3, 0.5, 0.7, 0.95]

    it 'covers the true p along the entire time path with the fixed seed' do
      threshold = Math.log(1 / 0.05)

      TRUE_PROBABILITIES.each do |true_p|
        prng = Random.new(FIXED_SEED + (true_p * 1000).round)
        violated_paths = 0

        PATHS.times do
          cs = described_class.new
          violated = false
          N.times do
            cs.observe(prng.rand < true_p ? 1 : 0)
            violated ||= cs.log_supermartingale(true_p) > threshold
          end
          violated_paths += 1 if violated
        end

        rate = violated_paths.to_f / PATHS
        expect(rate).to be <= 0.08
      end
    end

    it 'covers p = 0 and p = 1 at every prefix of deterministic degenerate paths' do
      [
        [[0] * 200, 0.0],
        [[1] * 200, 1.0]
      ].each do |observations, true_p|
        cs = described_class.new
        observations.each do |value|
          cs.observe(value)
          expect(cs.accept?(true_p)).to be true
          lower, upper = cs.interval
          expect(lower..upper).to cover(true_p)
        end
      end
    end

    it 'is not a prior credible interval: both constructions differ' do
      successes, failures = 8, 3
      cs = described_class.new
      cs.observe(Array.new(successes, 1) + Array.new(failures, 0))
      lower, upper = cs.interval

      posterior = RubyStatistics::Distribution::Beta.new(1 + successes, 1 + failures)
      quantile = lambda do |target|
        lo = 0.0
        hi = 1.0
        80.times do
          middle = (lo + hi) / 2.0
          if posterior.cumulative_function(middle) < target
            lo = middle
          else
            hi = middle
          end
        end
        (lo + hi) / 2.0
      end
      credible_lower = quantile.call(0.025)
      credible_upper = quantile.call(0.975)

      # The Beta(9, 4) central 95% credible interval is a different object
      # with a different meaning; it must not be silently returned in place
      # of the any-time-valid confidence sequence.
      expect(credible_lower).not_to be_within(1e-6).of(lower)
      expect(credible_upper).not_to be_within(1e-6).of(upper)
      expect(upper - lower).to be > credible_upper - credible_lower
    end
  end
end
