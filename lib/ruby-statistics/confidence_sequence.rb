module RubyStatistics
  # Bernoulli confidence sequence based on a likelihood-ratio supermartingale
  # with a pre-specified Beta mixture prior.
  #
  # For a sequence X_1, X_2, ... of Bernoulli observations with unknown
  # success probability p, only the running number of successes and failures
  # is stored. After every observation (success or failure) the accepted set
  # of p values is re-inverted, so each prefix receives its own interval;
  # a fixed-sample confidence interval is never reused.
  #
  # Given a fixed mixture of Beta(a_k, b_k) priors with weights w_k, the
  # e-process tested against the simple hypothesis p = p0 is
#
  #   E_n(p0) = [sum_k w_k * B(S + a_k, F + b_k) / B(a_k, b_k)] /
  #             [p0**S * (1 - p0)**F],
#
  # where S and F are the running success/failure counts and the denominator
  # is interpreted as 0 (so E_n = Infinity) when p0 is incompatible with the
  # prefix. Under p = p0 this is a nonnegative supermartingale (a martingale
  # for a single-component prior), so Ville's inequality gives the
  # simultaneous, any-time-valid statement
#
  #   P_p0(exists n >= 1 : E_n(p0) > 1/alpha) <= alpha.
#
  # Inverting the acceptance condition E_n(p) <= 1/alpha for every prefix
  # yields intervals C_n such that P_p(p belongs to C_n for all n) >= 1-alpha.
  # The guarantee therefore holds for the whole time path and any stopping
  # rule, not just for a single pre-fixed sample size. All computations live
  # in the log domain and p = 0, p = 1, all-success and all-failure prefixes
  # are handled exactly.
  #
  # This is a frequentist confidence sequence. The mixture prior only builds
  # the supermartingale used to construct it; the returned interval is NOT a
  # Bayesian credible interval computed from the posterior of the prior.
  #
  # The configuration (alpha and the Beta mixture) is part of the supermartingale
  # construction and must be fixed before any data are seen. Both are frozen
  # internally and exposed through readers only, so they cannot silently change
  # once observations have been collected.
  class BernoulliConfidenceSequence
    # Uniform Beta(1, 1) prior, the canonical pre-specified mixture.
    DEFAULT_MIXTURE_PRIOR = [{ weight: 1.0, alpha: 1.0, beta: 1.0 }].freeze

    DEFAULT_ALPHA = 0.05

    # Bisection precision for endpoint inversion (2**-70 is far below Float
    # rounding error, after which the interval stops shrinking anyway).
    BISECTION_ITERATIONS = 70

    # Frozen snapshot returned by #result: the configuration it carries is the
    # one that was fixed before data collection, never a post-hoc update.
    Result = Struct.new(:lower, :upper, :n, :successes, :failures, :alpha, :mixture_prior,
                        keyword_init: true) do
      def interval
        [lower, upper]
      end
    end

    attr_reader :successes, :failures, :alpha, :mixture_prior

    # @param alpha [Numeric] simultaneous miscoverage level in (0, 1).
    # @param mixture_prior [Array<Hash>] list of mixture components given as
    #   +{ weight:, alpha:, beta: }+ with strictly positive weights and Beta
    #   parameters. Weights need not be normalized. The same mixture is used
    #   for every prefix.
    def initialize(alpha: DEFAULT_ALPHA, mixture_prior: DEFAULT_MIXTURE_PRIOR)
      alpha = Float(alpha)
      raise ArgumentError, 'alpha must lie strictly between 0 and 1' unless alpha > 0.0 && alpha < 1.0

      mixture_prior = normalize_mixture(mixture_prior)

      @alpha = alpha
      @mixture_prior = mixture_prior.freeze
      @components = mixture_prior.map do |component|
        a = component[:alpha]
        b = component[:beta]
        {
          weight: component[:weight],
          log_weight: Math.log(component[:weight]),
          alpha: a,
          beta: b,
          log_prior_normalizer: log_beta(a, b)
        }.freeze
      end.freeze
      @log_threshold = Math.log(1.0 / alpha)
      @successes = 0
      @failures = 0
    end

    def n
      @successes + @failures
    end

    alias count n

    def observed?
      n > 0
    end

    # Registers one observation (1/true success, 0/false failure) or a batch of
    # observations. Only the running counts are stored; call #interval after
    # each update to invert the freshly accepted set for the new prefix.
    def update(observation = nil)
      if observation.nil?
        raise ArgumentError, 'provide an observation or a non-empty batch of observations'
      end

      if observation.is_a?(Array)
        raise ArgumentError, 'the observation batch must not be empty' if observation.empty?

        observation.each { |value| register(value) }
      else
        register(observation)
      end

      self
    end

    alias observe update

    # Log of the likelihood-ratio supermartingale E_n(p) for the current prefix:
    # log of the Beta mixture marginal likelihood minus the log likelihood
    # under p. Under p this is the log of a nonnegative supermartingale.
    def log_supermartingale(p)
      p = Float(p)
      raise ArgumentError, 'p must lie in [0, 1]' unless p >= 0.0 && p <= 1.0

      log_e_process(p, @successes, @failures)
    end

    # Whether p belongs to the current accepted set (E_n(p) <= 1/alpha).
    def accept?(p)
      log_supermartingale(p) <= @log_threshold
    end

    # The confidence interval [lower, upper] for the current prefix, obtained
    # by inverting the acceptance set in the log domain.
    def interval
      return [0.0, 1.0] if n.zero?

      lower = if @successes.zero?
                0.0
              else
                invert_root(0.0, @successes.to_r / n, @successes, @failures, :lower)
              end

      upper = if @failures.zero?
                1.0
              else
                invert_root(@successes.to_r / n, 1.0, @successes, @failures, :upper)
              end

      [lower, upper]
    end

    # Interval together with the running counts and the (fixed) configuration.
    def result
      lower, upper = interval
      result = Result.new(
        lower: lower,
        upper: upper,
        n: n,
        successes: @successes,
        failures: @failures,
        alpha: @alpha,
        mixture_prior: @mixture_prior
      )
      result.freeze
    end

    private

    def register(value)
      case value
      when true, 1 then @successes += 1
      when false, 0 then @failures += 1
      else raise ArgumentError, "unsupported Bernoulli observation: #{value.inspect}"
      end
    end

    def normalize_mixture(mixture)
      unless mixture.is_a?(Array) && !mixture.empty?
        raise ArgumentError, 'mixture_prior must be a non-empty array of Beta components'
      end

      components = mixture.map do |component|
        weight = Float(component.fetch(:weight))
        a = Float(component.fetch(:alpha))
        b = Float(component.fetch(:beta))
        unless weight > 0.0 && a > 0.0 && b > 0.0
          raise ArgumentError, 'mixture weights and Beta parameters must be strictly positive'
        end

        { weight: weight, alpha: a, beta: b }
      end

      total_weight = components.sum { |component| component[:weight] }
      components.each { |component| component[:weight] /= total_weight }
      components.each(&:freeze)
      components
    end

    def log_beta(a, b)
      Math.lgamma(a)[0] + Math.lgamma(b)[0] - Math.lgamma(a + b)[0]
    end

    # Log of the Bernoulli likelihood under p, including the p = 0/1 limits.
    def log_likelihood(successes, failures, p)
      if p.zero?
        return successes.zero? ? 0.0 : -Float::INFINITY
      end
      return failures.zero? ? 0.0 : -Float::INFINITY if p == 1.0

      successes * Math.log(p) + failures * Math.log(1.0 - p)
    end

    # Log of the Beta mixture marginal (predictive) likelihood of the prefix.
    def log_mixture_marginal(successes, failures)
      terms = @components.map do |component|
        component[:log_weight] +
          log_beta(successes + component[:alpha], failures + component[:beta]) -
          component[:log_prior_normalizer]
      end
      log_sum_exp(terms)
    end

    def log_sum_exp(terms)
      max_term = terms.max
      return max_term if max_term == -Float::INFINITY

      max_term + Math.log(terms.sum { |term| Math.exp(term - max_term) })
    end

    def log_e_process(p, successes, failures)
      # mixture marginal likelihood in the numerator, likelihood under p in
      # the denominator. The denominator is zero only when p is incompatible
      # with the prefix (p = 0 with successes, or p = 1 with failures); in
      # those mutually exclusive cases the e-process is +Infinity. For every
      # other p, including p = 0/1 with an all-failure/all-success prefix,
      # both sides are finite.
      marginal = log_mixture_marginal(successes, failures)
      likelihood = log_likelihood(successes, failures, p)
      return Float::INFINITY if likelihood == -Float::INFINITY && marginal != -Float::INFINITY

      marginal - likelihood
    end

    # Finds the boundary p where log E_n(p) = log(1/alpha). On the requested
    # side the log e-process is monotone, so bisection is safe.
    def invert_root(low, high, successes, failures, side)
      # The MLE bracket endpoint is always accepted (the e-process is minimized
      # there), and p = 0/1 is rejected whenever the opposite outcome has been
      # observed, so a root always exists on the requested side.

      BISECTION_ITERATIONS.times do
        middle = (low + high) / 2.0
        middle_log_ratio = log_e_process(middle, successes, failures)

        if side == :lower
          if middle_log_ratio <= @log_threshold
            high = middle
          else
            low = middle
          end
        else
          if middle_log_ratio <= @log_threshold
            low = middle
          else
            high = middle
          end
        end
      end

      endpoint = side == :lower ? high : low
      endpoint.to_f
    end
  end
end
