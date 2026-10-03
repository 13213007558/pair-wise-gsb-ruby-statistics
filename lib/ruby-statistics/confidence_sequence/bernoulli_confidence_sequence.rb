# frozen_string_literal: true

module RubyStatistics
  module ConfidenceSequence
    # Anytime-valid Bernoulli confidence sequence based on a likelihood ratio
    # e-process built from a pre-specified mixture of Beta prior components.
    #
    # For a sequence X_1, X_2, ... of Bernoulli(p) outcomes and a mixture prior
    # w over Beta(a, b) components, define for every prefix of length n with
    # s successes and f = n - s failures
    #
    #   K(s, f; p) = sum_k w_k * B(s + a_k, f + b_k) / B(a_k, b_k) /
    #                (p^s * (1-p)^f)
    #
    # Under the IID Bernoulli(p0) measure, K(S_n, F_n; p0) is a non-negative
    # martingale (hence a supermartingale) starting at 1. Ville's inequality
    # gives
    #
    #   P_p0( exists n >= 0 : K(S_n, F_n; p0) >= 1 / alpha ) <= alpha.
    #
    # Inverting the acceptance set { p : K(s, f; p) < 1/alpha } at every prefix
    # yields a confidence *sequence*: the true p stays inside every reported
    # interval simultaneously along the entire (possibly infinite) data path
    # with probability at least 1 - alpha. The bound is on the whole time path,
    # not a per-sample fixed-n coverage statement, and alpha is paid only once.
    #
    # The Beta mixture only builds the likelihood-ratio supermartingale. This
    # is a frequentist, pathwise guarantee; it is not a Bayesian credible
    # interval and must not be interpreted as one.
    #
    # Only the success/failure counts are stored: the object keeps updating in
    # constant memory as new observations arrive. All computations are carried
    # out in the log domain, and the boundary outcomes all successes, all
    # failures and p = 0 / p = 1 are handled explicitly.
    class BernoulliConfidenceSequence
      # Snapshot returned after each update.
      Result = Struct.new(:lower, :upper, :sample_count, :config, keyword_init: true) do
        def to_h
          {
            lower: lower,
            upper: upper,
            sample_count: sample_count,
            config: config
          }
        end
      end

      # One Beta(a, b) component with its (already normalized) mixture weight.
      Component = Struct.new(:weight, :alpha_prior, :beta_prior, keyword_init: true) do
        def to_h
          { weight: weight, alpha_prior: alpha_prior, beta_prior: beta_prior }
        end
      end

      DEFAULT_COMPONENTS = [Component.new(weight: 1.0, alpha_prior: 1.0, beta_prior: 1.0)].freeze

      # Number of bisection refinements. Each refinement halves the bracket;
      # 60 halvings go below Float epsilon precision and the search stalls
      # harmlessly once Float granularity is reached.
      ROOT_ITERATIONS = 60
      private_constant :ROOT_ITERATIONS

      attr_reader :successes, :failures, :alpha, :components

      # @param alpha [Float, Integer, Rational] miscoverage level in (0, 1).
      # @param prior [Array<Hash>] pre-specified mixture of Beta components.
      #   Each entry must have strictly positive :alpha_prior and :beta_prior
      #   and a non-negative :weight. Weights are normalized exactly once, at
      #   construction time, before any data is observed.
      def initialize(alpha:, prior: nil)
        alpha = Float(alpha)
        raise ArgumentError, 'alpha must be strictly between 0 and 1' unless alpha > 0.0 && alpha < 1.0

        raw_components = prior || DEFAULT_COMPONENTS
        build_components(raw_components)

        @alpha = alpha
        @log_threshold = -Math.log(alpha)
        @successes = 0
        @failures = 0
        freeze_config
      end

      # Convenience constructor for one-off computations. It computes the
      # accepted interval for a prefix of +successes+ / +failures+ without
      # mutating any running sequence.
      def self.interval(successes:, failures:, alpha:, prior: nil)
        new(alpha: alpha, prior: prior).tap do |cs|
          cs.observe!(1, successes)
          cs.observe!(0, failures)
        end.result
      end

      # Log of the likelihood-ratio e-process K(s, f; p) for the given prefix.
      #
      # Returns Float::INFINITY when the prefix is impossible under p (p = 0
      # with any success, or p = 1 with any failure) and -Float::INFINITY for
      # the zero-length prefix.
      def log_e_process(p = nil, successes: @successes, failures: @failures)
        p = Float(p)
        raise ArgumentError, 'p must be between 0 and 1' unless p >= 0.0 && p <= 1.0
        raise ArgumentError, 'counts must be non-negative integers' unless valid_counts?(successes, failures)

        n = successes + failures
        return -Float::INFINITY if n.zero?

        if p.zero?
          return successes.zero? ? log_marginal(successes, failures) : Float::INFINITY
        end
        return failures.zero? ? log_marginal(successes, failures) : Float::INFINITY if p == 1.0

        log_marginal(successes, failures) - successes * Math.log(p) - failures * Math.log(1.0 - p)
      end

      # Fold one or more observations into the running counts.
      #
      # @param outcome [Integer] 1 for a success, 0 for a failure.
      # @param times [Integer] how many identical observations to record.
      # @return [self] so calls can be chained; use #result to read the
      #   refreshed interval.
      def observe!(outcome, times = 1)
        times = Integer(times)
        raise ArgumentError, 'times must be a non-negative integer' if times.negative?
        return self if times.zero?

        case outcome
        when 1 then @successes += times
        when 0 then @failures += times
        else raise ArgumentError, "outcome must be 1 (success) or 0 (failure), got #{outcome.inspect}"
        end
        self
      end

      alias update! observe!

      def success!(times = 1)
        observe!(1, times)
      end

      def failure!(times = 1)
        observe!(0, times)
      end

      def sample_count
        @successes + @failures
      end

      # Recompute the interval for the current prefix. No state other than the
      # counts exists, so every refresh uses the current prefix only; a fixed
      # sample interval is never reused.
      def bounds(successes: @successes, failures: @failures)
        raise ArgumentError, 'counts must be non-negative integers' unless valid_counts?(successes, failures)

        return [0.0, 1.0] if successes + failures == 0

        # p is excluded wherever K(s, f; p) > 1/alpha. With any success,
        # K(s, f; 0) is infinite, so 0 is excluded and the lower endpoint is
        # the interior root; with no successes, K(0) is finite (at most 1) and
        # 0 belongs to every interval.
        lower = successes.zero? ? 0.0 : root(successes, failures, :left)

        # Symmetrically, 1 is excluded as soon as a failure is observed.
        upper = failures.zero? ? 1.0 : root(successes, failures, :right)

        [lower, upper]
      end

      # Fresh interval, sample size and immutable config for the current prefix.
      def result
        lower, upper = bounds
        Result.new(
          lower: lower,
          upper: upper,
          sample_count: sample_count,
          config: config
        )
      end

      def to_h
        result.to_h
      end

      def config
        {
          alpha: alpha,
          prior: components.map(&:to_h)
        }
      end

      private

      def valid_counts?(successes, failures)
        successes.is_a?(::Integer) && failures.is_a?(::Integer) &&
          successes >= 0 && failures >= 0
      end

      def build_components(raw_components)
        unless raw_components.is_a?(Array) && raw_components.any?
          raise ArgumentError, 'prior must be a non-empty array of Beta components'
        end

        @components = raw_components.map do |component|
          unless component.is_a?(::Hash) || component.is_a?(Component)
            raise ArgumentError,
                  'each prior component must be a Hash with :alpha_prior, :beta_prior and :weight'
          end
          provided_weight = component.is_a?(::Hash) ? component.fetch(:weight, 1.0) : component[:weight]
          weight = provided_weight.nil? ? 1.0 : Float(provided_weight)
          a = Float(component[:alpha_prior])
          b = Float(component[:beta_prior])
          raise ArgumentError, 'alpha_prior and beta_prior must be strictly positive' unless a > 0.0 && b > 0.0
          raise ArgumentError, 'component weights must be non-negative' unless weight >= 0.0
          Component.new(weight: weight, alpha_prior: a, beta_prior: b)
        rescue TypeError
          raise ArgumentError,
                'each prior component needs positive numeric :alpha_prior and :beta_prior'
        end

        total_weight = @components.sum(&:weight)
        raise ArgumentError, 'at least one component must have a positive weight' unless total_weight.positive?

        @components.each { |component| component.weight = component.weight / total_weight }
      end

      def freeze_config
        @components.each(&:freeze)
        @components.freeze
        @alpha.freeze
        @log_threshold.freeze
      end

      # log of the Beta mixture marginal (predictive) of the given prefix:
      # log sum_k w_k B(s + a_k, f + b_k) / B(a_k, b_k).
      def log_marginal(successes, failures)
        weighted = @components.map do |component|
          Math.log(component.weight) +
            log_beta(successes + component.alpha_prior, failures + component.beta_prior) -
            log_beta(component.alpha_prior, component.beta_prior)
        end
        log_sum_exp(weighted)
      end

      # log B(x, y) = log Gamma(x, y) - log Gamma(x + y)
      def log_beta(x, y)
        Math.lgamma(x).first + Math.lgamma(y).first - Math.lgamma(x + y).first
      end

      def log_sum_exp(terms)
        finite_terms = terms.reject { |term| term == -Float::INFINITY }
        return -Float::INFINITY if finite_terms.empty?

        maximum = finite_terms.max
        maximum + Math.log(finite_terms.sum { |term| Math.exp(term - maximum) })
      end

      # log K(s, f; p) - log(1/alpha), the strictly convex function whose two
      # zeros (when they exist inside the open interval) are the interval ends.
      def root_residual(p, successes, failures)
        log_e_process(p, successes: successes, failures: failures) - @log_threshold
      end

      # Bisection toward the boundary 0 (left root) or 1 (right root). At the
      # empirical mean the residual is strictly negative (the marginal is an
      # averaged likelihood, so K is at most 1 there and log(1/alpha) > 0),
      # while the residual diverges to +infinity at the rejected boundary.
      # Convexity of log K guarantees a single root on each side of the mean.
      def root(successes, failures, side)
        mode = successes.to_f / (successes + failures)

        if side == :left
          low = 0.0
          high = mode
          ROOT_ITERATIONS.times do
            middle = (low + high) / 2.0
            break if middle == low || middle == high

            if root_residual(middle, successes, failures) > 0.0
              low = middle # still excluded, root is above middle
            else
              high = middle
            end
          end
          high
        else
          low = mode
          high = 1.0
          ROOT_ITERATIONS.times do
            middle = (low + high) / 2.0
            break if middle == low || middle == high

            if root_residual(middle, successes, failures) > 0.0
              high = middle # still excluded, root is below middle
            else
              low = middle
            end
          end
          low
        end
      end
    end
  end
end
