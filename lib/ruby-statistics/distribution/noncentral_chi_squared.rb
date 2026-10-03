# frozen_string_literal: true

module RubyStatistics
  module Distribution
    # Non-central chi-squared distribution with k degrees of freedom and
    # non-centrality parameter lambda (lambda >= 0). This is the sampling
    # distribution of the chi-squared statistic under the alternative
    # hypothesis, so it is the tool needed for power analysis.
    #
    # The CDF and the survival function are evaluated through the Poisson
    # mixture representation:
    #
    #   F(x; k, l) = sum_j Poisson(j; l/2) * P(k/2 + j, x/2)
    #   S(x; k, l) = sum_j Poisson(j; l/2) * Q(k/2 + j, x/2)
    #
    # where P and Q are the regularised lower/upper incomplete gamma
    # functions. The mixture sum starts at the mode of the Poisson weights
    # (floor(l/2)) and expands outwards, so large non-centrality parameters
    # do not underflow the weights. Every evaluation carries a rigorous
    # bound on the truncated Poisson tail; if that bound cannot be driven
    # below the requested tolerance within max_terms terms, the float
    # returning methods raise ConvergenceError instead of silently
    # returning an inaccurate probability.
    #
    # The upper tail is computed directly from the regularised upper
    # incomplete gamma function (a Legendre continued fraction), never as
    # 1 - CDF, so extreme upper tails keep full relative accuracy.
    class NonCentralChiSquared
      # Raised when the mixture sum or the inverse-CDF iteration cannot
      # satisfy the requested tolerance. This is an explicit failure
      # status, distinct from the valid p = 0 / p = 1 quantile endpoints.
      class ConvergenceError < StandardError; end

      # Report of a mixture evaluation. error_bound is a rigorous upper
      # bound on the probability mass of the truncated Poisson tail (and
      # therefore on the absolute error of value). converged is false when
      # max_terms was exhausted before the bound dropped below the
      # requested tolerance; terms counts the Poisson terms used.
      Evaluation = Struct.new(:value, :error_bound, :terms, :converged)

      # Default relative tolerance for the truncation bound.
      DEFAULT_TOLERANCE = 1.0e-13
      # Default cap on the number of Poisson terms taken by the mixture.
      DEFAULT_MAX_TERMS = 100_000
      # A scaled Poisson weight below this threshold is only kept while it
      # can still affect the requested relative tolerance.
      LOG_WEIGHT_FLOOR = Math.log(1.0e-18)
      # log of the smallest positive denormal double: a truncation bound
      # below this (scaled by the mode weight) cannot change any
      # representable probability, so the mixture may stop there.
      LOG_MIN_POSITIVE = Math.log(5e-324)
      # Termination epsilon for the gamma series / continued fraction.
      GAMMA_EPSILON = 1.0e-16
      GAMMA_MAX_ITERATIONS = 1_000_000
      TINY = 1.0e-290

      attr_reader :degrees_of_freedom, :noncentrality

      def initialize(degrees_of_freedom, noncentrality = 0.0)
        degrees_of_freedom = Float(degrees_of_freedom)
        noncentrality = Float(noncentrality)

        unless degrees_of_freedom > 0.0
          raise ArgumentError, 'degrees_of_freedom must be positive'
        end
        unless noncentrality >= 0.0
          raise ArgumentError, 'noncentrality must be non-negative'
        end

        @degrees_of_freedom = degrees_of_freedom
        @noncentrality = noncentrality
        @shape = degrees_of_freedom / 2.0
        @poisson_mean = noncentrality / 2.0
      end

      def mean
        degrees_of_freedom + noncentrality
      end

      def variance
        2.0 * (degrees_of_freedom + 2.0 * noncentrality)
      end

      # Lower tail probability P(X <= value). Raises ConvergenceError if
      # the mixture cannot meet the requested tolerance within max_terms.
      def cumulative_function(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        unwrap(cumulative_evaluation(value, tolerance: tolerance, max_terms: max_terms))
      end

      # Upper tail probability P(X > value), computed directly from the
      # regularised upper incomplete gamma function (never as 1 - CDF).
      def survival_function(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        unwrap(survival_evaluation(value, tolerance: tolerance, max_terms: max_terms))
      end

      # log P(X > value), accumulated in log-space so it stays accurate
      # even when the survival probability itself underflows the double
      # range. Returns 0.0 for value <= 0 and -Infinity for value =
      # Infinity.
      def log_survival_function(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        unwrap(log_survival_evaluation(value, tolerance: tolerance, max_terms: max_terms))
      end

      # Full report of the lower tail evaluation: the probability plus a
      # rigorous bound on the truncated Poisson tail and a convergence
      # flag. Never raises for non-convergence; inspect the flag instead.
      def cumulative_evaluation(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        value = Float(value)
        return Evaluation.new(0.0, 0.0, 0, true) if value <= 0.0
        return Evaluation.new(1.0, 0.0, 0, true) if value.infinite?

        evaluate_mixture(value / 2.0, :lower, tolerance, max_terms)
      end

      # Full report of the upper tail evaluation. See cumulative_evaluation.
      def survival_evaluation(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        value = Float(value)
        return Evaluation.new(1.0, 0.0, 0, true) if value <= 0.0
        return Evaluation.new(0.0, 0.0, 0, true) if value.infinite?

        evaluate_mixture(value / 2.0, :upper, tolerance, max_terms)
      end

      # Full report of the log-space upper tail evaluation. The reported
      # value is log P(X > value); error_bound still bounds the survival
      # probability itself, not its logarithm.
      def log_survival_evaluation(value, tolerance: DEFAULT_TOLERANCE, max_terms: DEFAULT_MAX_TERMS)
        value = Float(value)
        return Evaluation.new(0.0, 0.0, 0, true) if value <= 0.0
        return Evaluation.new(-Float::INFINITY, 0.0, 0, true) if value.infinite?

        evaluate_mixture(value / 2.0, :log_upper, tolerance, max_terms)
      end

      # Smallest x with P(X <= x) >= probability. The p = 0 and p = 1
      # endpoints return 0.0 and Infinity; any other probability is
      # inverted by first bracketing the root and then iterating with a
      # safeguarded secant method (Illinois). Raises ConvergenceError if
      # the iteration does not converge, which is distinct from the valid
      # endpoint answers.
      def inverse_cumulative_function(probability, tolerance: 1.0e-12, max_iterations: 100)
        probability = Float(probability)
        unless (0.0..1.0).cover?(probability)
          raise ArgumentError, 'probability must be in [0, 1]'
        end

        return 0.0 if probability.zero?
        return Float::INFINITY if probability == 1.0

        # In the upper tail solve S(x) = 1 - p so the iteration uses the
        # survival function and keeps full relative accuracy; both
        # residuals are increasing functions of x with residual(0) < 0.
        residual =
          if probability > 0.5
            tail = 1.0 - probability
            ->(x) { tail - survival_function(x) }
          else
            ->(x) { cumulative_function(x) - probability }
          end

        find_quantile(residual, tolerance, max_iterations)
      end

      private

      def unwrap(evaluation)
        return evaluation.value if evaluation.converged

        raise ConvergenceError,
              'mixture did not converge: ' +
              'truncation bound ' + evaluation.error_bound.to_s +
              ' after ' + evaluation.terms.to_s + ' terms'
      end

      # ----------------------------------------------------------------
      # Poisson mixture engine
      # ----------------------------------------------------------------

      # Evaluates sum_j w_j * G_j, where w_j are Poisson(mu) weights and
      # G_j is P(shape + j, y) for the lower tail or Q(shape + j, y) for
      # the upper tail. Weights are handled relative to their mode
      # (v_j = w_j / w_j0, so v_j0 = 1) and computed per term in
      # log-space through a Stirling expansion, which keeps both huge
      # non-centrality and deep tails away from underflow and
      # cancellation. Every helper returns its partial sum together with
      # the log of a rigorous bound on the neglected Poisson tail.
      def evaluate_mixture(y, tail, tolerance, max_terms)
        if @poisson_mean.zero?
          # A single mixture term: exactly the central chi-squared.
          value =
            case tail
            when :lower then gamma_p(@shape, y)
            when :upper then gamma_q(@shape, y)
            else log_gamma_q(@shape, y)
            end
          return Evaluation.new(value, 0.0, 1, true)
        end

        mu = @poisson_mean
        log_mu = Math.log(mu)
        j0 = mu.floor
        log_w0 = log_poisson_weight(j0, mu, log_mu)
        w0 = Math.exp(log_w0)
        log_tolerance = Math.log(tolerance)
        state = { terms: 1, max_terms: max_terms, truncated: false }
        # In log-space mode the sum never underflows, so no absolute
        # floor applies; otherwise a bound that cannot move any
        # representable double is good enough.
        state[:log_floor] =
          tail == :log_upper ? -Float::INFINITY : LOG_MIN_POSITIVE - log_w0

        # Each direction gets half of the tolerance budget, so the
        # combined truncation bound meets the requested tolerance.
        split_tolerance = 0.5 * tolerance
        log_split_tolerance = Math.log(split_tolerance)

        if tail == :lower
          sum, log_lower = lower_tail_sum(y, mu, log_mu, j0, log_w0, split_tolerance, state)
          sum, log_upper = ascend_lower_tail(y, mu, log_mu, j0, log_w0, sum, log_split_tolerance, state)
          log_sum = Math.log(sum)
          value = (w0 * sum).clamp(0.0, 1.0)
        else
          log_space = tail == :log_upper
          sum, log_lower = descend_upper_tail(y, mu, log_mu, j0, log_w0, log_space, state)
          sum, log_upper = ascend_upper_tail(y, mu, log_mu, j0, log_w0, sum, log_split_tolerance, log_space, state)
          log_sum = log_space ? sum : Math.log(sum)
          value = log_space ? log_w0 + sum : (w0 * sum).clamp(0.0, 1.0)
        end

        log_remaining = log_sum_exp(log_lower, log_upper)
        converged =
          !state[:truncated] &&
          (log_remaining <= log_tolerance + log_sum || log_remaining <= state[:log_floor])

        Evaluation.new(value, w0 * Math.exp(log_remaining), state[:terms], converged)
      end

      # Lower tail, terms j0-1 downwards. P(shape+j-1, y) = P + t_{j-1}
      # adds only positive terms, so the recursion is stable. When the
      # recurrence term underflows at the mode (deep lower tail with
      # large lambda), P is recomputed per term so the dominant low-order
      # terms are not lost. Returns [sum, log_remaining_bound].
      def lower_tail_sum(y, mu, log_mu, j0, log_w0, tolerance, state)
        p_curr = gamma_p(@shape + j0, y)
        t_curr = gamma_t(@shape + j0, y)
        sum = p_curr # v_j0 = 1
        per_term = t_curr.zero?
        log_remaining = -Float::INFINITY

        j = j0
        while j.positive?
          if state[:terms] >= state[:max_terms]
            state[:truncated] = true
            log_remaining = log_poisson_weight(j, mu, log_mu) - log_w0 +
                              Math.log(j) - log_mu - log1p(-(j - 1.0) / mu)
            break
          end

          v_next = Math.exp(log_poisson_weight(j - 1, mu, log_mu) - log_w0)

          if per_term
            p_next = gamma_p(@shape + j - 1, y)
          else
            t_curr *= (@shape + j) / y # t_{j-1} = t_j * (shape + j) / y
            p_next = [p_curr + t_curr, 1.0].min
          end

          if j > 1
            # The weights decay geometrically with ratio at most
            # (j-1)/mu and P decreases with j, so the neglected mass is
            # bounded by v_{j-1} * P_{j-1} / (1 - (j-1)/mu).
            neglected = v_next * p_next / (1.0 - (j - 1.0) / mu)
            log_neglected = safe_log(neglected)
            if neglected <= tolerance * sum || log_neglected <= state[:log_floor]
              log_remaining = log_neglected
              break
            end
          end

          sum += v_next * p_next
          p_curr = p_next
          j -= 1
          state[:terms] += 1
        end

        [sum, log_remaining]
      end

      # Lower tail, terms j0+1 upwards. P(shape+j+1, y) = P - t_j can
      # only lose absolute precision, which is harmless for a probability.
      # P decreases with j, so the remaining mass is bounded by
      # P_j times the remaining weight.
      def ascend_lower_tail(y, mu, log_mu, j0, log_w0, sum, log_tolerance, state)
        p_curr = gamma_p(@shape + j0, y)
        t_curr = gamma_t(@shape + j0, y)
        j = j0
        log_remaining = log_remaining_bound(0.0, mu, j) + safe_log(p_curr)

        while log_remaining > log_tolerance + safe_log(sum) &&
              log_remaining > state[:log_floor]
          break if exhausted?(state)

          j += 1
          log_v = log_poisson_weight(j, mu, log_mu) - log_w0
          p_curr = [p_curr - t_curr, 0.0].max
          t_curr *= y / (@shape + j)
          sum += Math.exp(log_v) * p_curr
          state[:terms] += 1
          log_remaining = log_remaining_bound(log_v, mu, j) + safe_log(p_curr)
        end

        [sum, log_remaining]
      end

      # Upper tail, terms j0-1 downwards. Q(shape+j, y) increases with j,
      # so each term is evaluated directly in log-space as
      # exp(log_v + log_Q): a downward Q recursion would both underflow
      # and cancel catastrophically in the deep tail. Returns the partial
      # sum (or its logarithm in log-space mode) and the log of a bound
      # on the neglected mass, using Q_i <= Q_j for i < j.
      def descend_upper_tail(y, mu, log_mu, j0, log_w0, log_space, state)
        sum = log_space ? -Float::INFINITY : 0.0
        log_v = 0.0
        j = j0

        while j.positive? && log_v > LOG_WEIGHT_FLOOR
          break if exhausted?(state)

          j -= 1
          log_v = log_poisson_weight(j, mu, log_mu) - log_w0
          state[:terms] += 1
          sum = accumulate(sum, log_v + log_gamma_q(@shape + j, y), log_space)
        end

        log_remaining =
          if j.zero?
            -Float::INFINITY
          else
            ratio = j / mu
            log_v + Math.log(ratio) - log1p(-ratio) + log_gamma_q(@shape + j, y)
          end

        [sum, log_remaining]
      end

      # Upper tail, mode term plus terms j0+1 upwards. Q(shape+j+1, y) =
      # Q + t_j adds only positive terms, so the upward recursion is
      # stable; in log-space mode each Q is evaluated directly instead.
      def ascend_upper_tail(y, mu, log_mu, j0, log_w0, sum, log_tolerance, log_space, state)
        j = j0

        if log_space
          sum = log_sum_exp(sum, log_gamma_q(@shape + j0, y))
          log_remaining = log_remaining_bound(0.0, mu, j)
          while log_remaining > log_tolerance + sum &&
                log_remaining > state[:log_floor]
            break if exhausted?(state)

            j += 1
            log_v = log_poisson_weight(j, mu, log_mu) - log_w0
            sum = log_sum_exp(sum, log_v + log_gamma_q(@shape + j, y))
            state[:terms] += 1
            log_remaining = log_remaining_bound(log_v, mu, j)
          end
        else
          q_curr = gamma_q(@shape + j0, y)
          t_curr = gamma_t(@shape + j0, y)
          sum += q_curr
          log_remaining = log_remaining_bound(0.0, mu, j)
          while log_remaining > log_tolerance + safe_log(sum) &&
                log_remaining > state[:log_floor]
            break if exhausted?(state)

            j += 1
            log_v = log_poisson_weight(j, mu, log_mu) - log_w0
            q_curr += t_curr
            t_curr *= y / (@shape + j)
            sum += Math.exp(log_v) * q_curr
            state[:terms] += 1
            log_remaining = log_remaining_bound(log_v, mu, j)
          end
        end

        [sum, log_remaining]
      end

      # log of a bound on the scaled Poisson weight remaining above index
      # j: the weights decay geometrically with ratio at most mu/(j+1).
      # Returns 0.0 (a useless bound of one) while the weights still grow.
      def log_remaining_bound(log_v, mu, j)
        return 0.0 if j + 1.0 <= mu

        ratio = mu / (j + 1.0)
        log_v + Math.log(ratio) - log1p(-ratio)
      end

      def exhausted?(state)
        return false if state[:terms] < state[:max_terms]

        state[:truncated] = true
      end

      def safe_log(value)
        value <= 0.0 ? -Float::INFINITY : Math.log(value)
      end

      # Math.log1p does not exist in ruby; this is Kahan's log-based
      # formulation, accurate for arguments close to zero.
      def log1p(x)
        return x if x.zero?

        u = 1.0 + x
        u == 1.0 ? x : Math.log(u) * x / (u - 1.0)
      end

      # Adds a term given in log-space to an accumulator that holds either
      # the plain sum or its logarithm.
      def accumulate(sum, log_term, log_space)
        log_space ? log_sum_exp(sum, log_term) : sum + Math.exp(log_term)
      end

      def log_sum_exp(log_a, log_b)
        return log_b if log_a == -Float::INFINITY
        return log_a if log_b == -Float::INFINITY

        high, low = log_a >= log_b ? [log_a, log_b] : [log_b, log_a]
        high + log1p(Math.exp(low - high))
      end

      # ----------------------------------------------------------------
      # Regularised incomplete gamma helpers (plain float arithmetic)
      # ----------------------------------------------------------------

      # Recurrence term gamma_t(a, x) = x^a e^-x / Gamma(a + 1), linking
      # neighbouring regularised gammas: P(a+1, x) = P(a, x) - t and
      # Q(a+1, x) = Q(a, x) + t. Uses Gamma(a + 1) = a * Gamma(a).
      def gamma_t(a, x)
        Math.exp(log_gamma_kernel(a, x) - Math.log(a))
      end

      # Regularised lower incomplete gamma P(a, x).
      def gamma_p(a, x)
        return 0.0 if x <= 0.0
        return 1.0 if x.infinite?

        x < a + 1.0 ? gamma_p_series(a, x) : 1.0 - gamma_q_fraction(a, x)
      end

      # Regularised upper incomplete gamma Q(a, x), evaluated directly
      # (not as 1 - P) so deep upper tails keep full relative accuracy.
      def gamma_q(a, x)
        return 1.0 if x <= 0.0
        return 0.0 if x.infinite?

        x < a + 1.0 ? 1.0 - gamma_p_series(a, x) : gamma_q_fraction(a, x)
      end

      # log Q(a, x), accurate even when Q itself underflows the double
      # range. In the series region Q is bounded away from zero, so
      # log1p(-P) is safe there.
      def log_gamma_q(a, x)
        return 0.0 if x <= 0.0
        return -Float::INFINITY if x.infinite?

        if x < a + 1.0
          log1p(-gamma_p_series(a, x))
        else
          log_gamma_kernel(a, x) + Math.log(gamma_q_fraction_factor(a, x))
        end
      end

      # log of the kernel x^a e^-x / Gamma(a), that is
      # -x + a ln x - lgamma(a). For a >= 16 the three terms are huge and
      # nearly cancel, so lgamma is replaced by its Stirling expansion
      # and the result is assembled as a*(log1p(d) - d) with d = x/a - 1,
      # keeping the absolute error near 1e-15 even for very large a.
      STIRLING_THRESHOLD = 16.0

      def log_gamma_kernel(a, x)
        if a < STIRLING_THRESHOLD
          -x + a * Math.log(x) - Math.lgamma(a).first
        else
          delta = x / a - 1.0
          a * (log1p(delta) - delta) + 0.5 * Math.log(a / (2.0 * Math::PI)) - stirling_correction(a)
        end
      end

      # r(a) = lgamma(a) - (a - 1/2) ln a + a - ln(sqrt(2 pi)), via the
      # Stirling series. For a >= 16 the truncation error is below 1e-17.
      def stirling_correction(a)
        a2 = a * a
        r = 1.0 / 1188.0 - 691.0 / (360_360.0 * a2)
        r = r / a2 - 1.0 / 1680.0
        r = r / a2 + 1.0 / 1260.0
        r = r / a2 - 1.0 / 360.0
        (r / a2 + 1.0 / 12.0) / a
      end

      # log of the Poisson weight exp(-mu) mu^j / j!, computed as
      # j*(log1p(d) - d) - ln(sqrt(2 pi j)) - r(j) with d = mu/j - 1 for
      # j >= 16. A direct -mu + j ln(mu) - lgamma(j+1) would lose
      # several digits to cancellation for large j.
      def log_poisson_weight(j, mu, log_mu)
        if j < STIRLING_THRESHOLD
          -mu + j * log_mu - Math.lgamma(j + 1.0).first
        else
          delta = mu / j - 1.0
          j * (log1p(delta) - delta) - 0.5 * Math.log(2.0 * Math::PI * j) - stirling_correction(j)
        end
      end

      # P(a, x) through its power series (Numerical Recipes, gser).
      def gamma_p_series(a, x)
        ap = a
        sum = 1.0 / a
        delta = sum

        GAMMA_MAX_ITERATIONS.times do
          ap += 1.0
          delta *= x / ap
          sum += delta
          break if delta.abs <= sum.abs * GAMMA_EPSILON
        end

        sum * Math.exp(log_gamma_kernel(a, x))
      end

      # Q(a, x) = e^-x x^a / Gamma(a) * h, with h from the Legendre
      # continued fraction (Numerical Recipes, gcf).
      def gamma_q_fraction(a, x)
        Math.exp(log_gamma_kernel(a, x)) * gamma_q_fraction_factor(a, x)
      end

      def gamma_q_fraction_factor(a, x)
        b = x + 1.0 - a
        c = 1.0 / TINY
        d = 1.0 / [b, TINY].max
        h = d

        1.upto(GAMMA_MAX_ITERATIONS) do |i|
          an = -i * (i - a)
          b += 2.0
          d = an * d + b
          d = TINY if d.abs < TINY
          c = b + an / c
          c = TINY if c.abs < TINY
          d = 1.0 / d
          delta = d * c
          h *= delta
          break if (delta - 1.0).abs <= GAMMA_EPSILON
        end

        h
      end

      # ----------------------------------------------------------------
      # Root finding for the inverse CDF
      # ----------------------------------------------------------------

      # Brackets the quantile and solves for it. The upper end of the
      # bracket is found by doubling from the mean; the lower end by
      # stepping down from there with an accelerating (squaring) step,
      # so roots many orders of magnitude below the mean are still
      # bracketed. The residual is negative at zero for every
      # probability in (0, 1).
      def find_quantile(residual, tolerance, max_iterations)
        hi = [mean, 1.0].max
        f_hi = residual.call(hi)
        expansions = 0

        while f_hi.negative?
          hi *= 2.0
          hi = Float::MAX if hi.infinite?
          expansions += 1
          raise ConvergenceError, 'unable to bracket the quantile' if expansions > 2100

          f_hi = residual.call(hi)
        end
        return hi if f_hi.zero?

        upper = hi
        lower = hi
        step = 1
        loop do
          lower = hi * (0.5**step)
          # The quantile lies below the smallest positive double, so
          # zero is the only representable answer.
          return 0.0 if lower.zero?

          f_lo = residual.call(lower)
          return lower if f_lo.zero?
          break if f_lo.negative?

          upper = lower
          step *= 2
        end

        illinois_log(Math.log(lower), Math.log(upper), residual, tolerance, max_iterations)
      end

      # Safeguarded secant iteration (Illinois flavour of regula falsi)
      # in log-space: the root stays bracketed in [exp(u_lo), exp(u_hi)]
      # at all times and the secant step is taken from the endpoint
      # values, whose signs are preserved by the damping, so the
      # candidate always lands inside the bracket and the iteration
      # cannot diverge, even when the bracket spans hundreds of orders
      # of magnitude. Raises ConvergenceError when max_iterations is
      # exhausted, an explicit failure status that is distinct from the
      # valid p = 0 / p = 1 endpoint answers.
      def illinois_log(u_lo, u_hi, residual, tolerance, max_iterations)
        f_lo = residual.call(Math.exp(u_lo))
        f_hi = residual.call(Math.exp(u_hi))
        return Math.exp(u_lo) if f_lo.zero?
        return Math.exp(u_hi) if f_hi.zero?

        last_side = 0
        u = u_lo

        max_iterations.times do
          width = u_hi - u_lo
          u = u_hi - f_hi * width / (f_hi - f_lo)
          # Safeguard: when the secant candidate hugs an endpoint
          # (extremely asymmetric residuals), bisect instead so the
          # bracket still shrinks by a factor of two.
          margin = 0.1 * width
          u = 0.5 * (u_lo + u_hi) if u <= u_lo + margin || u >= u_hi - margin
          return Math.exp(u) if width <= tolerance

          f_u = residual.call(Math.exp(u))
          return Math.exp(u) if f_u.zero?

          if f_u.negative?
            u_lo = u
            f_lo = f_u
            f_hi *= 0.5 if last_side == -1
            last_side = -1
          else
            u_hi = u
            f_hi = f_u
            f_lo *= 0.5 if last_side == 1
            last_side = 1
          end
        end

        raise ConvergenceError, 'quantile iteration did not converge'
      end
    end
  end
end
