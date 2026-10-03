module RubyStatistics
  module Distribution
    # Noncentral chi-squared distribution with k degrees of freedom and
    # noncentrality parameter lambda.
    #
    # The distribution function is evaluated as a Poisson mixture of central
    # chi-squared distribution functions:
    #
    #   F(x; k, lambda) = sum_{j=0}^inf w_j * P(k/2 + j, x/2)
    #
    # where w_j = exp(-lambda/2) * (lambda/2)^j / j! and P(a, y) is the
    # regularised lower incomplete gamma function. The mixture is summed
    # starting at the mode of the Poisson weights, j0 = floor(lambda/2),
    # expanding downward and upward, so large noncentrality parameters do not
    # require summing hundreds of negligible terms and the weights never
    # underflow at the starting point. Each direction stops when a geometric
    # bound on the remaining Poisson weight (and hence on the remaining
    # mixture mass, since 0 <= P(a, y) <= 1) falls below a tolerance times
    # the accumulated partial sum; if the term cap is hit first, a
    # NotConvergedError is raised instead of returning a wrong answer.
    #
    # The upper tail (survival function) is evaluated as the same mixture of
    # regularised upper incomplete gammas, never as 1 - F(x), so it stays
    # accurate deep into the right tail. The log survival function is
    # accumulated in log space and remains finite even when the survival
    # probability itself underflows a double.
    #
    # References:
    #   Johnson, Kotz and Balakrishnan, "Continuous Univariate
    #     Distributions", Vol. 2, 2nd ed., chapter 29.
    #   Press, Teukolsky, Vetterling and Flannery, "Numerical Recipes",
    #     3rd ed., section 6.2 (incomplete gamma evaluation).
    class NoncentralChiSquared
      class NotConvergedError < StandardError; end

      MAX_SERIES_TERMS = 100_000
      MIXTURE_MAX_TERMS = 100_000
      LOG_SERIES_TOLERANCE = Math.log(1.0e-14).freeze
      GAMMA_EPSILON = 1.0e-16
      INVERSE_P_TOLERANCE = 1.0e-13
      INVERSE_X_TOLERANCE = 1.0e-12
      MAX_INVERSE_ITERATIONS = 100

      attr_reader :degrees_of_freedom, :noncentrality

      def initialize(degrees_of_freedom, noncentrality)
        unless degrees_of_freedom.is_a?(Numeric) && degrees_of_freedom.positive?
          raise ArgumentError, "degrees_of_freedom must be a positive number"
        end

        unless noncentrality.is_a?(Numeric) && !noncentrality.negative?
          raise ArgumentError, "noncentrality must be a non-negative number"
        end

        @degrees_of_freedom = degrees_of_freedom
        @noncentrality = noncentrality
        @half_degrees_of_freedom = degrees_of_freedom.to_f / 2.0
        @half_noncentrality = noncentrality.to_f / 2.0
      end

      def mean
        degrees_of_freedom + noncentrality
      end

      def variance
        2.0 * (degrees_of_freedom + 2.0 * noncentrality)
      end

      def cumulative_function(value)
        return 0.0 if value <= 0.0

        y = value.to_f / 2.0

        if @half_noncentrality.zero?
          return Math.exp(log_regularised_gamma_p(@half_degrees_of_freedom, y))
        end

        Math.exp(log_mixture(y, upper_tail: false))
      end

      def survival_function(value)
        return 1.0 if value <= 0.0

        Math.exp(log_survival_function(value))
      end

      def log_survival_function(value)
        return 0.0 if value <= 0.0

        y = value.to_f / 2.0

        if @half_noncentrality.zero?
          return log_regularised_gamma_q(@half_degrees_of_freedom, y)
        end

        log_mixture(y, upper_tail: true)
      end

      # Quantile function. Returns 0.0 for p == 0 and Float::INFINITY for
      # p == 1 (the distribution support is [0, infinity)); returns nil for
      # probabilities outside [0, 1]. For interior probabilities the root is
      # first bracketed by doubling the search bound and then refined with a
      # safeguarded Illinois iteration; failure to converge raises
      # NotConvergedError, distinguishing algorithmic failure from the
      # endpoint cases above.
      def inverse_cumulative_function(probability)
        return if probability.nil?
        return if probability.is_a?(Float) && probability.nan?
        return if probability < 0.0 || probability > 1.0
        return 0.0 if probability.zero?
        return Float::INFINITY if probability == 1.0

        # Invert on the survival function for the upper half of the
        # distribution: the survival is evaluated directly (never as
        # 1 - CDF), so extreme upper-tail quantiles stay accurate. Note that
        # 1 - p is exact in IEEE arithmetic for p >= 0.5 (Sterbenz lemma).
        if probability <= 0.5
          invert_tail(probability, upper: false)
        else
          invert_tail(1.0 - probability, upper: true)
        end
      end

      private

      # Root finding for tail_value(x) == target, where tail_value is the
      # CDF (upper = false) or the survival function (upper = true). The
      # root is bracketed by doubling the search bound and refined with a
      # safeguarded Illinois iteration (false-position steps with bisection
      # fallback and stagnation protection).
      def invert_tail(target, upper:)
        tail_value = ->(x) { upper ? survival_function(x) : cumulative_function(x) }

        lo = 0.0
        hi = [mean.to_f, 1.0].max
        expansions = 0

        # The CDF increases and the survival decreases with x, so the
        # bracketing condition depends on the tail being inverted.
        past_target = ->(value) { upper ? value <= target : value >= target }
        value_hi = tail_value.call(hi)

        until past_target.call(value_hi)
          lo = hi
          hi *= 2.0
          expansions += 1

          raise NotConvergedError, "could not bracket the quantile for target #{target}" if expansions > 200

          value_hi = tail_value.call(hi)
        end

        return hi if value_hi == target

        flo = tail_value.call(lo) - target
        fhi = value_hi - target

        side = 0
        MAX_INVERSE_ITERATIONS.times do
          x = (lo * fhi - hi * flo) / (fhi - flo)

          # Safeguard: fall back to a bracket-halving step when the
          # false-position step is degenerate or stagnates at one endpoint
          # (which happens when the tail value underflows over a wide
          # range). A geometric mean is used for brackets spanning several
          # orders of magnitude.
          if x.nan? || x <= lo || x >= hi
            x = lo > 0.0 && hi / lo > 4.0 ? Math.sqrt(lo * hi) : 0.5 * (lo + hi)
          elsif lo > 0.0 && hi / lo > 4.0 && Math.log(x / lo) / Math.log(hi / lo) < 0.25
            x = Math.sqrt(lo * hi)
          end

          fx = tail_value.call(x) - target

          return x if fx.zero? || fx.abs <= INVERSE_P_TOLERANCE * target

          if (fx < 0.0) == (flo < 0.0)
            lo = x
            flo = fx
            fhi *= 0.5 if side == -1
            side = -1
          else
            hi = x
            fhi = fx
            flo *= 0.5 if side == 1
            side = 1
          end

          return 0.5 * (lo + hi) if (hi - lo) <= INVERSE_X_TOLERANCE * [hi, 1e-300].max
        end

        raise NotConvergedError, "inverse CDF did not converge for target #{target}"
      end

      # Log of the Poisson mixture
      #
      #   sum_{j=0}^inf w_j * G_j
      #
      # where w_j are Poisson(mu = lambda/2) weights and G_j is the
      # regularised lower (upper_tail = false) or upper (upper_tail = true)
      # incomplete gamma function at (k/2 + j, y). The sum is accumulated in
      # log space via a streaming log-sum-exp, so the result stays
      # representable even when the mixture probability itself would
      # underflow a double.
      #
      # Remainder bounds (valid because 0 <= G_j <= 1):
      #   below j: sum_{i<j} w_i <= w_j * j / (mu - j)        (j < mu)
      #   above J: sum_{j>J} w_j <= w_{J+1} * (J+2)/(J+2-mu)  (J+2 > mu)
      def log_mixture(y, upper_tail:)
        mu = @half_noncentrality
        a0 = @half_degrees_of_freedom
        j0 = mu.floor

        log_w = -mu + j0 * Math.log(mu) - Math.lgamma(j0 + 1.0)[0]

        max_log = -Float::INFINITY
        sum = 0.0

        add_term = lambda do |log_t|
          if log_t > max_log
            sum = max_log == -Float::INFINITY ? 1.0 : sum * Math.exp(max_log - log_t) + 1.0
            max_log = log_t
          else
            sum += Math.exp(log_t - max_log)
          end
        end

        log_partial = -> { max_log + Math.log(sum) }

        # Downward pass: j = j0, j0-1, ..., 0.
        log_wj = log_w
        j = j0

        loop do
          log_g = upper_tail ? log_regularised_gamma_q(a0 + j, y) : log_regularised_gamma_p(a0 + j, y)
          add_term.call(log_wj + log_g)

          break if j.zero?

          if j < mu
            log_remainder = log_wj + Math.log(j) - Math.log(mu - j)
            break if log_remainder <= LOG_SERIES_TOLERANCE + log_partial.call
          end

          log_wj += Math.log(j) - Math.log(mu)
          j -= 1

          raise NotConvergedError, "Poisson mixture did not converge" if j0 - j >= MIXTURE_MAX_TERMS
        end

        # Upward pass: j = j0+1, j0+2, ...
        log_wj = log_w + Math.log(mu) - Math.log(j0 + 1.0)
        j = j0 + 1

        loop do
          log_remainder = log_wj + Math.log(j + 1.0) - Math.log(j + 1.0 - mu)
          break if log_remainder <= LOG_SERIES_TOLERANCE + log_partial.call

          log_g = upper_tail ? log_regularised_gamma_q(a0 + j, y) : log_regularised_gamma_p(a0 + j, y)
          add_term.call(log_wj + log_g)

          j += 1
          log_wj += Math.log(mu) - Math.log(j)

          raise NotConvergedError, "Poisson mixture did not converge" if j - j0 >= MIXTURE_MAX_TERMS
        end

        max_log + Math.log(sum)
      end

      # Log of the regularised lower incomplete gamma P(a, y). The series
      # branch (y < a + 1) is evaluated in log space so the result is exact
      # even when P underflows a double; the complementary branch uses the
      # continued fraction for Q.
      def log_regularised_gamma_p(a, y)
        return -Float::INFINITY if y <= 0.0

        if y < a + 1.0
          log_lower_gamma_series(a, y)
        else
          log1p(-Math.exp(log_upper_gamma_continued_fraction(a, y)))
        end
      end

      # Log of the regularised upper incomplete gamma Q(a, y). The
      # continued-fraction branch (y >= a + 1) is evaluated in log space so
      # the result is exact even when Q underflows a double; the
      # complementary branch uses the series for P.
      def log_regularised_gamma_q(a, y)
        return 0.0 if y <= 0.0

        if y < a + 1.0
          log1p(-Math.exp(log_lower_gamma_series(a, y)))
        else
          log_upper_gamma_continued_fraction(a, y)
        end
      end

      # Accurate log(1 + x) for small |x| (Ruby's Math has no log1p).
      def log1p(x)
        y = 1.0 + x
        return Math.log(y) if y == 1.0

        Math.log(y) * (x / (y - 1.0))
      end

      # Log of P(a, y) via the AS 147 series expansion, evaluated in log
      # space: P(a, y) = y^a * exp(-y) / Gamma(a + 1) * sum_j y^j/((a+1)...(a+j)).
      def log_lower_gamma_series(a, y)
        log_term = a * Math.log(y) - y - Math.lgamma(a + 1.0)[0]
        return -Float::INFINITY if log_term < -745.0

        sum = 1.0
        term = 1.0
        ap = a

        MAX_SERIES_TERMS.times do
          ap += 1.0
          term *= y / ap
          sum += term

          break if term.abs <= GAMMA_EPSILON * sum.abs
        end

        raise NotConvergedError, "lower incomplete gamma series did not converge" if term.abs > GAMMA_EPSILON * sum.abs

        log_term + Math.log(sum)
      end

      # Log of Q(a, y) via Legendre's continued fraction (valid for
      # y >= a + 1), evaluated with the modified Lentz method:
      # Q(a, y) = y^a * exp(-y) / Gamma(a) * h.
      def log_upper_gamma_continued_fraction(a, y)
        tiny = 1.0e-300
        b = y + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d

        (1..MAX_SERIES_TERMS).each do |i|
          an = -i * (i - a)
          b += 2.0

          d = an * d + b
          d = tiny if d.abs < tiny
          c = b + an / c
          c = tiny if c.abs < tiny

          d = 1.0 / d
          delta = d * c
          h *= delta

          if (delta - 1.0).abs < GAMMA_EPSILON
            raise NotConvergedError, "upper incomplete gamma continued fraction not positive" if h <= 0.0

            return -y + a * Math.log(y) - Math.lgamma(a)[0] + Math.log(h)
          end
        end

        raise NotConvergedError, "upper incomplete gamma continued fraction did not converge"
      end
    end
  end
end
