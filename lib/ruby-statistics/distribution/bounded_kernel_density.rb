module RubyStatistics
  module Distribution
    # Bounded kernel density estimation on a finite interval [lower, upper].
    #
    # A plain Gaussian KDE leaks probability mass outside the support of
    # bounded samples. This estimator uses the reflection (method of images)
    # technique: every observation x generates mirror images at
    # x + 2*k*L and 2*lower - x + 2*k*L for every integer k, where
    # L = upper - lower. Summing Gaussian kernels centred on the mirrors and
    # renormalising by the mass falling inside [lower, upper] gives a density
    # that integrates to exactly one on the interval and is zero outside it.
    #
    # The infinite mirror series is truncated to |k| <= reflections and the
    # omitted tail is bounded explicitly (see #truncation_error), so the
    # truncation error is quantified instead of silently ignored.
    #
    # The bandwidth can be given directly, or selected by maximising the
    # leave-one-out log-likelihood over a caller supplied positive interval
    # (bandwidth: :loo together with bandwidth_interval:). Each left-out
    # observation is removed together with *all* of its mirror images, so no
    # information about the held-out point leaks into its own estimate.
    # Duplicate observations, observations exactly on the boundaries and
    # optima lying at the edges of the search interval are reported through
    # the returned diagnostics; an interior unique optimum is not guaranteed
    # to exist.
    class BoundedKernelDensity
      DEFAULT_TOLERANCE = 1e-10
      MAX_REFLECTIONS = 1_000
      GOLDEN_RATIO = (Math.sqrt(5.0) - 1.0) / 2.0
      SQRT_2_PI = Math.sqrt(2.0 * Math::PI)
      EDGE_FRACTION = 0.02

      attr_reader :data, :lower, :upper, :bandwidth, :reflections,
                  :tolerance, :normalization, :bandwidth_selection

      def initialize(data, lower:, upper:, bandwidth: nil, bandwidth_interval: nil,
                     reflections: nil, tolerance: DEFAULT_TOLERANCE)
        raise ArgumentError, 'lower must be strictly less than upper' unless lower < upper

        @data = data.map(&:to_f)
        raise ArgumentError, 'at least one observation is required' if @data.empty?

        @lower = lower.to_f
        @upper = upper.to_f
        @tolerance = tolerance.to_f

        outside = @data.reject { |value| value >= @lower && value <= @upper }
        unless outside.empty?
          raise ArgumentError, "observations outside [lower, upper]: " + outside.inspect
        end

        if bandwidth == :loo
          if bandwidth_interval.nil?
            raise ArgumentError, 'bandwidth_interval is required when bandwidth is :loo'
          end

          @bandwidth_selection = self.class.select_bandwidth(
            @data, lower: @lower, upper: @upper,
            interval: bandwidth_interval, reflections: reflections, tolerance: @tolerance
          )
          @bandwidth = @bandwidth_selection.bandwidth
        elsif bandwidth.nil?
          raise ArgumentError, 'provide a numeric bandwidth or bandwidth: :loo'
        else
          @bandwidth = bandwidth.to_f
          raise ArgumentError, 'bandwidth must be positive' unless @bandwidth.positive?
        end

        @reflections = reflections || self.class.reflection_count(@bandwidth, length, @tolerance)
        @observation_images = @data.map do |value|
          self.class.images(value, @lower, length, @reflections)
        end
        @normalization = average_mass(@bandwidth)
      end

      def length
        upper - lower
      end

      # Density at +value+. Zero outside [lower, upper].
      def density_function(value)
        return 0.0 if value < lower || value > upper

        kernel_sum(value) / (data.size * normalization)
      end

      # Cumulative probability up to +value+. Exactly 0 below +lower+ and
      # exactly 1 above +upper+.
      def cumulative_function(value)
        return 0.0 if value < lower
        return 1.0 if value > upper

        mass_sum(value) / (data.size * normalization)
      end

      # Probability of the interval [from, to], clipped to the support.
      def probability(from, to)
        return 0.0 if from >= to

        cumulative_function([to, upper].min) - cumulative_function([from, lower].max)
      end

      # Explicit bounds on the error introduced by truncating the mirror
      # series at |k| <= reflections.
      #
      # The omitted mirrors contribute a non-negative density of at most
      # +omitted_density+ per observation anywhere on the interval, and at
      # most +omitted_mass+ of extra mass. Writing f for the returned
      # density, f* for the infinite-mirror density, Z for +normalization+
      # and (e_d, e_m) for the two tails, the errors propagate as:
      #
      #   |f*(x) - f(x)| <= (e_d + f(x) * e_m) / Z          (:density)
      #   |F*(x) - F(x)| <= (e_m + F(x) * e_m) / Z
      #                    <= 2 * e_m / Z                    (:cumulative)
      #
      # Interval probabilities inherit twice the cumulative bound.
      def truncation_error
        density_tail = self.class.truncation_density_bound(bandwidth, length, reflections)
        mass_tail = density_tail * length
        max_density = 2 * (2 * reflections + 1) *
                      self.class.gaussian_pdf(0.0) / (bandwidth * normalization)

        {
          omitted_density: density_tail,
          omitted_mass: mass_tail,
          normalization: normalization,
          density: (density_tail + max_density * mass_tail) / normalization,
          cumulative: 2.0 * mass_tail / normalization
        }
      end

      def diagnostics
        {
          sample_size: data.size,
          duplicates: data.tally.select { |_, count| count > 1 },
          boundary_points: data.select { |value| value == lower || value == upper }.uniq,
          bandwidth: bandwidth,
          reflections: reflections,
          bandwidth_selection: bandwidth_selection,
          warnings: warnings
        }
      end

      def warnings
        messages = []

        if data.tally.values.any? { |count| count > 1 }
          messages << 'duplicate observations detected; ties inflate the '                       'leave-one-out likelihood at small bandwidths'
        end

        if data.any? { |value| value == lower || value == upper }
          messages << 'observations exactly on the interval boundary; '                       'their kernels are half reflected by construction'
        end

        messages.concat(bandwidth_selection.warnings) if bandwidth_selection
        messages
      end

      # Mirror images of +value+: the value itself and its reflections on
      # both boundaries, unfolded onto two lattices with spacing 2*length.
      def self.images(value, lower, length, reflections)
        reflected = 2.0 * lower - value

        (-reflections..reflections).flat_map do |k|
          shift = 2.0 * k * length
          [value + shift, reflected + shift]
        end
      end

      # Smallest number of reflections whose omitted mirror tail is bounded
      # by +tolerance+ (density units, per observation).
      def self.reflection_count(bandwidth, length, tolerance)
        (1..MAX_REFLECTIONS).each do |k|
          return k if truncation_density_bound(bandwidth, length, k) <= tolerance
        end

        raise ArgumentError,
              'cannot reach tolerance with the maximum number of reflections; '               'increase the tolerance or use a smaller bandwidth'
      end

      # Upper bound on the density contributed anywhere on [lower, upper] by
      # the omitted mirrors (|k| > k) of a single observation.
      def self.truncation_density_bound(bandwidth, length, k)
        return Float::INFINITY if k < 1

        lattice_tail(bandwidth, length, (2 * k + 1) * length) +
          lattice_tail(bandwidth, length, 2 * k * length)
      end

      # The omitted mirrors of one observation sit on two interleaved
      # lattices spaced 2*length apart. For any point inside the interval
      # the omitted mirrors of a lattice are at distance at least +distance+
      # and occur every 2*length on both sides, so their total contribution
      # is bounded by a geometric series with ratio
      # exp(-2*length*(distance + length)/bandwidth^2).
      def self.lattice_tail(bandwidth, length, distance)
        ratio = Math.exp(-2.0 * length * (distance + length) / (bandwidth * bandwidth))
        2.0 * gaussian_pdf(distance / bandwidth) / bandwidth / (1.0 - ratio)
      end

      # Leave-one-out log-likelihood of +bandwidth+ for +data+ on
      # [lower, upper]. Observation i is scored with the density built from
      # the remaining observations only; all of its own mirror images are
      # removed as well.
      def self.leave_one_out_log_likelihood(data, lower:, upper:, bandwidth:,
                                            reflections: nil, tolerance: DEFAULT_TOLERANCE)
        values = data.map(&:to_f)
        k = reflections || reflection_count(bandwidth, upper - lower, tolerance)
        observation_images = values.map { |value| images(value, lower, upper - lower, k) }

        leave_one_out_score(values, observation_images, lower, upper, bandwidth)
      end

      # Maximises the leave-one-out log-likelihood over the positive
      # interval +interval+ using a golden-section search. Returns a
      # BandwidthSelection describing the optimum and any caveats; it does
      # not claim that a unique interior optimum exists.
      def self.select_bandwidth(data, lower:, upper:, interval:,
                                reflections: nil, tolerance: DEFAULT_TOLERANCE)
        lo, hi = interval.map(&:to_f)
        unless lo.positive? && hi > lo
          raise ArgumentError, 'bandwidth interval must satisfy 0 < min < max'
        end

        values = data.map(&:to_f)
        if values.size < 2
          raise ArgumentError, 'leave-one-out selection requires at least two observations'
        end

        length = upper - lower
        # The mirror count must be valid for every bandwidth in the search
        # interval; the largest bandwidth needs the most reflections.
        k = reflections || reflection_count(hi, length, tolerance)
        observation_images = values.map { |value| images(value, lower, length, k) }

        evaluations = 0
        objective = lambda do |h|
          evaluations += 1
          leave_one_out_score(values, observation_images, lower, upper, h)
        end

        a, b = lo, hi
        c = b - GOLDEN_RATIO * (b - a)
        d = a + GOLDEN_RATIO * (b - a)
        fc = objective.call(c)
        fd = objective.call(d)

        iterations = 0
        while (b - a) > 1e-10 * (a + b) && iterations < 500
          iterations += 1
          if fc < fd
            a = c
            c = d
            fc = fd
            d = a + GOLDEN_RATIO * (b - a)
            fd = objective.call(d)
          else
            b = d
            d = c
            fd = fc
            c = b - GOLDEN_RATIO * (b - a)
            fc = objective.call(c)
          end
        end

        best = 0.5 * (a + b)
        best_score = objective.call(best)

        # Golden-section search assumes unimodality; compare against the
        # interval edges as well so a monotone or boundary optimum is not
        # mistaken for an interior one.
        lo_score = objective.call(lo)
        hi_score = objective.call(hi)
        if lo_score >= best_score && lo_score >= hi_score
          best = lo
          best_score = lo_score
        elsif hi_score >= best_score
          best = hi
          best_score = hi_score
        end

        duplicates = values.tally.select { |_, count| count > 1 }
        boundary_points = values.select { |value| value == lower || value == upper }.uniq

        width = hi - lo
        at_lower = (best - lo) <= EDGE_FRACTION * width
        at_upper = (hi - best) <= EDGE_FRACTION * width

        warnings = []
        if at_lower
          warnings << 'optimum lies at the lower edge of the search interval; '                       'a smaller bandwidth may score higher, so no interior '                       'optimum was found'
        end
        if at_upper
          warnings << 'optimum lies at the upper edge of the search interval; '                       'widen the interval to confirm an interior optimum'
        end
        unless duplicates.empty?
          warnings << 'duplicate observations present; ties inflate the '                       'leave-one-out likelihood at small bandwidths and can '                       'prevent a unique interior optimum'
        end
        unless boundary_points.empty?
          warnings << 'observations exactly on the interval boundary affect '                       'the leave-one-out score'
        end

        BandwidthSelection.new(
          bandwidth: best,
          log_likelihood: best_score,
          interval: [lo, hi],
          duplicates: duplicates,
          boundary_points: boundary_points,
          warnings: warnings,
          evaluations: evaluations,
          at_lower_bound: at_lower,
          at_upper_bound: at_upper
        )
      end

      def self.gaussian_pdf(z)
        Math.exp(-0.5 * z * z) / SQRT_2_PI
      end

      def self.gaussian_cdf(z)
        0.5 * (1.0 + Math.erf(z / Math.sqrt(2.0)))
      end

      def self.leave_one_out_score(values, observation_images, lower, upper, bandwidth)
        masses = observation_images.map do |images|
          images.sum do |y|
            gaussian_cdf((upper - y) / bandwidth) - gaussian_cdf((lower - y) / bandwidth)
          end
        end
        total_mass = masses.sum

        values.each_with_index.sum do |value, i|
          numerator = 0.0
          observation_images.each_with_index do |images, j|
            next if i == j

            numerator += images.sum { |y| gaussian_pdf((value - y) / bandwidth) / bandwidth }
          end

          density = numerator / (total_mass - masses[i])
          density.positive? ? Math.log(density) : -Float::INFINITY
        end
      end
      private_class_method :leave_one_out_score

      # Result of a leave-one-out bandwidth search.
      class BandwidthSelection
        attr_reader :bandwidth, :log_likelihood, :interval, :duplicates,
                    :boundary_points, :warnings, :evaluations

        def initialize(bandwidth:, log_likelihood:, interval:, duplicates:,
                       boundary_points:, warnings:, evaluations:,
                       at_lower_bound:, at_upper_bound:)
          @bandwidth = bandwidth
          @log_likelihood = log_likelihood
          @interval = interval
          @duplicates = duplicates
          @boundary_points = boundary_points
          @warnings = warnings
          @evaluations = evaluations
          @at_lower_bound = at_lower_bound
          @at_upper_bound = at_upper_bound
        end

        def at_lower_bound?
          @at_lower_bound
        end

        def at_upper_bound?
          @at_upper_bound
        end

        # False when the optimum sits at (or next to) an edge of the search
        # interval, in which case no interior optimum was located.
        def interior_optimum?
          !@at_lower_bound && !@at_upper_bound
        end
      end

      private

      def kernel_sum(value)
        h = bandwidth
        @observation_images.sum do |images|
          images.sum { |y| self.class.gaussian_pdf((value - y) / h) / h }
        end
      end

      def mass_sum(value)
        h = bandwidth
        @observation_images.sum do |images|
          images.sum do |y|
            self.class.gaussian_cdf((value - y) / h) -
              self.class.gaussian_cdf((lower - y) / h)
          end
        end
      end

      def average_mass(h)
        total = @observation_images.sum do |images|
          images.sum do |y|
            self.class.gaussian_cdf((upper - y) / h) -
              self.class.gaussian_cdf((lower - y) / h)
          end
        end
        total / data.size
      end
    end
  end
end
