module RubyStatistics
  module StatisticalTest
    # Wilcoxon signed-rank test for paired (dependent) samples.
    #
    # Unlike WilcoxonRankSumTest (two independent groups), this test works on the
    # paired differences d_i = x_i - y_i measured on the same subjects (e.g.
    # before/after), or on a single sample of differences passed directly.
    #
    # Zero-difference handling (+zero_method+), applied consistently to the test
    # statistic, the Hodges-Lehmann estimate and the acceptance set:
    # * :wilcoxon - zero differences are discarded before ranking.
    # * :pratt    - zero differences are kept while ranking |d_i| (they occupy
    #               the lowest rank positions) but contribute nothing to the
    #               statistic or to the null distribution.
    # Consequently the Walsh averages used for the estimate are computed from the
    # non-zero differences only with :wilcoxon, and from all differences with
    # :pratt. If every difference is zero, the statistic is 0, the p-value is 1
    # and the estimate is 0.0.
    #
    # Ties: tied absolute values receive midranks (average ranks), which may be
    # half-integers. For the exact null distribution the midranks are scaled by
    # 2 so the subset-sum dynamic program runs on integers; the distribution is
    # always built from the actual (possibly tied) ranks, never from the no-ties
    # sign-rank table.
    #
    # Exact p-values: the 2^n sign assignments are enumerated via dynamic
    # programming. One-sided p-values are P(W+ <= w) (:left_tail) and
    # P(W+ >= w) (:right_tail); the two-sided p-value is
    # min(1, 2 * min(less, greater)). With method: :normal (or automatically for
    # n > EXACT_N_LIMIT) a tie-corrected normal approximation with continuity
    # correction is used instead.
    #
    # Location-shift inference: the Hodges-Lehmann estimate is the median of the
    # Walsh averages (d_i + d_j)/2, i <= j. The acceptance set is obtained by
    # inverting the very same test: p(mu) only changes at the Walsh averages, so
    # every open interval between consecutive Walsh averages and every Walsh
    # average itself is evaluated, and accepted pieces are merged into
    # components. +acceptance_set+ is the exact list of those components.
    # +confidence_interval+ is the interval envelope [min lower, max upper] of
    # the acceptance set: when the acceptance set has more than one component
    # (flagged by +acceptance_gaps+) the envelope contains small rejected gaps,
    # so it is conservative - its true coverage is at least +confidence_level+.
    class WilcoxonSignedRankTest
      # Maximum number of non-zero differences for the exact p-value when
      # method: :auto is used.
      EXACT_N_LIMIT = 50

      # Maximum sample size for exact dynamic programming inside the acceptance
      # set inversion when method: :auto is used. Larger samples invert the
      # normal approximation instead.
      EXACT_INVERSION_N_LIMIT = 25

      VALID_TAILS = %i[two_tail left_tail right_tail].freeze
      VALID_ZERO_METHODS = %i[wilcoxon pratt].freeze
      VALID_METHODS = %i[auto exact normal].freeze

      # sample_two may be nil, in which case sample_one is treated as the
      # sample of differences (one-sample mode).
      def perform(alpha, tails, sample_one, sample_two = nil, mu: 0, zero_method: :wilcoxon, method: :auto)
        validate!(tails, zero_method, method)

        differences = differences_for(sample_one, sample_two)
        if differences.empty?
          raise ArgumentError, 'at least one paired observation is required.'
        end

        core = signed_rank_core(differences.map { |d| d - mu }, zero_method)
        use_exact = exact_test?(method, core[:n])
        p_values = use_exact ? exact_p_values(core) : normal_p_values(core)
        p_value = select_p_value(p_values, tails)

        estimation_pool = zero_method == :pratt ? differences : differences.reject(&:zero?)
        estimate = hodges_lehmann_estimate(estimation_pool)

        inversion_exact = method == :exact ||
                          (method != :normal && differences.size <= EXACT_INVERSION_N_LIMIT)
        components = acceptance_set(alpha, tails, differences, zero_method, inversion_exact)
        envelope = components.empty? ? nil : [components.first[:lower], components.last[:upper]]

        { statistic: core[:w_plus],
          w_minus: core[:w_minus],
          n: core[:n],
          mu: mu,
          tails: tails,
          zero_method: zero_method,
          exact: use_exact,
          z: core[:n].positive? ? core[:z] : nil,
          p_value: p_value,
          p_value_less: p_values[:less],
          p_value_greater: p_values[:greater],
          estimate: estimate,
          acceptance_set: components,
          acceptance_gaps: components.size > 1,
          acceptance_exact: inversion_exact,
          confidence_interval: envelope,
          alpha: alpha,
          null: alpha < p_value,
          alternative: p_value <= alpha,
          confidence_level: 1 - alpha }
      end

      private def validate!(tails, zero_method, method)
        unless VALID_TAILS.include?(tails)
          raise ArgumentError, "tails must be one of #{VALID_TAILS.inspect}."
        end

        unless VALID_ZERO_METHODS.include?(zero_method)
          raise ArgumentError, "zero_method must be one of #{VALID_ZERO_METHODS.inspect}."
        end

        unless VALID_METHODS.include?(method)
          raise ArgumentError, "method must be one of #{VALID_METHODS.inspect}."
        end
      end

      private def differences_for(sample_one, sample_two)
        one = sample_one.to_a
        return one if sample_two.nil?

        two = sample_two.to_a
        unless one.size == two.size
          raise ArgumentError, 'paired samples must have the same size.'
        end

        one.each_with_index.map { |value, index| value - two[index] }
      end

      # Ranks |d_i| with midranks for ties and applies the zero-difference rule.
      # Returns the statistic, the effective items (non-zero differences with
      # their ranks) and the normal-approximation moments.
      private def signed_rank_core(differences, zero_method)
        absolute = differences.map(&:abs)

        items = if zero_method == :pratt
                  ranks = midranks(absolute)
                  differences.each_index
                             .reject { |i| differences[i].zero? }
                             .map { |i| [differences[i], ranks[i]] }
                else
                  kept = differences.each_index.reject { |i| differences[i].zero? }
                  ranks = midranks(kept.map { |i| absolute[i] })
                  kept.each_with_index.map { |index, k| [differences[index], ranks[k]] }
                end

        ranks = items.map { |_, rank| rank }
        w_plus = items.select { |difference, _| difference.positive? }
                      .reduce(0.0) { |memo, (_, rank)| memo + rank }
        rank_sum = ranks.reduce(0.0, :+)

        mean = rank_sum / 2.0
        variance = ranks.reduce(0.0) { |memo, rank| memo + (rank * rank) } / 4.0
        standard_deviation = Math.sqrt(variance)

        { items: items,
          n: items.size,
          ranks: ranks,
          w_plus: w_plus,
          w_minus: rank_sum - w_plus,
          z: standard_deviation.positive? ? (w_plus - mean) / standard_deviation : 0.0,
          mean: mean,
          standard_deviation: standard_deviation }
      end

      # Average ranks (1-based) of the given values; ties share their midrank.
      private def midranks(values)
        order = values.each_index.sort_by { |i| values[i] }
        ranks = Array.new(values.size)

        index = 0
        while index < values.size
          upper = index
          upper += 1 while upper < values.size && values[order[upper]] == values[order[index]]
          midrank = (index + 1 + upper) / 2.0
          (index...upper).each { |k| ranks[order[k]] = midrank }
          index = upper
        end

        ranks
      end

      private def exact_test?(method, n)
        case method
        when :exact then true
        when :normal then false
        else n <= EXACT_N_LIMIT
        end
      end

      # Exact null distribution of W+ via subset-sum dynamic programming.
      # Midranks are scaled by 2 so half-integer ranks become integers; the
      # counts are therefore always computed from the actual tied ranks and
      # never from a no-ties table.
      private def exact_p_values(core)
        scaled_ranks = core[:ranks].map { |rank| (rank * 2).round }
        total = scaled_ranks.reduce(0, :+)

        counts = Array.new(total + 1, 0)
        counts[0] = 1
        scaled_ranks.each do |rank|
          next if rank.zero?

          total.downto(rank) { |sum| counts[sum] += counts[sum - rank] }
        end

        assignments = 2**core[:n]
        observed = (core[:w_plus] * 2).round

        less = counts[0..observed].reduce(0, :+) / assignments.to_f
        greater = counts[observed..-1].reduce(0, :+) / assignments.to_f

        { less: less, greater: greater, two: [1.0, 2 * [less, greater].min].min }
      end

      # Tie-corrected normal approximation with continuity correction.
      private def normal_p_values(core)
        return { less: 1.0, greater: 1.0, two: 1.0 } if core[:n].zero?

        normal = Distribution::StandardNormal.new
        w = core[:w_plus]
        mean = core[:mean]
        sd = core[:standard_deviation]

        less = normal.cumulative_function((w + 0.5 - mean) / sd)
        greater = 1 - normal.cumulative_function((w - 0.5 - mean) / sd)
        z_two = [(w - mean).abs - 0.5, 0].max / sd
        two = [1.0, 2 * (1 - normal.cumulative_function(z_two))].min

        { less: less, greater: greater, two: two }
      end

      private def select_p_value(p_values, tails)
        case tails
        when :left_tail then p_values[:less]
        when :right_tail then p_values[:greater]
        else p_values[:two]
        end
      end

      # Walsh averages (d_i + d_j)/2 for i <= j.
      private def walsh_averages(differences)
        sorted = differences.sort
        averages = []
        sorted.each_with_index do |value, i|
          (i...sorted.size).each { |j| averages << (value + sorted[j]) / 2.0 }
        end
        averages
      end

      # Hodges-Lehmann estimate: median of the Walsh averages. The pool already
      # reflects the zero-difference convention; an empty pool (all differences
      # zero under :wilcoxon) estimates the location as 0.0.
      private def hodges_lehmann_estimate(pool)
        return 0.0 if pool.empty?

        averages = walsh_averages(pool).sort
        size = averages.size
        middle = size / 2

        if size.odd?
          averages[middle]
        else
          (averages[middle - 1] + averages[middle]) / 2.0
        end
      end

      # Inverts the test: p(mu) is constant between consecutive Walsh averages
      # of the full difference sample, so every open interval between them and
      # every Walsh average itself is evaluated exactly once. Accepted pieces
      # are merged into maximal connected components.
      private def acceptance_set(alpha, tails, differences, zero_method, use_exact)
        points = walsh_averages(differences).uniq.sort

        probes = [points.first - 1.0]
        points.each_cons(2) { |a, b| probes << (a + b) / 2.0 }
        probes << points.last + 1.0

        region_accepted = probes.map do |mu|
          accepted?(alpha, tails, differences, mu, zero_method, use_exact)
        end
        point_accepted = points.map do |mu|
          accepted?(alpha, tails, differences, mu, zero_method, use_exact)
        end

        components = []
        open_component = nil
        close = lambda do
          components << open_component unless open_component.nil?
          open_component = nil
        end

        bounds = [[-Float::INFINITY, points.first]] +
                 points.each_cons(2).to_a +
                 [[points.last, Float::INFINITY]]

        region_accepted.each_with_index do |accepted, index|
          if accepted
            if open_component.nil?
              open_component = { lower: bounds[index][0], upper: bounds[index][1],
                                 include_lower: false, include_upper: false }
            else
              open_component[:upper] = bounds[index][1]
              open_component[:include_upper] = false
            end
          else
            close.call
          end

          next if index >= points.size

          if point_accepted[index]
            if open_component.nil?
              open_component = { lower: points[index], upper: points[index],
                                 include_lower: true, include_upper: true }
            else
              open_component[:include_upper] = true
            end
          else
            close.call
          end
        end
        close.call

        components
      end

      private def accepted?(alpha, tails, differences, mu, zero_method, use_exact)
        core = signed_rank_core(differences.map { |d| d - mu }, zero_method)
        p_values = use_exact ? exact_p_values(core) : normal_p_values(core)

        select_p_value(p_values, tails) > alpha
      end
    end
  end
end
