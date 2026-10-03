module RubyStatistics
  module StatisticalTest
    class KolmogorovSmirnovTest
      DEFAULT_EXACT_STATE_BUDGET = 10_000
      EXACT_ALTERNATIVES = {
        two_sided: :two_sided,
        :'two-sided' => :two_sided,
        :'two.sided' => :two_sided,
        greater: :greater,
        less: :less
      }.freeze

      # Common alpha, and critical D are calculated following formulas from: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
      def self.two_samples(group_one:,
                           group_two:,
                           alpha: 0.05,
                           exact: false,
                           alternative: :two_sided,
                           state_budget: DEFAULT_EXACT_STATE_BUDGET,
                           algorithm: :auto)
        return exact_two_samples(group_one: group_one,
                                 group_two: group_two,
                                 alternative: alternative,
                                 state_budget: state_budget,
                                 algorithm: algorithm) if exact

        samples = group_one + group_two # We can use unbalaced group samples

        ecdf_one = Distribution::Empirical.new(samples: group_one)
        ecdf_two = Distribution::Empirical.new(samples: group_two)

        d_max = samples.sort.map do |sample|
          d1 = ecdf_one.cumulative_function(x: sample)
          d2 = ecdf_two.cumulative_function(x: sample)

          (d1 - d2).abs
        end.max

        # TODO: Validate calculation of Common alpha.
        common_alpha = Math.sqrt((-0.5 * Math.log(alpha)))
        radicand = (group_one.size + group_two.size) / (group_one.size * group_two.size).to_r

        critical_d = common_alpha * Math.sqrt(radicand)
        # critical_d = self.critical_d(alpha: alpha, n: samples.size)

        # We are unable to calculate the p_value, because we don't have the Kolmogorov distribution
        # defined. We reject the null hypotesis if Dmax is > than Dcritical.
        { d_max: d_max,
          d_critical: critical_d,
          total_samples: samples.size,
          alpha: alpha,
          null: d_max <= critical_d,
          alternative: d_max > critical_d,
          confidence_level: 1.0 - alpha }
      end

      def self.exact_two_samples(group_one:,
                                 group_two:,
                                 alternative: :two_sided,
                                 state_budget: DEFAULT_EXACT_STATE_BUDGET,
                                 algorithm: :auto)
        validate_exact_arguments(group_one, group_two, state_budget, algorithm)

        normalized_alternative = normalize_alternative(alternative)
        n_one = group_one.size
        n_two = group_two.size
        total_samples = n_one + n_two
        blocks = tied_blocks(group_one, group_two)
        observed_d = observed_statistic(blocks, n_one, n_two, normalized_alternative)
        selected_algorithm = algorithm == :auto && total_samples > 10 ? :dynamic_programming : algorithm
        selected_algorithm = :enumeration if selected_algorithm == :auto

        favorable_assignments, total_assignments =
          case selected_algorithm
          when :enumeration
            raise ArgumentError, 'enumeration is supported only for at most 10 observations' if total_samples > 10

            enumerate_assignments(blocks, n_one, n_two, observed_d, normalized_alternative)
          when :dynamic_programming
            count_assignments(blocks, n_one, n_two, observed_d, normalized_alternative, state_budget)
          end

        if favorable_assignments.is_a?(Symbol)
          return incomplete_exact_result(observed_d,
                                         total_samples,
                                         normalized_alternative,
                                         state_budget,
                                         total_assignments)
        end

        exact_p_value = Rational(favorable_assignments, total_assignments)

        {
          d_max: observed_d,
          p_value: exact_p_value.to_f,
          p_value_exact: exact_p_value,
          total_samples: total_samples,
          exact: true,
          completed: true,
          algorithm: selected_algorithm,
          alternative_hypothesis: normalized_alternative,
          favorable_assignments: favorable_assignments,
          total_assignments: total_assignments
        }
      end

      # This is an implementation of the formula presented by Paul Molin and Hervé Abdi in a paper,
      # called "New Table and numerical approximations for Kolmogorov-Smirnov / Lilliefors / Van Soest
      # normality test".
      # In this paper, the authors defines a couple of 6th-degree polynomial functions that allow us
      # to find an aproximation of the real critical value. This is based in the conclusions made by
      # Dagnelie (1968), where indicates that critical values given by Lilliefors can be approximated
      # numerically.
      #
      # In general, the formula found is:
      #  C(N, alpha) ^ -2  = A(alpha) * N + B(alpha).
      #
      # Where A(alpha), B(alpha) are two 6th degree polynomial functions computed using the principle
      # of Monte Carlo simulations.
      #
      # paper can be found here: https://utdallas.edu/~herve/MolinAbdi1998-LillieforsTechReport.pdf
      # def self.critical_d(alpha:, n:)
      #   confidence = 1.0 - alpha

      #   a_alpha = 6.32207539843126 -17.1398870006148 * confidence +
      #     38.42812675101057 * (confidence ** 2) - 45.93241384693391 * (confidence ** 3) +
      #     7.88697700041829 * (confidence ** 4) + 29.79317711037858 * (confidence ** 5) -
      #     18.48090137098585 * (confidence ** 6)

      #   b_alpha = 12.940399038404 - 53.458334259532 * confidence +
      #     186.923866119699 * (confidence ** 2) - 410.582178349305 * (confidence ** 3) +
      #     517.377862566267 * (confidence ** 4) - 343.581476222384 * (confidence ** 5) +
      #     92.123451358715 * (confidence ** 6)

      #   Math.sqrt(1.0 / (a_alpha * n + b_alpha))
      # end

      def self.validate_exact_arguments(group_one, group_two, state_budget, algorithm)
        if group_one.empty? || group_two.empty?
          raise ArgumentError, 'both samples must contain at least one observation'
        end

        unless state_budget.is_a?(Integer) && state_budget.positive?
          raise ArgumentError, 'state_budget must be a positive integer'
        end

        return if %i[auto enumeration dynamic_programming].include?(algorithm)

        raise ArgumentError, 'algorithm must be :auto, :enumeration, or :dynamic_programming'
      end
      private_class_method :validate_exact_arguments

      def self.normalize_alternative(alternative)
        normalized = EXACT_ALTERNATIVES[alternative.is_a?(String) ? alternative.to_sym : alternative]
        return normalized if normalized

        raise ArgumentError, 'alternative must be :two_sided, :greater, or :less'
      end
      private_class_method :normalize_alternative

      def self.tied_blocks(group_one, group_two)
        group_one_counts = group_one.each_with_object(Hash.new(0)) { |value, counts| counts[value] += 1 }
        group_two_counts = group_two.each_with_object(Hash.new(0)) { |value, counts| counts[value] += 1 }

        (group_one + group_two).sort.chunk_while { |left, right| left == right }.map do |values|
          value = values.first

          {
            value: value,
            size: values.size,
            group_one: group_one_counts[value],
            group_two: group_two_counts[value]
          }
        end
      end
      private_class_method :tied_blocks

      def self.observed_statistic(blocks, n_one, n_two, alternative)
        cumulative_one = 0
        cumulative_two = 0

        blocks.map do |block|
          cumulative_one += block[:group_one]
          cumulative_two += block[:group_two]
          difference = Rational(cumulative_one, n_one) - Rational(cumulative_two, n_two)

          case alternative
          when :greater
            difference
          when :less
            -difference
          else
            difference.abs
          end
        end.max
      end
      private_class_method :observed_statistic

      def self.enumerate_assignments(blocks, n_one, n_two, threshold, alternative)
        total_samples = n_one + n_two
        selected = Array.new(total_samples, false)
        favorable = 0

        (0...total_samples).to_a.combination(n_one) do |indices|
          indices.each { |index| selected[index] = true }
          cumulative_one = 0
          cumulative_two = 0
          offset = 0
          qualifies = false

          blocks.each do |block|
            group_one_count = 0
            block[:size].times do |index|
              group_one_count += 1 if selected[offset + index]
            end

            cumulative_one += group_one_count
            cumulative_two += block[:size] - group_one_count
            difference = Rational(cumulative_one, n_one) - Rational(cumulative_two, n_two)
            statistic = case alternative
                        when :greater then difference
                        when :less then -difference
                        else difference.abs
                        end
            qualifies ||= statistic >= threshold
            offset += block[:size]
          end

          favorable += 1 if qualifies
          indices.each { |index| selected[index] = false }
        end

        [favorable, binomial_coefficient(total_samples, n_one)]
      end
      private_class_method :enumerate_assignments

      def self.count_assignments(blocks, n_one, n_two, threshold, alternative, state_budget)
        swapped = n_two < n_one
        n_small = swapped ? n_two : n_one
        n_large = swapped ? n_one : n_two
        total_samples = n_small + n_large
        states = { 0 => [1, 0] }
        cumulative_size = 0
        binomial_weights = {}

        blocks.each do |block|
          cumulative_size += block[:size]
          minimum_state = [0, cumulative_size - n_large].max
          maximum_state = [n_small, cumulative_size].min
          required_states = maximum_state - minimum_state + 1

          return [:state_budget_exceeded, required_states] if required_states > state_budget

          new_states = Hash.new { |hash, state| hash[state] = [0, 0] }

          states.each do |previous_state, (active_count, accepted_count)|
            minimum_assigned = [0, minimum_state - previous_state].max
            maximum_assigned = [block[:size], maximum_state - previous_state].min

            (minimum_assigned..maximum_assigned).each do |assigned|
              next_state = previous_state + assigned
              weight = binomial_weights[[block[:size], assigned]] ||= binomial_coefficient(block[:size], assigned)

              if accepted_count.positive?
                new_states[next_state][1] += weight * accepted_count
              end

              next if active_count.zero?

              if path_qualifies?(next_state,
                                 cumulative_size,
                                 n_small,
                                 n_large,
                                 swapped,
                                 threshold,
                                 alternative)
                new_states[next_state][1] += weight * active_count
              else
                new_states[next_state][0] += weight * active_count
              end
            end
          end

          states = new_states
        end

        favorable = states.values.sum { |_active, accepted| accepted }
        [favorable, binomial_coefficient(total_samples, n_small)]
      end
      private_class_method :count_assignments

      def self.path_qualifies?(state, cumulative_size, n_small, n_large, swapped, threshold, alternative)
        difference = if swapped
                       Rational(cumulative_size * n_small - state * (n_small + n_large), n_small * n_large)
                     else
                       Rational(state * (n_small + n_large) - cumulative_size * n_small, n_small * n_large)
                     end

        statistic = case alternative
                    when :greater then difference
                    when :less then -difference
                    else difference.abs
                    end

        statistic >= threshold
      end
      private_class_method :path_qualifies?

      def self.binomial_coefficient(n, k)
        return 0 if k.negative? || k > n

        k = [k, n - k].min
        (1..k).reduce(1) do |coefficient, index|
          coefficient * (n - k + index) / index
        end
      end
      private_class_method :binomial_coefficient

      def self.incomplete_exact_result(observed_d, total_samples, alternative, state_budget, required_states)
        {
          d_max: observed_d,
          total_samples: total_samples,
          exact: true,
          completed: false,
          algorithm: :dynamic_programming,
          alternative_hypothesis: alternative,
          state_budget: state_budget,
          states_required: required_states,
          reason: "exact computation requires at least #{required_states} states; received state_budget #{state_budget}"
        }
      end
      private_class_method :incomplete_exact_result
    end

    KSTest = KolmogorovSmirnovTest # Alias
  end
end
