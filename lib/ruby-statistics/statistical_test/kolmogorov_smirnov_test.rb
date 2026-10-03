module RubyStatistics
  module StatisticalTest
    class KolmogorovSmirnovTest
      DEFAULT_EXACT_STATE_BUDGET = 100_000

      # Common alpha, and critical D are calculated following formulas from: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test
      def self.two_samples(group_one:, group_two:, alpha: 0.05, exact: false, alternative: :two_sided, state_budget: DEFAULT_EXACT_STATE_BUDGET)
        return exact_two_samples(
          group_one: group_one,
          group_two: group_two,
          alpha: alpha,
          alternative: alternative,
          state_budget: state_budget
        ) if exact

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

      def self.exact_two_samples(group_one:, group_two:, alpha: 0.05, alternative: :two_sided, state_budget: DEFAULT_EXACT_STATE_BUDGET)
        validate_exact_arguments(group_one, group_two, alternative, state_budget)

        size_one = group_one.size
        size_two = group_two.size
        alternative = normalize_alternative(alternative)
        statistic = exact_statistic(group_one, group_two, alternative)
        critical_d = critical_value(group_one.size, group_two.size, alpha)

        base_result = {
          d_max: statistic[:d_two_sided],
          d_statistic: statistic[:d_statistic],
          d_critical: critical_d,
          total_samples: size_one + size_two,
          alpha: alpha,
          alternative_hypothesis: alternative,
          confidence_level: 1.0 - alpha,
          exact: true
        }

        return completed_result(base_result, 1r) if statistic[:d_statistic].zero?

        frequency = combined_frequency(group_one, group_two)
        tail_count, completed, states_used, required_states, reason = exact_tail_count(
          frequency: frequency,
          size_one: size_one,
          size_two: size_two,
          threshold: statistic[:d_statistic],
          alternative: alternative,
          state_budget: state_budget
        )

        unless completed
          return base_result.merge(
            p_value: nil,
            null: nil,
            alternative: nil,
            completed: false,
            states_used: states_used,
            required_states: required_states,
            state_budget: state_budget,
            reason: reason
          )
        end

        completed_result(
          base_result.merge(states_used: states_used, state_budget: state_budget),
          tail_count.to_r / choose(size_one + size_two, size_one)
        )
      end

      private_class_method def self.validate_exact_arguments(group_one, group_two, alternative, state_budget)
        if group_one.empty? || group_two.empty?
          raise ArgumentError, 'both samples must contain at least one observation'
        end

        normalize_alternative(alternative)

        unless state_budget.is_a?(Integer) && state_budget.positive?
          raise ArgumentError, 'state_budget must be a positive Integer'
        end
      end

      private_class_method def self.normalize_alternative(alternative)
        case alternative
        when :two_sided, :two_tail, :two_tailed, :two_sided_test
          :two_sided
        when :greater, :greater_than, :one_sided_greater
          :greater
        when :less, :less_than, :one_sided_less
          :less
        else
          raise ArgumentError, "unsupported alternative: #{alternative.inspect}"
        end
      end

      private_class_method def self.exact_statistic(group_one, group_two, alternative)
        size_one = group_one.size
        size_two = group_two.size
        frequency = combined_frequency(group_one, group_two)

        cumulative_one = 0
        cumulative_two = 0
        positive = 0
        negative = 0

        frequency.each do |counts|
          cumulative_one += counts[0]
          cumulative_two += counts[1]
          difference = cumulative_one * size_two - cumulative_two * size_one

          positive = [positive, difference].max
          negative = [negative, -difference].max
        end

        d_greater = positive / (size_one * size_two).to_r
        d_less = negative / (size_one * size_two).to_r
        d_two_sided = [d_greater, d_less].max

        d_statistic = case alternative
                      when :greater
                        d_greater
                      when :less
                        d_less
                      else
                        d_two_sided
                      end

        { d_greater: d_greater, d_less: d_less, d_two_sided: d_two_sided, d_statistic: d_statistic }
      end

      private_class_method def self.combined_frequency(group_one, group_two)
        counts = Hash.new { |hash, value| hash[value] = [0, 0] }

        group_one.each { |value| counts[value][0] += 1 }
        group_two.each { |value| counts[value][1] += 1 }

        counts.keys.sort.map { |value| counts[value] }
      end

      private_class_method def self.exact_tail_count(frequency:, size_one:, size_two:, threshold:, alternative:, state_budget:)
        if size_one <= size_two
          tracked_size = size_one
          other_size = size_two
          sign = 1
        else
          tracked_size = size_two
          other_size = size_one
          sign = -1
        end

        limit = ceiling_ratio(
          threshold.numerator * tracked_size * other_size,
          threshold.denominator
        )
        states = { 0 => 1 }
        cumulative = 0
        suffix_total = size_one + size_two
        states_used = 1

        frequency.each do |counts|
          block_size = counts.sum
          cumulative += block_size
          suffix_total -= block_size

          minimum_tracked_count = [0, tracked_size - suffix_total, cumulative - other_size].max
          candidate_minimum = [states.keys.min, minimum_tracked_count].max
          candidate_maximum = [states.keys.max + block_size, tracked_size, cumulative].min
          allowed_minimum, allowed_maximum = allowed_tracked_range(
            cumulative,
            tracked_size,
            other_size,
            suffix_total,
            limit,
            alternative,
            sign
          )
          minimum_next_tracked_count = [candidate_minimum, allowed_minimum].max
          maximum_next_tracked_count = [candidate_maximum, allowed_maximum].min
          reachable_width = [0, maximum_next_tracked_count - minimum_next_tracked_count + 1].max

          if reachable_width.zero?
            return [choose(size_one + size_two, size_one), true, states_used, reachable_width, nil]
          elsif reachable_width > state_budget
            return [
              0,
              false,
              states_used,
              reachable_width,
              "required more than #{state_budget} exact enumeration states"
            ]
          end

          next_states = {}
          block_weights = {}

          states.each do |tracked_count, path_count|
            0.upto(block_size) do |assigned_count|
              next_tracked_count = tracked_count + assigned_count
              next_other_count = cumulative - next_tracked_count

              next unless next_tracked_count >= minimum_tracked_count &&
                next_tracked_count <= tracked_size &&
                next_other_count >= 0 &&
                next_other_count <= other_size
              next if exceeds_alternative_limit?(
                next_tracked_count,
                next_other_count,
                tracked_size,
                other_size,
                limit,
                alternative,
                sign
              )

              weight = block_weights.fetch(assigned_count) do
                block_weights[assigned_count] = choose(block_size, assigned_count)
              end
              next_states[next_tracked_count] = next_states.fetch(next_tracked_count, 0) + path_count * weight
            end
          end

          if next_states.empty?
            return [choose(size_one + size_two, size_one), true, states_used, 0, nil]
          end

          states = next_states
          states_used = [states_used, states.size].max
        end

        non_tail_count = states.fetch(tracked_size, 0)
        [choose(size_one + size_two, size_one) - non_tail_count, true, states_used, states_used, nil]
      end

      private_class_method def self.exceeds_alternative_limit?(tracked_count, other_count, tracked_size, other_size, limit, alternative, sign)
        difference = sign * (tracked_count * other_size - other_count * tracked_size)

        case alternative
        when :greater
          difference >= limit
        when :less
          -difference >= limit
        else
          difference.abs >= limit
        end
      end

      private_class_method def self.ceiling_ratio(numerator, denominator)
        (numerator + denominator - 1) / denominator
      end

      private_class_method def self.allowed_tracked_range(total_count, tracked_size, other_size, suffix_total, limit, alternative, sign)
        total_size = tracked_size + other_size
        minimum = [0, tracked_size - suffix_total, total_count - other_size].max
        maximum = [tracked_size, total_count].min
        upper_bound = (total_count * tracked_size + limit - 1) / total_size
        lower_bound = ceiling_ratio(total_count * tracked_size - limit + 1, total_size)

        case alternative
        when :greater
          if sign == 1
            maximum = [maximum, upper_bound].min
          else
            minimum = [minimum, lower_bound].max
          end
        when :less
          if sign == 1
            minimum = [minimum, lower_bound].max
          else
            maximum = [maximum, upper_bound].min
          end
        else
          minimum = [minimum, lower_bound].max
          maximum = [maximum, upper_bound].min
        end

        [minimum, maximum]
      end

      private_class_method def self.choose(total, selected)
        selected = [selected, total - selected].min
        (1..selected).reduce(1) do |result, value|
          result * (total - selected + value) / value
        end
      end

      private_class_method def self.critical_value(size_one, size_two, alpha)
        common_alpha = Math.sqrt((-0.5 * Math.log(alpha)))
        radicand = (size_one + size_two) / (size_one * size_two).to_r

        common_alpha * Math.sqrt(radicand)
      end

      private_class_method def self.completed_result(result, p_value)
        result.merge(
          p_value: p_value,
          null: result[:alpha] < p_value,
          alternative: p_value <= result[:alpha],
          completed: true,
          reason: nil
        )
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
    end

    KSTest = KolmogorovSmirnovTest # Alias
  end
end
