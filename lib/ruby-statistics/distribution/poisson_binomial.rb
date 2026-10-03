module RubyStatistics
  module Distribution
    # Distribution of the number of successes from independent Bernoulli trials
    # with potentially different success probabilities.
    #
    # Certain 0 and 1 probabilities only shift the feasible support. Remaining
    # probabilities are convolved in log space, so tiny probabilities do not
    # underflow inside the recurrence. CDFs and survival probabilities are
    # summed as the corresponding left and right tails; the survival tail is
    # never calculated as 1 - CDF. Log probabilities are available when a tail
    # is smaller than the representable Float range after exponentiation.
    #
    # Time complexity is O(m^2) and memory complexity is O(m), where m is the
    # number of non-deterministic probabilities. This exact dynamic program is
    # practical for thousands of trials; much larger arrays need an
    # approximation such as a Fourier or saddle-point method.
    class PoissonBinomial
      attr_reader :probabilities, :number_of_trials, :certain_successes, :certain_failures

      def initialize(probabilities)
        unless probabilities.respond_to?(:map)
          raise ArgumentError, 'probabilities must be an array-like collection'
        end

        @probabilities = probabilities.map do |probability|
          unless probability.is_a?(Numeric) && probability.real? && probability >= 0 && probability <= 1
            raise ArgumentError, 'all probabilities must be finite numbers between 0 and 1'
          end

          probability.to_f
        end.freeze

        @number_of_trials = @probabilities.size
        @certain_successes = @probabilities.count(1.0)
        @certain_failures = @probabilities.count(0.0)
        @minimum = @certain_successes
        @maximum = number_of_trials - @certain_failures
      end

      def probability_mass_function(k)
        log_probability = log_probability_mass_function(k)
        return if log_probability.nil?

        Math.exp(log_probability)
      end

      alias_method :pmf, :probability_mass_function

      def log_probability_mass_function(k)
        return unless integer_number?(k)

        k = k.to_i
        return if k < 0 || k > number_of_trials
        return -Float::INFINITY if k < minimum || k > maximum

        log_probabilities[k - minimum]
      end

      alias_method :log_pmf, :log_probability_mass_function

      def cumulative_function(k)
        return unless k.is_a?(Numeric) && k.real?
        return 0.0 if k < minimum
        return 1.0 if k >= maximum

        upper_index = [k.floor - minimum, variable_count - 1].min
        Math.exp(log_sum_exp(log_probabilities[0..upper_index]))
      end

      alias_method :cdf, :cumulative_function

      def log_cumulative_function(k)
        return unless k.is_a?(Numeric) && k.real?
        return -Float::INFINITY if k < minimum
        return 0.0 if k >= maximum

        upper_index = [k.floor - minimum, variable_count - 1].min
        log_sum_exp(log_probabilities[0..upper_index])
      end

      alias_method :log_cdf, :log_cumulative_function

      # Survival is P(X > k). At least-k probability is survival_function(k - 1).
      def survival_function(k)
        return unless k.is_a?(Numeric) && k.real?
        return 1.0 if k < minimum
        return 0.0 if k >= maximum

        lower_index = [k.floor + 1 - minimum, 0].max
        Math.exp(log_sum_exp(log_probabilities[lower_index..]))
      end

      alias_method :ccdf, :survival_function
      alias_method :survival, :survival_function

      def log_survival_function(k)
        return unless k.is_a?(Numeric) && k.real?
        return 0.0 if k < minimum
        return -Float::INFINITY if k >= maximum

        lower_index = [k.floor + 1 - minimum, 0].max
        log_sum_exp(log_probabilities[lower_index..])
      end

      alias_method :log_ccdf, :log_survival_function
      alias_method :log_survival, :log_survival_function

      # Smallest integer k for which P(X <= k) is at least probability.
      def inverse_cumulative_function(probability)
        return unless probability.is_a?(Numeric) && probability.real?
        return if probability < 0 || probability > 1
        return minimum if probability.zero?
        return maximum if probability == 1

        cumulative_log_probability = -Float::INFINITY
        log_probabilities.each_with_index do |log_probability, index|
          cumulative_log_probability = log_add_exp(cumulative_log_probability, log_probability)
          return minimum + index if Math.exp(cumulative_log_probability) >= probability
        end

        maximum
      end

      alias_method :quantile, :inverse_cumulative_function
      alias_method :discrete_quantile, :inverse_cumulative_function

      def mean
        probabilities.sum
      end

      def variance
        probabilities.sum { |probability| probability * (1.0 - probability) }
      end

      private

      attr_reader :minimum, :maximum

      def variable_count
        probabilities.size - certain_successes - certain_failures
      end

      def log_probabilities
        @log_probabilities ||= begin
          values = [0.0]

          probabilities.each do |probability|
            next if probability.zero? || probability == 1.0

            log_success = Math.log(probability)
            log_failure = Math.log(1.0 - probability)
            next_values = Array.new(values.size + 1, -Float::INFINITY)

            values.each_with_index do |log_probability, successes|
              next_values[successes] = log_add_exp(next_values[successes], log_probability + log_failure)
              next_values[successes + 1] = log_add_exp(next_values[successes + 1], log_probability + log_success)
            end

            values = next_values
          end

          values
        end
      end

      def integer_number?(value)
        value.is_a?(Numeric) && value.real? && value == value.to_i
      end

      def log_sum_exp(log_values)
        log_values.reduce(-Float::INFINITY) do |result, log_value|
          log_add_exp(result, log_value)
        end
      end

      def log_add_exp(left, right)
        return right if left == -Float::INFINITY
        return left if right == -Float::INFINITY

        larger = [left, right].max
        larger + Math.log(Math.exp(left - larger) + Math.exp(right - larger))
      end
    end
  end
end
