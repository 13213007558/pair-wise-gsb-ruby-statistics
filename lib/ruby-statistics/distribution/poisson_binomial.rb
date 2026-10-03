module RubyStatistics
  module Distribution
    # Distribution of the number of successes from independent Bernoulli
    # trials that may have distinct success probabilities.
    #
    # Deterministic probabilities (0 and 1) are handled as a shift of the
    # support, while the remaining probabilities use an O(n^2) dynamic
    # program in log space. The PMF, CDF and right tail each use the
    # directly accumulated mass, so tiny tails remain available through the
    # log methods even when Float probability results underflow to zero.
    # The direct implementation is practical for hundreds to a few thousand
    # trials; much larger n should use an approximation such as FFT.
    class PoissonBinomial
      attr_reader :probabilities, :number_of_trials

      def initialize(*probabilities)
        probabilities = probabilities.first if probabilities.one? && probabilities.first.respond_to?(:to_ary)

        @probabilities = probabilities.map do |probability|
          unless probability.is_a?(Numeric) && probability.finite? && probability >= 0 && probability <= 1
            raise ArgumentError, "all probabilities must be finite Numeric values in [0, 1]"
          end

          probability.to_f
        end.freeze

        @number_of_trials = @probabilities.size
        @certain_successes = @probabilities.count { |probability| probability == 1.0 }
        uncertain_probabilities = @probabilities.reject { |probability| probability.zero? || probability == 1.0 }

        @mean = @probabilities.sum
        @variance = @probabilities.sum { |probability| probability * (1.0 - probability) }
        @log_masses = Array.new(@number_of_trials + 1, -Float::INFINITY)

        calculate_log_probabilities(uncertain_probabilities).each_with_index do |log_probability, index|
          @log_masses[@certain_successes + index] = log_probability
        end
        calculate_log_tails
      end

      def probability_mass_function(k)
        log_probability = log_probability_mass_function(k)
        return if log_probability.nil?

        Math.exp(log_probability)
      end
      alias pmf probability_mass_function

      def log_probability_mass_function(k)
        return unless valid_integer?(k) && k >= 0 && k <= number_of_trials

        @log_masses[k.to_i]
      end
      alias log_pmf log_probability_mass_function

      def cumulative_function(k)
        log_probability = log_cumulative_function(k)
        return if log_probability.nil?

        normalize_probability(Math.exp(log_probability))
      end

      def log_cumulative_function(k)
        return unless valid_integer?(k)
        return -Float::INFINITY if k < 0
        return 0.0 if k >= number_of_trials

        @log_cdf[k.to_i]
      end

      def survival_function(k)
        log_probability = log_survival_function(k)
        return if log_probability.nil?

        normalize_probability(Math.exp(log_probability))
      end
      alias survival survival_function

      def log_survival_function(k)
        return unless valid_integer?(k)
        return 0.0 if k < 0
        return -Float::INFINITY if k >= number_of_trials

        @log_survival[k.to_i]
      end
      alias log_survival log_survival_function

      def quantile(probability)
        unless probability.is_a?(Numeric) && probability.finite? && probability >= 0 && probability <= 1
          raise ArgumentError, "probability must be a finite Numeric value in [0, 1]"
        end

        return 0 if probability.zero?
        return number_of_trials if probability == 1

        converted_probability = probability.to_f
        target_log_probability = if converted_probability.zero?
                                   -Float::INFINITY
                                 else
                                   Math.log(converted_probability)
                                 end
        lower = 0
        upper = number_of_trials

        while lower < upper
          middle = (lower + upper) / 2
          middle_log_cdf = log_cumulative_function(middle)
          reached = if target_log_probability == -Float::INFINITY
                      middle_log_cdf > -Float::INFINITY
                    else
                      middle_log_cdf >= target_log_probability
                    end

          if reached
            upper = middle
          else
            lower = middle + 1
          end
        end

        lower
      end

      def mean
        @mean
      end

      def variance
        @variance
      end

      private

      def calculate_log_probabilities(probabilities)
        probabilities.each_with_object([0.0]) do |probability, masses|
          log_failure = log_one_minus(probability)
          log_success = Math.log(probability)
          next_masses = Array.new(masses.size + 1, 0.0)

          next_masses[0] = masses[0] + log_failure
          (1...masses.size).each do |successes|
            next_masses[successes] = log_addition(
              masses[successes] + log_failure,
              masses[successes - 1] + log_success
            )
          end
          next_masses[masses.size] = masses.last + log_success

          masses.replace(next_masses)
        end
      end

      def calculate_log_tails
        @log_cdf = Array.new(number_of_trials + 1, -Float::INFINITY)
        @log_survival = Array.new(number_of_trials + 1, -Float::INFINITY)

        cumulative = -Float::INFINITY
        @log_masses.each_with_index do |log_mass, k|
          cumulative = log_addition(cumulative, log_mass)
          @log_cdf[k] = normalize_log_probability(cumulative)
        end

        tail = -Float::INFINITY
        number_of_trials.downto(0) do |k|
          @log_survival[k] = normalize_log_probability(tail)
          tail = log_addition(tail, @log_masses[k])
        end
      end

      def log_addition(first, second)
        return second if first == -Float::INFINITY
        return first if second == -Float::INFINITY
        return second + log_one_plus(Math.exp(first - second)) if first < second

        first + log_one_plus(Math.exp(second - first))
      end

      def log_one_plus(value)
        return 0.0 if value.zero?
        return value if value.abs < Float::EPSILON

        Math.log(1.0 + value)
      end

      def log_one_minus(value)
        return -value if value < Float::EPSILON

        Math.log(1.0 - value)
      end

      def normalize_probability(probability)
        return 0.0 if probability.negative?
        return 1.0 if probability > 1.0

        probability
      end

      def normalize_log_probability(log_probability)
        log_probability > 0.0 ? 0.0 : log_probability
      end

      def valid_integer?(value)
        value.is_a?(Numeric) && value == value.to_i
      end
    end
  end
end
