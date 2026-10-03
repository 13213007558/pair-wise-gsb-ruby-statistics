require_relative 'normal'

module RubyStatistics
  module Distribution
    class MultivariateNormal
      class ArgumentError < ::ArgumentError; end
      class DimensionError < RangeError; end
      class NonPositiveSemidefiniteError < ArgumentError; end

      Result = Struct.new(
        :probability,
        :estimated_error,
        :confidence_level,
        :error_budget,
        :budget_exhausted,
        :strict_error_bound,
        :random_seed,
        :samples_used,
        :algorithm,
        :rank,
        :rank_deficient,
        :near_singular,
        :deterministic,
        keyword_init: true
      ) do
        def exact?
          deterministic == true && strict_error_bound == true && budget_exhausted == false
        end
      end

      MAX_DIMENSIONS = 6
      DEFAULT_SAMPLES = 100_000

      attr_reader :mean, :covariance, :dimension, :eigenvalues, :eigenvectors,
                  :rank, :rank_deficient, :near_singular

      def initialize(mean, covariance)
        @mean = validate_vector(mean, :mean, finite: true)
        @dimension = @mean.length

        if @dimension > MAX_DIMENSIONS
          raise DimensionError, "multivariate rectangular probabilities support at most #{MAX_DIMENSIONS} dimensions"
        end

        @covariance = validate_matrix(covariance, @dimension)
        @eigenvalues, @eigenvectors = eigen_symmetric(@covariance)
        validate_positive_semidefinite!

        @rank = @eigenvalues.count(&:positive?)
        @rank_deficient = @rank < @dimension
        @near_singular = @rank_deficient ||
          (@eigenvalues.last.positive? && @eigenvalues.first / @eigenvalues.last > 1e10)
      end

      def self.rectangular_probability(mean:, covariance:, lower_bound:, upper_bound:, error_budget:,
                                        confidence_level: 0.95, seed: Random.new_seed,
                                        max_samples: DEFAULT_SAMPLES)
        new(mean, covariance).rectangular_probability(
          lower_bound: lower_bound,
          upper_bound: upper_bound,
          error_budget: error_budget,
          confidence_level: confidence_level,
          seed: seed,
          max_samples: max_samples
        )
      end

      def rectangular_probability(lower_bound:, upper_bound:, error_budget:,
                                   confidence_level: 0.95, seed: Random.new_seed,
                                   max_samples: DEFAULT_SAMPLES)
        lower = validate_vector(lower_bound, :lower_bound, finite: false)
        upper = validate_vector(upper_bound, :upper_bound, finite: false)
        budget = validate_error_budget(error_budget)
        confidence = validate_confidence_level(confidence_level)
        sample_limit = validate_positive_integer(max_samples, :max_samples)
        random_seed = validate_seed(seed)

        dimension.times do |i|
          if lower[i].nan? || upper[i].nan? || lower[i] > upper[i]
            raise ArgumentError, 'lower_bound must be component-wise less than or equal to upper_bound'
          end
        end

        if independent?
          independent_product(lower, upper, budget)
        elsif rank == 1
          rank_one_interval(lower, upper, budget)
        elsif dimension == 2
          bivariate_rectangle(lower, upper, budget, confidence, random_seed, sample_limit)
        else
          monte_carlo_rectangle(lower, upper, budget, confidence, random_seed, sample_limit)
        end
      end

      private

      def validate_vector(value, name, finite:)
        unless value.is_a?(Array) && !value.empty?
          raise ArgumentError, "#{name} must be a non-empty Array"
        end

        value.map do |item|
          unless item.is_a?(Numeric) && !item.is_a?(Complex)
            raise ArgumentError, "#{name} must contain only numeric values"
          end
          float = item.to_f
          if finite
            raise ArgumentError, "#{name} must contain only finite values" unless float.finite?
          elsif float.nan?
            raise ArgumentError, "#{name} must contain finite values or +/- Float::INFINITY"
          end
          float
        end
      rescue FloatDomainError
        raise ArgumentError, "#{name} must contain numeric values convertible to Float"
      end

      def validate_matrix(value, expected_size)
        unless value.is_a?(Array) && value.size == expected_size
          raise ArgumentError, 'covariance dimensions must match mean length'
        end

        matrix = value.map do |row|
          unless row.is_a?(Array) && row.size == expected_size
            raise ArgumentError, 'covariance must be a square matrix matching mean length'
          end

          row.map do |item|
            unless item.is_a?(Numeric) && !item.is_a?(Complex) && item.finite?
              raise ArgumentError, 'covariance must contain only finite numeric values'
            end
            item.to_f
          end
        end

        scale = [1.0, matrix.flatten.map(&:abs).max].max
        expected_size.times do |i|
          expected_size.times do |j|
            if (matrix[i][j] - matrix[j][i]).abs > 1e-12 * scale
              raise ArgumentError, 'covariance must be symmetric'
            end
            matrix[i][j] = matrix[j][i] if i > j
          end
        end

        matrix
      rescue FloatDomainError
        raise ArgumentError, 'covariance must contain numeric values convertible to Float'
      end

      def validate_error_budget(value)
        unless value.is_a?(Numeric) && value.finite? && value.positive?
          raise ArgumentError, 'error_budget must be a finite positive number'
        end
        value.to_f
      end

      def validate_confidence_level(value)
        unless value.is_a?(Numeric) && value.finite? && value > 0 && value < 1
          raise ArgumentError, 'confidence_level must be between 0 and 1 exclusively'
        end
        value.to_f
      end

      def validate_positive_integer(value, name)
        unless value.is_a?(Integer) && value.positive?
          raise ArgumentError, "#{name} must be a positive Integer"
        end
        value
      end

      def validate_seed(value)
        unless value.is_a?(Integer)
          raise ArgumentError, 'seed must be an Integer'
        end
        value
      end

      def singular_tolerance
        @singular_tolerance ||= 1e-12 * [1.0, eigenvalues.max].max
      end

      def validate_positive_semidefinite!
        if eigenvalues.last < -singular_tolerance
          raise NonPositiveSemidefiniteError, 'covariance is not positive semidefinite'
        end

        eigenvalues.map! { |value| value.negative? ? 0.0 : value }
      end

      def independent?
        dimension.times do |i|
          dimension.times do |j|
            return false if i != j && covariance[i][j] != 0
          end
        end
        true
      end

      def point_inside?(index, lower, upper)
        mean[index] >= lower[index] && mean[index] <= upper[index]
      end

      def standard_cdf(value)
        return 0.0 if value == -Float::INFINITY
        return 1.0 if value == Float::INFINITY
        0.5 * (1.0 + Math.erf(value / Math.sqrt(2.0)))
      end

      def clamp_probability(value)
        return 0.0 if value.negative? && value > -1e-12
        return 1.0 if value > 1.0 && value < 1.0 + 1e-12
        [[value, 0.0].max, 1.0].min
      end

      def independent_product(lower, upper, budget)
        probability = 1.0

        dimension.times do |i|
          standard_deviation = Math.sqrt(covariance[i][i])

        if standard_deviation.zero?
          probability = 0.0 unless point_inside?(i, lower, upper)
        else
            left = (lower[i] - mean[i]) / standard_deviation
            right = (upper[i] - mean[i]) / standard_deviation
            probability *= standard_cdf(right) - standard_cdf(left)
          end
        end

        exact_result(clamp_probability(probability), 'independent_product', budget)
      end

      def rank_one_interval(lower, upper, budget)
        direction = eigenvectors[0]
        lower_quantile = -Float::INFINITY
        upper_quantile = Float::INFINITY

        dimension.times do |i|
          coefficient = Math.sqrt(eigenvalues.first) * direction[i]

          if coefficient.zero?
            return exact_result(0.0, 'rank_one_support', budget) unless point_inside?(i, lower, upper)
            next
          end

          if lower[i].finite?
            quantile = (lower[i] - mean[i]) / coefficient
            if coefficient.positive?
              lower_quantile = [lower_quantile, quantile].max
            else
              upper_quantile = [upper_quantile, quantile].min
            end
          end

          if upper[i].finite?
            quantile = (upper[i] - mean[i]) / coefficient
            if coefficient.positive?
              upper_quantile = [upper_quantile, quantile].min
            else
              lower_quantile = [lower_quantile, quantile].max
            end
          end
        end

        probability = if lower_quantile >= upper_quantile
                        0.0
                      else
                        standard_cdf(upper_quantile) - standard_cdf(lower_quantile)
                      end

        exact_result(clamp_probability(probability), 'rank_one_support', budget)
      end

      def bivariate_rectangle(lower, upper, budget, confidence, seed, sample_limit)
        correlation = covariance[0][1] / Math.sqrt(covariance[0][0] * covariance[1][1])

        return monte_carlo_rectangle(lower, upper, budget, confidence, seed, sample_limit) if correlation.abs > 0.999999

        corners = [
          [upper[0], upper[1], 1.0],
          [lower[0], upper[1], -1.0],
          [upper[0], lower[1], -1.0],
          [lower[0], lower[1], 1.0]
        ]

        probability = 0.0
        estimated_error = 0.0
        budget_exhausted = false

        corners.each do |first_bound, second_bound, sign|
          value, error, exhausted = bivariate_orthant(
            first_bound, second_bound, correlation, budget / 4.0
          )
          probability += sign * value
          estimated_error += error
          budget_exhausted ||= exhausted
        end

        budget_exhausted ||= estimated_error > budget + 1e-15
        numerical_result(
          clamp_probability(probability), estimated_error, budget, budget_exhausted,
          'bivariate_quadrant_formula'
        )
      end

      def bivariate_orthant(first, second, correlation, tolerance)
        return [0.0, 0.0, false] if first == -Float::INFINITY || second == -Float::INFINITY
        return [1.0, 0.0, false] if first == Float::INFINITY && second == Float::INFINITY
        return [standard_cdf(second), 0.0, false] if first == Float::INFINITY
        return [standard_cdf(first), 0.0, false] if second == Float::INFINITY

        value, error, exhausted = adaptive_simpson(0.0, correlation, tolerance) do |t|
          one_minus_rho_squared = 1.0 - t * t
          exponent = -(first * first - 2.0 * t * first * second + second * second) /
            (2.0 * one_minus_rho_squared)
          Math.exp(exponent) / (2.0 * Math::PI * Math.sqrt(one_minus_rho_squared))
        end

        [clamp_probability(standard_cdf(first) * standard_cdf(second) + value), error, exhausted]
      end

      def simpson_rule(lower, upper, lower_value, middle_value, upper_value)
        (upper - lower) * (lower_value + 4.0 * middle_value + upper_value) / 6.0
      end

      def adaptive_simpson(lower, upper, tolerance, &function)
        if upper < lower
          value, error, exhausted = adaptive_simpson(upper, lower, tolerance) do |point|
            -function.call(point)
          end
          return [-value, error, exhausted]
        end

        middle = (lower + upper) / 2.0
        whole = simpson_rule(
          lower, upper, function.call(lower), function.call(middle), function.call(upper)
        )
        adaptive_simpson_recursion(
          lower, upper, function.call(lower), function.call(middle), function.call(upper),
          whole, tolerance, 20, &function
        )
      end

      def adaptive_simpson_recursion(lower, upper, lower_value, middle_value, upper_value,
                                     whole, tolerance, depth, &function)
        middle = (lower + upper) / 2.0
        left_middle = (lower + middle) / 2.0
        right_middle = (middle + upper) / 2.0
        left_value = function.call(left_middle)
        right_value = function.call(right_middle)
        left = simpson_rule(lower, middle, lower_value, left_value, middle_value)
        right = simpson_rule(middle, upper, middle_value, right_value, upper_value)
        refined = left + right
        error = (refined - whole).abs / 15.0

        if depth.zero? || error <= tolerance
          [refined + (refined - whole) / 15.0, error, depth.zero?]
        else
          left_result, left_error, left_exhausted = adaptive_simpson_recursion(
            lower, middle, lower_value, left_value, middle_value,
            left, tolerance / 2.0, depth - 1, &function
          )
          right_result, right_error, right_exhausted = adaptive_simpson_recursion(
            middle, upper, middle_value, right_value, upper_value,
            right, tolerance / 2.0, depth - 1, &function
          )
          [
            left_result + right_result,
            left_error + right_error,
            left_exhausted || right_exhausted
          ]
        end
      end

      def monte_carlo_rectangle(lower, upper, budget, confidence, seed, sample_limit)
        rows = monte_carlo_rows(lower, upper)

        return exact_result(1.0, 'unconstrained_gaussian', budget) if rows.empty?

        random = Random.new(seed)
        successes = 0
        used = 0
        estimated_error = Float::INFINITY
        batch_size = [[sample_limit, 10_000].min, 1].max
        probability = 0.0
        quantile = InverseStandardNormal.new.cumulative_function(1.0 - (1.0 - confidence) / 2.0)

        while used < sample_limit
          current_batch = [batch_size, sample_limit - used].min

          current_batch.times do
            point = Array.new(rank) { standard_gaussian(random) }
            successes += 1 if rows.all? do |coefficients, bound|
              coefficients.each_with_index.sum { |coefficient, index| coefficient * point[index] } <= bound
            end
          end

          used += current_batch
          probability = successes.to_f / used.to_f
          estimated_error = wilson_half_width(probability, used, quantile)
          break if estimated_error <= budget
        end

        budget_exhausted = estimated_error > budget + 1e-15
        numerical_result(
          clamp_probability(probability), estimated_error, budget, budget_exhausted,
          'monte_carlo_linear_gaussian', confidence: confidence, seed: seed, samples: used
        )
      end

      def monte_carlo_rows(lower, upper)
        rows = []

        dimension.times do |i|
          if covariance[i][i].zero? && !point_inside?(i, lower, upper)
            return [[Array.new(rank, 0.0), -1.0]]
          end
        end

        dimension.times do |i|
          lower_coefficients = Array.new(rank, 0.0)
          upper_coefficients = Array.new(rank, 0.0)

          rank.times do |j|
            coefficient = Math.sqrt(eigenvalues[j]) * eigenvectors[j][i]
            lower_coefficients[j] = -coefficient
            upper_coefficients[j] = coefficient
          end

          rows << [lower_coefficients, mean[i] - lower[i]] if lower[i].finite?
          rows << [upper_coefficients, upper[i] - mean[i]] if upper[i].finite?
        end

        rows
      end

      def standard_gaussian(random)
        first = random.rand
        second = random.rand
        first = random.rand while first <= Float::EPSILON
        Math.sqrt(-2.0 * Math.log(first)) * Math.cos(2.0 * Math::PI * second)
      end

      def wilson_half_width(probability, samples, quantile)
        z_squared = quantile * quantile
        denominator = 1.0 + z_squared / samples
        center = (probability + z_squared / (2.0 * samples)) / denominator
        variance = probability * (1.0 - probability) / samples +
          z_squared / (4.0 * samples * samples)
        half_width = quantile * Math.sqrt([0.0, variance].max) / denominator
        [center - half_width, center + half_width]
        half_width
      end

      def exact_result(probability, algorithm, budget)
        numerical_result(probability, 0.0, budget, false, algorithm, strict_error_bound: true)
      end

      def numerical_result(probability, estimated_error, budget, budget_exhausted, algorithm,
                           confidence: nil, seed: nil, samples: 0, strict_error_bound: false)
        Result.new(
          probability: probability,
          estimated_error: estimated_error,
          confidence_level: confidence,
          error_budget: budget,
          budget_exhausted: budget_exhausted,
          strict_error_bound: strict_error_bound,
          random_seed: seed,
          samples_used: samples,
          algorithm: algorithm,
          rank: rank,
          rank_deficient: rank_deficient,
          near_singular: near_singular,
          deterministic: seed.nil?
        )
      end

      def eigen_symmetric(matrix)
        size = matrix.size
        values = matrix.map(&:dup)
        vectors = Array.new(size) { |i| Array.new(size) { |j| i == j ? 1.0 : 0.0 } }
        scale = [1.0, values.flatten.map(&:abs).max].max

        100.times do
          off_diagonal = 0.0
          size.times do |i|
            size.times do |j|
              off_diagonal += values[i][j].abs if i < j
            end
          end
          break if off_diagonal <= 1e-14 * scale

          size.times do |p|
            (p + 1...size).each do |q|
              next if values[p][q].abs <= 1e-15 * scale

              tau = (values[q][q] - values[p][p]) / (2.0 * values[p][q])
              tangent = tau >= 0 ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau)) :
                                   -1.0 / (-tau + Math.sqrt(1.0 + tau * tau))
              cosine = 1.0 / Math.sqrt(1.0 + tangent * tangent)
              sine = tangent * cosine
              size.times do |i|
                values_p = values[i][p]
                values_q = values[i][q]
                values[i][p] = cosine * values_p - sine * values_q
                values[i][q] = sine * values_p + cosine * values_q
              end
              size.times do |j|
                values_p = values[p][j]
                values_q = values[q][j]
                values[p][j] = cosine * values_p - sine * values_q
                values[q][j] = sine * values_p + cosine * values_q
              end
              size.times do |i|
                vectors_p = vectors[i][p]
                vectors_q = vectors[i][q]
                vectors[i][p] = cosine * vectors_p - sine * vectors_q
                vectors[i][q] = sine * vectors_p + cosine * vectors_q
              end
            end
          end
        end

        pairs = size.times.map do |i|
          [values[i][i], vectors.transpose[i]]
        end
        pairs.sort_by! { |value, _| -value }
        [pairs.map(&:first), pairs.map(&:last)]
      end
    end
  end
end
