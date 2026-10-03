module RubyStatistics
  module Distribution
    class MultivariateNormal
      Result = Data.define(
        :probability,
        :error_estimate,
        :standard_error,
        :confidence_level,
        :strict_error_bound,
        :algorithm,
        :seed,
        :evaluations_used,
        :max_evaluations,
        :budget_exhausted,
        :rank,
        :rank_deficient,
        :near_singular,
        :zero_variance_dimensions
      )

      PRIMES = [2, 3, 5, 7, 11, 13].freeze
      MAX_DIMENSIONS = 6

      attr_reader :mean, :covariance

      def initialize(mean, covariance)
        @mean = validate_mean(mean)
        @covariance = validate_covariance_shape(covariance)
        validate_positive_semidefinite(@covariance)
      end

      def rectangular_probability(
        lower_bounds,
        upper_bounds,
        error_budget:,
        max_evaluations: 50_000,
        seed: Random.new_seed,
        confidence_level: 0.95
      )
        lower = validate_bounds(lower_bounds, @mean.length, :lower_bounds)
        upper = validate_bounds(upper_bounds, @mean.length, :upper_bounds)
        tolerance = validate_error_budget(error_budget)
        max_evaluations = validate_positive_integer(max_evaluations, :max_evaluations)
        confidence = validate_confidence_level(confidence_level)
        seed = validate_seed(seed)

        original_zero_dimensions = []
        active_positions = []

        @mean.each_with_index do |_, index|
          variance = @covariance[index][index]

          if variance.zero?
            unless lower[index] <= @mean[index] && @mean[index] <= upper[index]
              return exact_result(
                0.0,
                :degenerate_constraint,
                seed,
                max_evaluations,
                0,
                original_zero_dimensions + [index]
              )
            end

            original_zero_dimensions << index
          else
            active_positions << index
          end
        end

        if active_positions.empty?
          return exact_result(
            1.0,
            :degenerate,
            seed,
            max_evaluations,
            0,
            original_zero_dimensions
          )
        end

        active_mean = active_positions.map { |index| @mean[index] }
        active_lower = active_positions.map { |index| lower[index] }
        active_upper = active_positions.map { |index| upper[index] }
        active_covariance = active_positions.map do |row_index|
          active_positions.map { |column_index| @covariance[row_index][column_index] }
        end

        standard_deviations = active_covariance.map.with_index do |row, index|
          Math.sqrt(row[index])
        end

        correlation = active_covariance.map.with_index do |row, row_index|
          row.map.with_index do |value, column_index|
            value / (standard_deviations[row_index] * standard_deviations[column_index])
          end
        end

        standardized_lower = active_lower.map.with_index do |value, index|
          (value - active_mean[index]) / standard_deviations[index]
        end
        standardized_upper = active_upper.map.with_index do |value, index|
          (value - active_mean[index]) / standard_deviations[index]
        end

        decomposition = pivoted_ldl(correlation)
        permutation = decomposition.fetch(:permutation)
        rank = decomposition.fetch(:rank)
        active_dimension = correlation.length
        near_singular = rank == active_dimension &&
          decomposition.fetch(:condition_estimate) >
            1.0 / (active_dimension * Math.sqrt(Float::EPSILON))

        ordered_lower = permutation.map { |index| standardized_lower[index] }
        ordered_upper = permutation.map { |index| standardized_upper[index] }
        ordered_correlation = permutation.map do |row|
          permutation.map { |column| correlation[row][column] }
        end

        unit_lower = decomposition.fetch(:lower)
        diagonal = decomposition.fetch(:diagonal)
        factor = active_dimension.times.map do |row|
          rank.times.map do |column|
            if column <= row
              unit_lower[row][column] * Math.sqrt(diagonal[column])
            else
              0.0
            end
          end
        end

        whole_space = ordered_lower.all?(-Float::INFINITY) &&
                      ordered_upper.all?(Float::INFINITY)
        if whole_space
          return exact_result(
            1.0,
            :whole_space,
            seed,
            max_evaluations,
            rank,
            original_zero_dimensions
          )
        end

        if active_dimension == 1
          return numeric_result(
            univariate_probability(ordered_lower[0], ordered_upper[0]),
            0.0,
            0.0,
            1.0,
            nil,
            :univariate,
            true,
            0,
            max_evaluations,
            rank,
            rank < (@mean.length - original_zero_dimensions.length),
            false,
            original_zero_dimensions
          )
        end

        if independent?(active_covariance)
          probability = active_positions.map.with_index do |_, index|
            univariate_probability(standardized_lower[index], standardized_upper[index])
          end.reduce(1.0, :*)

          return numeric_result(
            probability,
            0.0,
            0.0,
            1.0,
            nil,
            :independent_product,
            true,
            0,
            max_evaluations,
            active_dimension,
            rank < @mean.length,
            false,
            original_zero_dimensions
          )
        end

        if rank == 1
          return rank_one_probability(
            ordered_lower,
            ordered_upper,
            factor,
            seed,
            max_evaluations,
            rank < (@mean.length - original_zero_dimensions.length),
            original_zero_dimensions
          )
        end

        if rank == 2 && active_dimension == 2 && !near_singular
          return bivariate_rectangular_probability(
            ordered_lower,
            ordered_upper,
            ordered_correlation,
            tolerance,
            seed,
            max_evaluations,
            rank,
            rank < (@mean.length - original_zero_dimensions.length),
            original_zero_dimensions
          )
        end

        randomized_rectangular_probability(
          ordered_lower,
          ordered_upper,
          factor,
          rank,
          active_dimension,
          tolerance,
          seed,
          max_evaluations,
          confidence,
          near_singular,
          original_zero_dimensions
        )
      end

      def self.rectangular_probability(
        mean,
        covariance,
        lower_bounds,
        upper_bounds,
        **options
      )
        new(mean, covariance).rectangular_probability(
          lower_bounds, upper_bounds, **options
        )
      end

      private

      def validate_mean(mean)
        unless mean.is_a?(Array) && !mean.empty? && mean.length <= MAX_DIMENSIONS
          raise ArgumentError, "mean must be an array with 1 to #{MAX_DIMENSIONS} finite numbers"
        end

        values = mean.map(&:to_f)
        unless values.all?(&:finite?)
          raise ArgumentError, "mean must contain finite numbers"
        end

        values.freeze
      end

      def validate_covariance_shape(covariance)
        dimension = @mean.length

        unless covariance.is_a?(Array) && covariance.length == dimension &&
               covariance.all? { |row| row.is_a?(Array) && row.length == dimension }
          raise ArgumentError, "covariance must be a #{dimension}x#{dimension} numeric matrix"
        end

        matrix = covariance.map { |row| row.map(&:to_f) }

        unless matrix.all? { |row| row.all?(&:finite?) }
          raise ArgumentError, "covariance must contain finite numbers"
        end

        dimension.times do |row|
          dimension.times do |column|
            if matrix[row][column] != matrix[column][row]
              raise ArgumentError, "covariance must be symmetric"
            end
          end

          if matrix[row][row].negative?
            raise ArgumentError, "covariance must be positive semidefinite"
          end
        end

        matrix
      end

      def validate_positive_semidefinite(matrix)
        pivoted_ldl(matrix)
      end

      def validate_bounds(bounds, dimension, name)
        unless bounds.is_a?(Array) && bounds.length == dimension
          raise ArgumentError, "#{name} must have #{dimension} entries"
        end

        values = bounds.map do |value|
          raise ArgumentError, "#{name} must contain numeric values" unless value.is_a?(Numeric)
          value.to_f
        end

        unless values.all? { |value| value.finite? || value.infinite? }
          raise ArgumentError, "#{name} must not contain NaN"
        end

        values.freeze
      end

      def validate_error_budget(value)
        number = value.to_f

        unless number.finite? && number.positive?
          raise ArgumentError, "error_budget must be a finite positive number"
        end

        number
      end

      def validate_positive_integer(value, name)
        integer = value.to_i

        unless value.is_a?(Integer) && integer.positive?
          raise ArgumentError, "#{name} must be a positive integer"
        end

        integer
      end

      def validate_confidence_level(value)
        number = value.to_f

        unless number.finite? && number > 0.0 && number < 1.0
          raise ArgumentError, "confidence_level must be between 0 and 1"
        end

        number
      end

      def validate_seed(value)
        unless value.is_a?(Integer) && value >= 0
          raise ArgumentError, "seed must be a non-negative integer"
        end

        value
      end

      def pivoted_ldl(matrix)
        work = matrix.map(&:dup)
        dimension = work.length
        permutation = (0...dimension).to_a
        unit_lower = Array.new(dimension) { Array.new(dimension, 0.0) }
        dimension.times { |index| unit_lower[index][index] = 1.0 }
        scale = work.flatten.map(&:abs).max.to_f
        tolerance = dimension * Float::EPSILON * scale
        rank = 0

        dimension.times do |pivot_step|
          pivot = (pivot_step...dimension).max_by { |index| work[index][index].abs }
          pivot_value = work[pivot][pivot]

          if pivot_value.negative? && pivot_value.abs > tolerance
            raise ArgumentError, "covariance must be positive semidefinite"
          end

          if pivot_value.abs <= tolerance
            residual = 0.0
            work[pivot_step..].each do |row|
              row[pivot_step..].each { |value| residual = [residual, value.abs].max }
            end

            if residual > tolerance
              raise ArgumentError, "covariance must be positive semidefinite"
            end

            break
          end

          swap_pivots(work, unit_lower, permutation, pivot_step, pivot)
          rank = pivot_step + 1

          (pivot_step + 1...dimension).each do |row|
            multiplier = work[row][pivot_step] / work[pivot_step][pivot_step]
            unit_lower[row][pivot_step] = multiplier
          end
          pivot_value = work[pivot_step][pivot_step]
          (pivot_step + 1...dimension).each do |row|
            (pivot_step + 1...dimension).each do |column|
              work[row][column] -= unit_lower[row][pivot_step] *
                                   pivot_value *
                                   unit_lower[column][pivot_step]
            end
          end
        end

        positive_diagonal = rank.times.map { |index| work[index][index] }
        condition_estimate = if positive_diagonal.empty?
                               Float::INFINITY
                             else
                               positive_diagonal.max / positive_diagonal.min
                             end

        {
          lower: unit_lower,
          diagonal: dimension.times.map { |index| index < rank ? work[index][index] : 0.0 },
          permutation: permutation,
          rank: rank,
          tolerance: tolerance,
          condition_estimate: condition_estimate
        }.freeze
      end

      def swap_pivots(matrix, lower, permutation, first, second)
        return if first == second

        matrix[first], matrix[second] = matrix[second], matrix[first]
        matrix.each { |row| row[first], row[second] = row[second], row[first] }
        lower[first], lower[second] = lower[second], lower[first]
        lower.each { |row| row[first], row[second] = row[second], row[first] }
        permutation[first], permutation[second] = permutation[second], permutation[first]
      end

      def independent?(covariance)
        covariance.each_with_index.all? do |row, row_index|
          row.each_with_index.all? do |value, column_index|
            row_index == column_index || value.zero?
          end
        end
      end

      def standard_normal_cdf(value)
        0.5 * (1.0 + Math.erf(value / Math.sqrt(2.0)))
      end

      def inverse_standard_normal_cdf(probability)
        probability = probability.clamp(0.0, 1.0)
        return(-Float::INFINITY) if probability.zero?
        return Float::INFINITY if probability == 1.0

        a1 = -39.6968302866538
        a2 = 220.946098424521
        a3 = -275.928510446969
        a4 = 138.357751867269
        a5 = -30.6647980661472
        a6 = 2.50662827745924
        b1 = -54.4760987982241
        b2 = 161.585836858041
        b3 = -155.698979859887
        b4 = 66.8013118877197
        b5 = -13.2806815528857
        c1 = -0.00778489400243029
        c2 = -0.322396458041136
        c3 = -2.40075827716184
        c4 = -2.54973253934373
        c5 = 4.37466414146497
        c6 = 2.93816398269878
        d1 = 0.00778469570904146
        d2 = 0.32246712907004
        d3 = 2.445134137143
        d4 = 3.75440866190742
        p_low = 0.02425
        p_high = 1.0 - p_low

        if probability < p_low
          q = Math.sqrt(-2.0 * Math.log(probability))
          (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        elsif probability <= p_high
          q = probability - 0.5
          r = q * q
          (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        else
          q = Math.sqrt(-2.0 * Math.log(1.0 - probability))
          -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        end
      end

      def univariate_probability(lower, upper)
        return 0.0 if lower >= upper

        standard_normal_cdf(upper) - standard_normal_cdf(lower)
      end

      def rank_one_probability(
        lower,
        upper,
        factor,
        seed,
        max_evaluations,
        rank_deficient,
        zero_dimensions
      )
        z_lower = -Float::INFINITY
        z_upper = Float::INFINITY

        lower.zip(upper, factor).each do |lower_bound, upper_bound, row|
          coefficient = row[0]

          if coefficient.positive?
            z_lower = [z_lower, lower_bound / coefficient].max
            z_upper = [z_upper, upper_bound / coefficient].min
          else
            z_lower = [z_lower, upper_bound / coefficient].max
            z_upper = [z_upper, lower_bound / coefficient].min
          end
        end

        probability = univariate_probability(z_lower, z_upper).clamp(0.0, 1.0)

        numeric_result(
          probability,
          0.0,
          0.0,
          1.0,
          nil,
          :rank_one,
          true,
          0,
          max_evaluations,
          1,
          rank_deficient,
          false,
          zero_dimensions
        )
      end

      def bivariate_rectangular_probability(
        lower,
        upper,
        correlation,
        tolerance,
        seed,
        max_evaluations,
        rank,
        rank_deficient,
        zero_dimensions
      )
        rho = correlation[0][1].clamp(-1.0, 1.0)
        calls = [
          [upper[0], upper[1]],
          [upper[0], lower[1]],
          [lower[0], upper[1]],
          [lower[0], lower[1]]
        ]
        target = [tolerance / 8.0, 1.0e-12].min
        evaluations = 0
        converged = true
        values = calls.map do |x, y|
          value, used_evaluations, call_converged =
            bivariate_cdf(x, y, rho, target, max_evaluations - evaluations)
          evaluations += used_evaluations
          converged &&= call_converged
          value
        end

        probability = (values[0] - values[1] - values[2] + values[3]).clamp(0.0, 1.0)

        numeric_result(
          probability,
          0.0,
          0.0,
          nil,
          nil,
          :bivariate_quadrant,
          false,
          evaluations,
          max_evaluations,
          rank,
          rank_deficient,
          false,
          zero_dimensions,
          nil,
          converged
        )
      end

      def bivariate_cdf(x, y, rho, tolerance, evaluation_budget)
        return [0.0, 0, true] if x == -Float::INFINITY || y == -Float::INFINITY
        return [standard_normal_cdf(y), 0, true] if x == Float::INFINITY
        return [standard_normal_cdf(x), 0, true] if y == Float::INFINITY
        return [0.0, 0, true] if (x.respond_to?(:nan?) && x.nan?) ||
                                  (y.respond_to?(:nan?) && y.nan?)

        if evaluation_budget.zero?
          return [standard_normal_cdf(x) * standard_normal_cdf(y), 0, false]
        end

        boundary = standard_normal_cdf(x) * standard_normal_cdf(y)
        counter = { count: 0 }
        angle_limit = Math.asin(rho)
        f = lambda do |angle|
          counter[:count] += 1
          s = Math.sin(angle)
          c2 = 1.0 - s * s
          numerator = x * x - 2.0 * x * y * s + y * y * s * s
          exponent = -(numerator / c2 + y * y) / 2.0
          Math.exp(exponent) / (2.0 * Math::PI)
        end
        integral, call_converged = adaptive_gauss_legendre(
          f,
          [0.0, angle_limit].min,
          [0.0, angle_limit].max,
          tolerance,
          evaluation_budget,
          counter
        )
        integral = -integral if angle_limit.negative?
        probability = (boundary + integral).clamp(0.0, 1.0)

        [probability, counter[:count], call_converged]
      end

      def adaptive_gauss_legendre(
        function,
        lower,
        upper,
        tolerance,
        evaluation_budget,
        counter = nil
      )
        nodes = [
          -0.9602898564975363,
          -0.7966664774136267,
          -0.5255324099163290,
          -0.1834346424956498,
          0.1834346424956498,
          0.5255324099163290,
          0.7966664774136267,
          0.9602898564975363
        ]
        weights = [
          0.1012285362903763,
          0.2223810344533745,
          0.3137066458778873,
          0.3626837833783620,
          0.3626837833783620,
          0.3137066458778873,
          0.2223810344533745,
          0.1012285362903763
        ]
        gl8 = lambda do |a, b|
          midpoint = (a + b) / 2.0
          half_length = (b - a) / 2.0
          nodes.each_with_index.sum do |node, index|
            weights.fetch(index) * function.call(midpoint + half_length * node)
          end * half_length
        end
        whole = gl8.call(lower, upper)
        counter[:count] += 8 if counter
        stack = [[lower, upper, whole, tolerance]]
        total = 0.0
        converged = true

        until stack.empty?
          a, b, whole_value, local_tolerance = stack.pop
          m = (a + b) / 2.0

          if counter && counter[:count] + 16 > evaluation_budget
            total += whole_value
            converged = false
            next
          end

          left_value = gl8.call(a, m)
          right_value = gl8.call(m, b)
          counter[:count] += 16 if counter
          refined = left_value + right_value

          if (refined - whole_value).abs <= local_tolerance
            total += refined
          else
            stack << [a, m, left_value, local_tolerance / 2.0]
            stack << [m, b, right_value, local_tolerance / 2.0]
          end
        end

        [total, converged]
      end

      def randomized_rectangular_probability(
        lower,
        upper,
        factor,
        rank,
        dimension,
        tolerance,
        seed,
        max_evaluations,
        confidence,
        near_singular,
        zero_dimensions
      )
        random = Random.new(seed)
        shifts = max_evaluations >= 1024 ? 32 : 4
        block_size = (max_evaluations / shifts).to_i
        estimates = []
        evaluations = 0
        used_shifts = 0

        if block_size.zero?
          return numeric_result(
            0.0,
            Float::NAN,
            Float::NAN,
            confidence,
            seed,
            :randomized_qmc,
            false,
            0,
            max_evaluations,
            rank,
            rank < dimension,
            near_singular,
            zero_dimensions,
            tolerance
          )
        end

        shifts.times do |shift_index|
          break if evaluations + block_size > max_evaluations

          shift = Array.new(rank) { random.rand }
          total = 0.0

          block_size.times do |point_index|
            uniforms = Array.new(rank) do |dimension_index|
              sequence_index = shift_index * block_size + point_index + 1
              radical_inverse(
                sequence_index, PRIMES.fetch(dimension_index), shift.fetch(dimension_index)
              )
            end

            total += rectangular_conditional_integrand(
              lower, upper, factor, rank, dimension, uniforms
            )
            evaluations += 1
          end

          estimates << total / block_size
          used_shifts += 1

          break if used_shifts >= 4 &&
            confidence_half_width(estimates, confidence) <= tolerance
        end

        probability = estimates.sum / estimates.length
        standard_error = if estimates.length > 1
                           Math.sqrt(sample_variance(estimates) / estimates.length)
                         else
                           Float::NAN
                         end
        error_estimate = if estimates.length > 1
                           inverse_standard_normal_cdf(0.5 + confidence / 2.0) * standard_error
                         else
                           Float::NAN
                         end

        numeric_result(
          probability.clamp(0.0, 1.0),
          error_estimate,
          standard_error,
          confidence,
          seed,
          :randomized_qmc,
          false,
          evaluations,
          max_evaluations,
          rank,
          rank < dimension,
          near_singular,
          zero_dimensions,
          tolerance
        )
      end

      def rectangular_conditional_integrand(lower, upper, factor, rank, dimension, uniforms)
        residual_lower = lower.dup
        residual_upper = upper.dup
        weight = 1.0

        rank.times do |random_index|
          diagonal = factor[random_index][random_index]
          condition_lower = residual_lower[random_index] / diagonal
          condition_upper = residual_upper[random_index] / diagonal
          probability = univariate_probability(condition_lower, condition_upper)
          lower_probability = standard_normal_cdf(condition_lower)
          return 0.0 if condition_lower > condition_upper || probability <= 0.0
          weight *= probability

          quantile = inverse_standard_normal_cdf(
            lower_probability + uniforms.fetch(random_index) * probability
          )
          return 0.0 if quantile.infinite? || quantile.nan?

          (random_index + 1...dimension).each do |constraint_index|
            coefficient = factor[constraint_index][random_index]
            next if coefficient.zero?

            residual_lower[constraint_index] -= coefficient * quantile
            residual_upper[constraint_index] -= coefficient * quantile
          end
        end

        (rank...dimension).each do |constraint_index|
          return 0.0 unless residual_lower[constraint_index] <= 0.0 &&
                             0.0 <= residual_upper[constraint_index]
        end

        weight
      end

      def radical_inverse(index, base, shift)
        result = 0.0
        factor = 1.0 / base
        value = index

        while value.positive?
          digit = value % base
          result += digit * factor
          value /= base
          factor /= base
        end

        (result + shift) % 1.0
      end

      def sample_variance(values)
        mean = values.sum / values.length
        values.sum { |value| (value - mean) ** 2 } / (values.length - 1)
      end

      def confidence_half_width(values, confidence)
        inverse_standard_normal_cdf(0.5 + confidence / 2.0) *
          Math.sqrt(sample_variance(values) / values.length)
      end

      def numeric_result(
        probability,
        error_estimate,
        standard_error,
        confidence,
        seed,
        algorithm,
        strict_error_bound,
        evaluations,
        max_evaluations,
        rank,
        rank_deficient,
        near_singular,
        zero_dimensions,
        error_target = nil,
        deterministic_converged = true
      )
        Result.new(
          probability: probability,
          error_estimate: error_estimate,
          standard_error: standard_error,
          confidence_level: confidence,
          strict_error_bound: strict_error_bound,
          algorithm: algorithm,
          seed: seed,
          evaluations_used: evaluations,
          max_evaluations: max_evaluations,
          budget_exhausted: if algorithm == :randomized_qmc
                              error_estimate.nan? || error_estimate > error_target
                            elsif strict_error_bound
                              false
                            else
                              !deterministic_converged
                            end,
          rank: rank,
          rank_deficient: rank < mean.length,
          near_singular: near_singular,
          zero_variance_dimensions: zero_dimensions
        )
      end

      def exact_result(
        probability,
        algorithm,
        seed,
        max_evaluations,
        rank,
        zero_dimensions,
        total_dimensions = mean.length
      )
        Result.new(
          probability: probability,
          error_estimate: 0.0,
          standard_error: 0.0,
          confidence_level: 1.0,
          strict_error_bound: true,
          algorithm: algorithm,
          seed: seed,
          evaluations_used: 0,
          max_evaluations: max_evaluations,
          budget_exhausted: false,
          rank: rank,
          rank_deficient: rank < total_dimensions,
          near_singular: false,
          zero_variance_dimensions: zero_dimensions
        )
      end
    end
  end
end
