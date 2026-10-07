require_relative 'distribution/chi_squared'

module RubyStatistics
  class MinimumCovarianceDeterminant
    SingularCovarianceError = Class.new(StandardError)

    Result = Struct.new(
      :h,
      :support,
      :center,
      :scatter,
      :determinant,
      :weights,
      :threshold,
      :reweighted,
      :reweighted_center,
      :reweighted_scatter,
      :optimality,
      :mode,
      :singular_subsets,
      :starts,
      :iterations,
      keyword_init: true
    ) do
      def verified_global_optimum?
        optimality == :global_optimum_verified
      end

      def search_candidate?
        optimality == :search_candidate
      end

      def reweighted?
        reweighted
      end
    end

    def self.fit(
      data,
      h: nil,
      reweight: false,
      threshold: nil,
      exact: :auto,
      max_combinations: 100_000,
      starts: 500,
      seed: 479_001_599
    )
      new(
        data,
        h: h,
        reweight: reweight,
        threshold: threshold,
        exact: exact,
        max_combinations: max_combinations,
        starts: starts,
        seed: seed
      ).fit
    end

    def self.reweight(data, center:, scatter:, threshold: nil)
      new(data).reweight(center: center, scatter: scatter, threshold: threshold)
    end

    def self.squared_mahalanobis_distances(data, center:, scatter:)
      new(data).squared_mahalanobis_distances(center: center, scatter: scatter)
    end

    def initialize(
      data,
      h: nil,
      reweight: false,
      threshold: nil,
      exact: :auto,
      max_combinations: 100_000,
      starts: 500,
      seed: 479_001_599
    )
      @data = normalize_data(data)
      @dimension = @data.first.size
      @h = h.nil? ? minimum_h : validate_h(h)
      ensure_feasible_h!
      @reweight = reweight
      @threshold = threshold
      @exact = exact
      @max_combinations = max_combinations
      @starts = starts
      @seed = seed
      validate_options!
    end

    def fit
      ensure_feasible_h!
      ensure_full_rank_data!

      result = if exact_enumeration?
                 enumerate_exactly
               else
                 search_with_c_steps
               end

      weights, threshold = weighted_result(result.center, result.scatter)
      result.weights = weights
      result.threshold = threshold

      if @reweight
        reweighted = reweight(center: result.center, scatter: result.scatter, threshold: threshold)
        result.reweighted = true
        result.reweighted_center = reweighted[:center]
        result.reweighted_scatter = reweighted[:scatter]
      else
        result.reweighted = false
      end

      result
    end

    def reweight(center:, scatter:, threshold: nil)
      weights, cutoff = weighted_result(center, scatter, threshold)
      selected = weights.each_index.select { |index| weights[index] == 1 }

      if selected.empty?
        raise ArgumentError, 'Reweighting selected no observations; choose a larger threshold.'
      end

      center, scatter = statistics_for(selected)
      raise SingularCovarianceError, 'The reweighted scatter matrix is singular.' unless matrix_rank(scatter) == @dimension

      { weights: weights, threshold: cutoff, center: center, scatter: scatter }
    end

    def squared_mahalanobis_distances(center:, scatter:)
      squared_distances(center, scatter)
    end

    private

    def normalize_data(data)
      raise ArgumentError, 'data must contain observations' if data.nil? || data.empty?

      rows = data.map do |observation|
        unless observation.respond_to?(:size) && observation.respond_to?(:[])
          raise ArgumentError, 'Each observation must be a numeric vector.'
        end

        unless observation.respond_to?(:to_a)
          raise ArgumentError, 'Each observation must be a numeric vector.'
        end

        values = observation.to_a.map do |value|
          raise ArgumentError, 'Each observation must contain numeric values.' unless value.is_a?(Numeric)

          numeric_value = value.to_f
          raise ArgumentError, 'Each observation must contain finite values.' unless numeric_value.finite?

          numeric_value
        end

        raise ArgumentError, 'Each observation must be non-empty.' if values.empty?
        values
      end

      dimension = rows.first.size
      if rows.any? { |row| row.size != dimension }
        raise ArgumentError, 'All observations must have the same dimension.'
      end

      rows
    end

    def minimum_h
      (@data.size + @dimension + 1) / 2
    end

    def validate_h(value)
      unless value.is_a?(Integer) && value >= 0
        raise ArgumentError, 'h must be a non-negative integer.'
      end

      if value < minimum_h || value > @data.size
        raise ArgumentError, "h must be between #{minimum_h} and #{@data.size}."
      end

      if value <= @dimension
        raise ArgumentError, "h must be greater than the dimension (#{@dimension}) for invertible scatter."
      end

      value
    end

    def ensure_feasible_h!
      return unless @h <= @dimension

      raise ArgumentError, "h must be greater than the dimension (#{@dimension}) for invertible scatter."
    end

    def validate_options!
      unless @max_combinations.is_a?(Integer) && @max_combinations > 0
        raise ArgumentError, 'max_combinations must be a positive integer.'
      end

      unless [true, false, :auto].include?(@exact)
        raise ArgumentError, 'exact must be true, false, or :auto.'
      end
    end

    def ensure_full_rank_data!
      _center, scatter = statistics_for((0...@data.size).to_a)
      return if matrix_rank(scatter) == @dimension

      raise SingularCovarianceError,
            'The data lie in a lower-dimensional subspace; no invertible MCD scatter exists.'
    end

    def exact_enumeration?
      case @exact
      when true
        true
      when false
        false
      when :auto
        combination_count(@data.size, @h) <= @max_combinations
      end
    end

    def enumerate_exactly
      best = nil
      singular_subsets = 0

      (0...@data.size).to_a.combination(@h) do |support|
        candidate = candidate_for(support)
        if candidate.nil?
          singular_subsets += 1
        elsif best.nil? || smaller_determinant?(candidate[:determinant], best[:determinant])
          best = candidate
        end
      end

      raise SingularCovarianceError, 'Every h-subset is singular.' if best.nil?

      base_result(best).tap do |result|
        result.optimality = :global_optimum_verified
        result.mode = :exact_enumeration
        result.singular_subsets = singular_subsets
        result.starts = nil
        result.iterations = nil
      end
    end

    def search_with_c_steps
      unless @starts.is_a?(Integer) && @starts > 0
        raise ArgumentError, 'starts must be a positive integer.'
      end

      random = Random.new(@seed)
      tried_supports = {}
      best = nil
      total_iterations = 0
      total_singular = 0
      attempted_starts = 0

      @starts.times do |attempt|
        attempted_starts += 1
        support = if attempt.zero?
                    deterministic_full_rank_support(tried_supports)
                  else
                    initial_support(random, tried_supports)
                  end
        break if support.nil?

        candidate, iterations, singular_steps = c_step(support)
        total_iterations += iterations
        total_singular += singular_steps
        next if candidate.nil?

        best = candidate if best.nil? || smaller_determinant?(candidate[:determinant], best[:determinant])
      end

      raise SingularCovarianceError, 'Unable to construct a non-singular initial subset.' if best.nil?

      base_result(best).tap do |result|
        result.optimality = :search_candidate
        result.mode = :multi_start_c_step
        result.singular_subsets = total_singular + tried_supports.count { |_key, status| status == :singular }
        result.starts = attempted_starts
        result.iterations = total_iterations
      end
    end

    def deterministic_full_rank_support(tried_supports)
      selected = nil

      (0...@data.size).each do |anchor|
        candidate = [anchor]
        directions = []

        while candidate.size <= @dimension
          found = (0...@data.size).find do |index|
            next false if candidate.include?(index)

            direction = @data[index].each_with_index.map { |value, coordinate| value - @data[anchor][coordinate] }
            vector_rank(directions + [direction]) == directions.size + 1
          end

          break if found.nil?

          candidate << found
          directions << @data[found].each_with_index.map { |value, coordinate| value - @data[anchor][coordinate] }
        end

        selected = candidate if candidate.size > @dimension
        break unless selected.nil?
      end

      raise SingularCovarianceError, 'Could not construct a deterministic full-rank subset.' if selected.nil?

      remaining = (0...@data.size).to_a - selected
      support = (selected + remaining.first(@h - selected.size)).sort
      tried_supports[support.join(',')] = :full_rank
      support
    end

    def initial_support(random, tried_supports)
      indices = (0...@data.size).to_a
      maximum_attempts = [@starts * 20, 100].max
      attempts = 0

      loop do
        return nil if attempts >= maximum_attempts

        attempts += 1
        shuffled = indices.shuffle(random: random)
        support = shuffled.first(@h).sort
        key = support.join(',')
        next if tried_supports.key?(key)

        candidate = candidate_for(support)
        if candidate.nil?
          tried_supports[key] = :singular
        else
          tried_supports[key] = :full_rank
          return support
        end
      end
    end

    def c_step(support)
      iterations = 0
      singular_steps = 0
      seen = { support.join(',') => true }

      loop do
        current = candidate_for(support)
        break [nil, iterations, singular_steps] if current.nil?

        distances = squared_distances(current[:center], current[:scatter])
        next_support = distances.each_with_index.sort_by { |distance, index| [distance, index] }.first(@h).map(&:last).sort
        iterations += 1

        break [current, iterations, singular_steps] if next_support == support

        next_candidate = candidate_for(next_support)
        if next_candidate.nil?
          singular_steps += 1
          break [current, iterations, singular_steps]
        end

        unless next_candidate[:determinant] <= current[:determinant] + smaller_tolerance(next_candidate[:determinant], current[:determinant])
          break [current, iterations, singular_steps]
        end

        key = next_support.join(',')
        break [next_candidate, iterations, singular_steps] if seen.key?(key)

        seen[key] = true
        support = next_support
      end
    end

    def smaller_determinant?(candidate, best)
      tolerance = smaller_tolerance(candidate, best)
      candidate + tolerance < best
    end

    def smaller_tolerance(candidate, best)
      1e-12 * [candidate.abs, best.abs, 1.0].max
    end

    def candidate_for(support)
      center, scatter = statistics_for(support)
      return nil unless matrix_rank(scatter) == @dimension

      determinant = determinant(scatter)
      return nil unless determinant.finite? && determinant > 0.0

      { support: support, center: center, scatter: scatter, determinant: determinant }
    end

    def statistics_for(indices)
      selected = indices.map { |index| @data[index] }
      center = Array.new(@dimension) do |coordinate|
        selected.sum { |row| row[coordinate] } / selected.size.to_f
      end

      scatter = Array.new(@dimension) { Array.new(@dimension, 0.0) }
      selected.each do |row|
        centered = row.each_with_index.map { |value, coordinate| value - center[coordinate] }
        @dimension.times do |row_index|
          @dimension.times do |column_index|
            scatter[row_index][column_index] += centered[row_index] * centered[column_index]
          end
        end
      end

      divisor = selected.size.to_f
      scatter.each { |scatter_row| scatter_row.map! { |value| value / divisor } }
      [center, scatter]
    end

    def squared_distances(center, scatter)
      @data.map do |row|
        centered = row.each_with_index.map { |value, index| value - center[index] }
        adjusted = solve(scatter, centered)
        distance = centered.each_with_index.sum { |value, index| value * adjusted[index] }
        distance < 0.0 && distance.abs < 1e-12 ? 0.0 : distance
      end
    end

    def matrix_rank(matrix)
      work = matrix.map(&:dup)
      rank = 0
      0.upto(@dimension - 1) do |pivot_column|
        pivot_row = rank.upto(@dimension - 1).max_by { |row| work[row][pivot_column].abs }
        pivot = work[pivot_row][pivot_column]
        next if pivot.abs.zero?

        work[rank], work[pivot_row] = work[pivot_row], work[rank] if pivot_row != rank
        pivot_value = work[rank][pivot_column]

        (rank + 1).upto(@dimension - 1) do |row|
          factor = work[row][pivot_column] / pivot_value
          pivot_column.upto(@dimension - 1) do |column|
            work[row][column] -= factor * work[rank][column]
          end
        end

        rank += 1
      end
      rank
    end

    def vector_rank(vectors)
      return 0 if vectors.empty?

      work = vectors.map(&:dup)
      rank = 0
      0.upto(@dimension - 1) do |pivot_column|
        pivot_row = rank.upto(work.size - 1).max_by { |row| work[row][pivot_column].abs }
        next if pivot_row.nil? || work[pivot_row][pivot_column].abs.zero?

        work[rank], work[pivot_row] = work[pivot_row], work[rank] if pivot_row != rank
        pivot_value = work[rank][pivot_column]

        (rank + 1).upto(work.size - 1) do |row|
          factor = work[row][pivot_column] / pivot_value
          pivot_column.upto(@dimension - 1) do |column|
            work[row][column] -= factor * work[rank][column]
          end
        end

        rank += 1
        return rank if rank == work.size
      end
      rank
    end

    def determinant(matrix)
      work = matrix.map(&:dup)
      result = 1.0

      0.upto(@dimension - 1) do |pivot_column|
        pivot_row = pivot_column.upto(@dimension - 1).max_by { |row| work[row][pivot_column].abs }
        return 0.0 if work[pivot_row][pivot_column].abs.zero?

        work[pivot_column], work[pivot_row] = work[pivot_row], work[pivot_column] if pivot_row != pivot_column
        result *= -1.0 if pivot_row != pivot_column
        pivot_value = work[pivot_column][pivot_column]
        result *= pivot_value

        (pivot_column + 1).upto(@dimension - 1) do |row|
          factor = work[row][pivot_column] / pivot_value
          pivot_column.upto(@dimension - 1) do |column|
            work[row][column] -= factor * work[pivot_column][column]
          end
        end
      end

      result
    end

    def solve(matrix, vector)
      augmented = matrix.each_with_index.map do |row, index|
        row.dup << vector[index]
      end

      0.upto(@dimension - 1) do |pivot|
        pivot_row = pivot.upto(@dimension - 1).max_by { |row| augmented[row][pivot].abs }
        pivot_value = augmented[pivot_row][pivot]
        raise SingularCovarianceError, 'Scatter matrix is singular.' if pivot_value.abs.zero?

        augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot] if pivot_row != pivot
        normalized = augmented[pivot][pivot]
        augmented[pivot] = augmented[pivot].map { |value| value / normalized }

        @dimension.times do |row|
          next if row == pivot

          factor = augmented[row][pivot]
          augmented[row] = augmented[row].each_with_index.map do |value, column|
            value - factor * augmented[pivot][column]
          end
        end
      end

      augmented.map(&:last)
    end

    def weighted_result(center, scatter, threshold = @threshold)
      cutoff = threshold || chi_squared_quantile(0.975, @dimension)
      unless cutoff.is_a?(Numeric) && cutoff.finite? && cutoff >= 0
        raise ArgumentError, 'threshold must be a finite non-negative squared Mahalanobis distance.'
      end

      weights = squared_distances(center, scatter).map { |distance| distance <= cutoff ? 1 : 0 }
      [weights, cutoff.to_f]
    end

    def chi_squared_quantile(probability, degrees_of_freedom)
      unless probability.is_a?(Numeric) && probability > 0 && probability < 1
        raise ArgumentError, 'probability must be between 0 and 1.'
      end

      distribution = RubyStatistics::Distribution::ChiSquared.new(degrees_of_freedom)
      lower = 0.0
      upper = [degrees_of_freedom.to_f, 1.0].max
      upper *= 2.0 while distribution.cumulative_function(upper) < probability

      64.times do
        middle = (lower + upper) / 2.0
        if distribution.cumulative_function(middle) < probability
          lower = middle
        else
          upper = middle
        end
      end

      (lower + upper) / 2.0
    end

    def combination_count(n, k)
      k = [k, n - k].min
      (1..k).inject(1) { |product, index| product * (n - k + index) / index }
    end

    def base_result(candidate)
      Result.new(
        h: @h,
        support: candidate[:support],
        center: candidate[:center],
        scatter: candidate[:scatter].to_a,
        determinant: candidate[:determinant],
        weights: nil,
        threshold: nil,
        reweighted: nil,
        reweighted_center: nil,
        reweighted_scatter: nil,
        optimality: nil,
        mode: nil,
        singular_subsets: nil,
        starts: nil,
        iterations: nil
      )
    end
  end
end
