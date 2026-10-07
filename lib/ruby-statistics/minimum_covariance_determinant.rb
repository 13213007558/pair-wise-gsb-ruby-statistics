require File.dirname(__FILE__) + '/../math'

module RubyStatistics
  class MinimumCovarianceDeterminant
    Result = Struct.new(
      :support,
      :center,
      :scatter,
      :rank,
      :determinant,
      :generalized_determinant,
      :h,
      :status,
      :warnings,
      :reweighted_weights,
      :reweighted_support,
      :reweighted_center,
      :reweighted_scatter,
      :reweighted_rank,
      :reweighted_determinant,
      :reweighted_generalized_determinant,
      :reweighted_cutoff,
      keyword_init: true
    ) do
      attr_accessor :estimator

      def verified_global_optimum?
        status == :verified_global_optimum
      end

      def search_candidate?
        status == :search_candidate
      end

      def singular?
        rank < center.size
      end

      def reweighted?
        !reweighted_weights.nil?
      end

      def reweighted_singular?
        reweighted? && reweighted_rank < center.size
      end

      def reweight(cutoff: DEFAULT_REWEIGHTED_QUANTILE)
        estimator.reweight_result(self, cutoff: cutoff)
      end
    end

    DEFAULT_REWEIGHTED_QUANTILE = 0.975
    DEFAULT_MAX_COMBINATIONS = 10_000
    DEFAULT_STARTS = 100

    attr_reader :data, :dimension, :rank, :h, :status, :warnings, :coordinates

    def initialize(data, h: nil, exact: nil, max_combinations: DEFAULT_MAX_COMBINATIONS,
                   starts: DEFAULT_STARTS, seed: 1, reweight: false,
                   cutoff: DEFAULT_REWEIGHTED_QUANTILE)
      @data = normalize_data(data)
      @dimension = @data.first.size
      @h = normalize_h(h || ((@data.size + @dimension + 1) / 2))
      @starts = normalize_positive_integer(starts, 'starts')
      @seed = normalize_integer(seed, 'seed')
      @cutoff = cutoff.to_f
      @warnings = []

      unless @cutoff > 0.0 && @cutoff < 1.0
        raise ArgumentError, 'cutoff must be between 0 and 1'
      end

      build_affine_coordinates
      raise ArgumentError, "h must be at least #{rank + 1} to span the data" if @h <= rank

      max_combinations = normalize_positive_integer(max_combinations, 'max_combinations')
      use_exact = exact.nil? ? combination_count_at_most?(max_combinations) : exact
      @status = use_exact ? :verified_global_optimum : :search_candidate

      fit
      apply_reweighting if reweight
    end

    def self.fit(data, **options)
      new(data, **options).result
    end

    def reweight_result(source_result, cutoff: DEFAULT_REWEIGHTED_QUANTILE)
      copy = self.class.allocate
      copy.instance_variable_set(:@data, data)
      copy.instance_variable_set(:@dimension, dimension)
      copy.instance_variable_set(:@rank, rank)
      copy.instance_variable_set(:@h, h)
      copy.instance_variable_set(:@status, status)
      copy.instance_variable_set(:@warnings, warnings.dup)
      copy.instance_variable_set(:@basis, @basis)
      copy.instance_variable_set(:@coordinates, @coordinates)
      copy.instance_variable_set(:@result, source_result)
      copy.instance_variable_set(:@raw_center_coordinates, @raw_center_coordinates)
      copy.instance_variable_set(:@raw_scatter_coordinates, @raw_scatter_coordinates)
      copy.send(:apply_reweighting, cutoff, source_result.warnings)
      reweighted_result = copy.instance_variable_get(:@result)
      reweighted_result.estimator = copy
      reweighted_result
    end

    def result
      @result
    end

    private

    def normalize_data(values)
      raise ArgumentError, 'data must be a non-empty array of vectors' unless values.is_a?(Array) && !values.empty?
      if values.any? { |point| !point.is_a?(Array) || point.empty? }
        raise ArgumentError, 'data must be a non-empty array of vectors'
      end

      dimension = values.first.size
      unless values.all? { |point| point.size == dimension && point.all? { |value| value.is_a?(Numeric) } }
        raise ArgumentError, 'all vectors must have the same numeric dimension'
      end

      values.map { |point| point.map(&:to_f) }
    end

    def normalize_h(value)
      integer = normalize_positive_integer(value, 'h')
      unless integer <= data.size
        raise ArgumentError, 'h must be between 1 and the number of observations'
      end

      integer
    end

    def normalize_positive_integer(value, name)
      integer = value.to_i
      integer_value = value.is_a?(Integer) || value == integer
      raise ArgumentError, "#{name} must be an integer" unless integer_value
      raise ArgumentError, "#{name} must be positive" unless integer.positive?

      integer
    end

    def normalize_integer(value, name)
      integer = value.to_i
      integer_value = value.is_a?(Integer) || value == integer
      raise ArgumentError, "#{name} must be an integer" unless integer_value

      integer
    end

    def build_affine_coordinates
      origin = data.first
      vectors = data.map { |point| subtract(point, origin) }
      @basis = []
      work = vectors.map(&:dup)
      row_tolerances = vectors.map { |vector| relative_tolerance(vector) }

      vectors.each_with_index do |_, pivot_row|
        next if work[pivot_row].all? { |value| value.abs <= row_tolerances[pivot_row] }

        column = (0...dimension).find do |candidate|
          work[pivot_row][candidate].abs > row_tolerances[pivot_row] &&
            @basis.none? { |vector| vector[:column] == candidate }
        end
        next unless column

        pivot_value = work[pivot_row][column]
        normalized = work[pivot_row].map { |value| value / pivot_value }
        normalized[column] = 1.0

        work.each_with_index do |row, row_index|
          next if row_index == pivot_row

          factor = row[column]
          normalized.each_with_index { |value, index| row[index] -= factor * value }
        end

        @basis << { column: column, row: normalized }
      end

      @basis.sort_by! { |vector| vector[:column] }
      @rank = @basis.size
      @coordinates = vectors.map { |vector| reduced_coordinates(vector) }
    end

    def reduced_coordinates(vector)
      residual = vector.dup
      values = []
      tolerance = relative_tolerance(vector)

      @basis.each do |entry|
        value = residual[entry[:column]]
        values << value
        entry[:row].each_with_index { |basis_value, index| residual[index] -= value * basis_value }
      end

      unless residual.all? { |value| value.abs <= tolerance * 100.0 }
        raise RuntimeError, 'numeric affine rank check failed'
      end

      values
    end

    def fit
      if status == :verified_global_optimum
        exact_search
      else
        heuristic_search
      end
      raw_candidate = evaluate(@result.support)
      @raw_center_coordinates = raw_candidate[:center_coordinates]
      @raw_scatter_coordinates = raw_candidate[:scatter_coordinates]
    end

    def exact_search
      best = nil

      data.size.times.to_a.combination(h) do |support|
        candidate = evaluate(support)
        next unless candidate

        best = candidate if better_candidate?(candidate, best)
      end

      raise RuntimeError, 'no admissible subset found' unless best
      @result = result_from(best)
    end

    def heuristic_search
      random = Random.new(@seed)
      starts = initial_supports

      remaining = @starts - starts.size
      remaining.times do
        support = []
        while support.size < h
          candidate = random.rand(data.size)
          support << candidate unless support.include?(candidate)
        end
        starts << repair_support(support.sort)
      end

      best = nil
      starts.uniq.each do |support|
        candidate = c_step(support)
        best = candidate if better_candidate?(candidate, best)
      end

      raise RuntimeError, 'no admissible subset found' unless best
      @result = result_from(best)
    end

    def initial_supports
      return [all_identical_support] if rank.zero?

      supports = []
      supports << repair_support((0...h).to_a)

      center = coordinate_center(coordinates)
      distances = coordinates.map { |point| squared_distance(point, center) }
      ordered = (0...data.size).sort_by { |index| [distances[index], index] }
      supports << repair_support(ordered.first(h))

      supports.uniq
    end

    def all_identical_support
      (0...h).to_a
    end

    def repair_support(support)
      return all_identical_support if rank.zero?

      chosen = support.dup

      until coordinate_rank(coordinates.values_at(*chosen)) == rank
        added = (0...data.size).find do |index|
          next false if chosen.include?(index)

          coordinate_rank(coordinates.values_at(*(chosen + [index]))) > coordinate_rank(coordinates.values_at(*chosen))
        end
        raise RuntimeError, 'failed to complete subset affine span' unless added

        chosen << added
      end

      while chosen.size > h
        removable = chosen.find do |index|
          reduced = chosen - [index]
          reduced.size > rank && coordinate_rank(coordinates.values_at(*reduced)) == rank
        end
        raise RuntimeError, 'failed to reduce subset while preserving affine span' unless removable

        chosen.delete(removable)
      end

      chosen.sort
    end

    def c_step(support)
      previous = nil
      current = support.sort
      visited = {}

      while current != previous
        break if visited[current]

        visited[current] = true
        candidate = evaluate(current)
        return candidate unless candidate

        previous = current
        distances = coordinates.map.with_index do |point, index|
          [mahalanobis_squared(point, candidate[:center_coordinates], candidate[:scatter_coordinates]), index]
        end
        current = distances.sort.map(&:last).first(h)
      end

      evaluate(current)
    end

    def evaluate(support)
      points = coordinates.values_at(*support)
      point_rank = coordinate_rank(points)
      return nil if point_rank < rank

      center_coordinates = coordinate_center(points)
      scatter_coordinates = weighted_scatter(points, Array.new(support.size, 1.0), support.size.to_f)

      if point_rank == rank
        determinant = rank.zero? ? 0.0 : determinant_positive_definite(scatter_coordinates)
        return nil unless determinant
      else
        determinant = 0.0
      end

      {
        support: support.sort,
        center_coordinates: center_coordinates,
        scatter_coordinates: scatter_coordinates,
        center: ambient_center(support),
        scatter: ambient_scatter(scatter_coordinates),
        rank: point_rank,
        determinant: determinant
      }
    end

    def better_candidate?(candidate, best)
      return true if best.nil?
      return true if candidate[:determinant] < best[:determinant]
      return false if candidate[:determinant] > best[:determinant]

      (candidate[:support] <=> best[:support]) == -1
    end

    def coordinate_center(points)
      return Array.new(rank, 0.0) if points.empty?

      points.each_with_object(Array.new(rank, 0.0)) do |point, center|
        point.each_with_index { |value, index| center[index] += value }
      end.map { |value| value / points.size.to_f }
    end

    def weighted_center(points, weights)
      total_weight = weights.sum.to_f
      center = Array.new(rank, 0.0)

      points.each_with_index do |point, index|
        point.each_with_index { |value, coordinate| center[coordinate] += value * weights[index] }
      end

      center.map { |value| value / total_weight }
    end

    def weighted_scatter(points, weights, denominator)
      center = weighted_center(points, weights)
      scatter = Array.new(rank) { Array.new(rank, 0.0) }

      points.each_with_index do |point, point_index|
        centered = point.each_with_index.map { |value, index| value - center[index] }

        rank.times do |row|
          rank.times do |column|
            scatter[row][column] += weights[point_index] * centered[row] * centered[column]
          end
        end
      end

      scatter.map { |row| row.map { |value| value / denominator.to_f } }
    end

    def ambient_center(support)
      center = Array.new(dimension, 0.0)

      support.each do |index|
        dimension.times { |coordinate| center[coordinate] += data[index][coordinate] }
      end

      center.map { |value| value / support.size.to_f }
    end

    def weighted_ambient_center(weights)
      total_weight = weights.sum.to_f
      center = Array.new(dimension, 0.0)

      data.each_with_index do |point, index|
        dimension.times { |coordinate| center[coordinate] += point[coordinate] * weights[index] }
      end

      center.map { |value| value / total_weight }
    end

    def ambient_scatter(coordinate_scatter)
      basis_rows = @basis.map { |entry| entry[:row] }
      result = Array.new(dimension) { Array.new(dimension, 0.0) }

      dimension.times do |row|
        dimension.times do |column|
          rank.times do |i|
            rank.times do |j|
              result[row][column] += basis_rows[i][row] * coordinate_scatter[i][j] * basis_rows[j][column]
            end
          end
        end
      end

      result
    end

    def squared_distance(point, center)
      point.each_with_index.sum { |value, index| (value - center[index]) ** 2 }
    end

    def mahalanobis_squared(point, center, scatter)
      centered = point.each_with_index.map { |value, index| value - center[index] }
      solved = solve_symmetric(scatter, centered)
      centered.each_with_index.sum { |value, index| value * solved[index] }
    end

    def solve_symmetric(matrix, vector)
      cholesky = cholesky(matrix)
      return solve_cholesky(cholesky, vector) if cholesky

      solve_gaussian(matrix.map(&:dup), vector.dup)
    end

    def cholesky(matrix)
      size = matrix.size
      factor = Array.new(size) { Array.new(size, 0.0) }
      tolerance = matrix_epsilon(matrix)

      size.times do |row|
        diagonal = matrix[row][row]
        row.times { |index| diagonal -= factor[row][index] ** 2 }
        return nil if diagonal <= tolerance

        factor[row][row] = Math.sqrt(diagonal)

        (row + 1...size).each do |column|
          sum = matrix[column][row]
          row.times { |index| sum -= factor[column][index] * factor[row][index] }
          factor[column][row] = sum / factor[row][row]
        end
      end

      factor
    end

    def solve_cholesky(factor, vector)
      size = factor.size
      y = Array.new(size, 0.0)

      size.times do |row|
        sum = vector[row]
        row.times { |index| sum -= factor[row][index] * y[index] }
        y[row] = sum / factor[row][row]
      end

      x = Array.new(size, 0.0)
      (size - 1).downto(0) do |row|
        sum = y[row]
        (row + 1...size).each { |index| sum -= factor[index][row] * x[index] }
        x[row] = sum / factor[row][row]
      end

      x
    end

    def solve_gaussian(matrix, vector)
      size = matrix.size
      augmented = matrix.each_with_index.map { |row, index| row + [vector[index]] }
      tolerance = matrix_epsilon(matrix)

      size.times do |pivot|
        pivot_row = (pivot...size).max_by { |row| augmented[row][pivot].abs }
        if augmented[pivot_row][pivot].abs <= tolerance
          raise RuntimeError, 'singular covariance encountered while solving distances'
        end

        augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot] if pivot_row != pivot
        pivot_value = augmented[pivot][pivot]

        ((pivot + 1)...size).each do |row|
          factor = augmented[row][pivot] / pivot_value
          (pivot..size).each { |column| augmented[row][column] -= factor * augmented[pivot][column] }
        end
      end

      solution = Array.new(size, 0.0)
      (size - 1).downto(0) do |row|
        sum = augmented[row][size]
        ((row + 1)...size).each { |column| sum -= augmented[row][column] * solution[column] }
        solution[row] = sum / augmented[row][row]
      end

      solution
    end

    def determinant_positive_definite(matrix)
      cholesky = cholesky(matrix)
      if cholesky
        diagonal = cholesky.each_with_index.map { |row, index| row[index] }
        return diagonal.inject(1.0) { |product, value| product * value } ** 2
      end

      determinant_gaussian(matrix.map(&:dup))
    end

    def determinant_gaussian(matrix)
      size = matrix.size
      determinant = 1.0

      size.times do |pivot|
        pivot_row = (pivot...size).max_by { |row| matrix[row][pivot].abs }
        value = matrix[pivot_row][pivot]
        return 0.0 if value.zero?

        matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot] if pivot_row != pivot
        determinant *= -1.0 if pivot_row != pivot
        determinant *= matrix[pivot][pivot]

        ((pivot + 1)...size).each do |row|
          factor = matrix[row][pivot] / matrix[pivot][pivot]
          (pivot...size).each { |column| matrix[row][column] -= factor * matrix[pivot][column] }
        end
      end

      determinant.abs
    end

    def coordinate_rank(points)
      return 0 if points.empty? || rank.zero?

      matrix = points[1..].map { |point| point.map(&:to_f) }
      matrix_rank(matrix)
    end

    def matrix_rank(matrix)
      return 0 if matrix.empty?

      work = matrix.map(&:dup)
      columns = work.first.size
      row_tolerances = work.map { |row| relative_tolerance(row) }
      used_columns = []
      pivot_count = 0

      work.each_with_index do |row, row_index|
        column = (0...columns).find do |candidate|
          next false if used_columns.include?(candidate)

          row[candidate].abs > row_tolerances[row_index]
        end
        next unless column

        pivot_value = row[column]
        row.map! { |value| value / pivot_value }
        used_columns << column
        pivot_count += 1

        work.each do |other|
          next if other.equal?(row)

          factor = other[column]
          columns.times { |index| other[index] -= factor * row[index] }
        end
      end

      pivot_count
    end

    def subtract(left, right)
      left.each_with_index.map { |value, index| value - right[index] }
    end

    def relative_tolerance(vector)
      scale = vector.map { |value| value.abs }.max.to_f
      [Float::EPSILON, vector.size * Float::EPSILON * scale].max
    end

    def matrix_epsilon(matrix)
      scale = matrix.flatten.map(&:abs).max.to_f
      [Float::EPSILON, matrix.first.size * Float::EPSILON * scale].max
    end

    def combination_count_at_most?(limit)
      count = 1.0
      selection = [h, data.size - h].min
      selection.times do |offset|
        count *= (data.size - offset).to_f / (offset + 1)
        return false if count > limit
      end
      true
    end

    def result_from(candidate)
      if rank.zero?
        warnings << 'all observations are identical; scatter is a zero matrix'
      elsif rank < dimension
        warnings << "data lie in an affine subspace of rank #{rank}; reported scatter is singular"
      end
      warnings << 'duplicate observations are retained without adding noise' if data.uniq.size < data.size

      result = Result.new(
        support: candidate[:support],
        center: candidate[:center],
        scatter: candidate[:scatter],
        rank: rank,
        determinant: rank == dimension ? candidate[:determinant] : 0.0,
        generalized_determinant: candidate[:determinant],
        h: h,
        status: status,
        warnings: warnings.dup,
        reweighted_weights: nil,
        reweighted_support: nil,
        reweighted_center: nil,
        reweighted_scatter: nil,
        reweighted_rank: nil,
        reweighted_determinant: nil,
        reweighted_generalized_determinant: nil,
        reweighted_cutoff: nil
      )
      result.estimator = self
      result
    end

    def apply_reweighting(cutoff = @cutoff, source_warnings = warnings)
      cutoff = cutoff.to_f
      unless cutoff > 0.0 && cutoff < 1.0
        raise ArgumentError, 'cutoff must be between 0 and 1'
      end

      distances = coordinates.map do |point|
        if rank.zero?
          0.0
        else
          mahalanobis_squared(point, @raw_center_coordinates, @raw_scatter_coordinates)
        end
      end

      threshold = rank.zero? ? Float::INFINITY : chi_squared_quantile(rank, cutoff)
      weights = distances.map { |distance| distance <= threshold ? 1.0 : 0.0 }
      selected = weights.each_index.select { |index| weights[index] == 1.0 }
      raise RuntimeError, 'reweighting retained no observations; relax the cutoff' if selected.empty?

      selected_points = coordinates.values_at(*selected)
      selected_weights = weights.values_at(*selected)
      reweighted_rank = coordinate_rank(selected_points)
      center_coordinates = weighted_center(selected_points, selected_weights)
      scatter_coordinates = weighted_scatter(selected_points, selected_weights, selected_weights.sum.to_f)
      local_warnings = source_warnings.dup

      if reweighted_rank < rank
        local_warnings << "reweighted support has rank #{reweighted_rank}; reweighted scatter is singular"
      elsif rank < dimension
        local_warnings << "reweighted support remains in an affine subspace of rank #{rank}"
      end
      local_warnings.uniq!

      generalized_determinant = if reweighted_rank == rank && reweighted_rank.positive?
                                  determinant_positive_definite(scatter_coordinates)
                                else
                                  0.0
                                end
      determinant = rank == dimension ? generalized_determinant : 0.0

      reweighted_result = Result.new(
        support: @result.support,
        center: @result.center,
        scatter: @result.scatter,
        rank: @result.rank,
        determinant: @result.determinant,
        generalized_determinant: @result.generalized_determinant,
        h: @result.h,
        status: @result.status,
        warnings: local_warnings,
        reweighted_weights: weights,
        reweighted_support: selected,
        reweighted_center: weighted_ambient_center(weights),
        reweighted_scatter: ambient_scatter(scatter_coordinates),
        reweighted_rank: reweighted_rank,
        reweighted_determinant: determinant,
        reweighted_generalized_determinant: generalized_determinant,
        reweighted_cutoff: cutoff
      )
      reweighted_result.estimator = self
      @result = reweighted_result
    end

    def chi_squared_quantile(degrees_of_freedom, probability)
      lower = 0.0
      upper = [1.0, degrees_of_freedom.to_f].max
      upper *= 2.0 while chi_squared_cumulative(upper, degrees_of_freedom) < probability

      64.times do
        middle = (lower + upper) / 2.0
        if chi_squared_cumulative(middle, degrees_of_freedom) < probability
          lower = middle
        else
          upper = middle
        end
      end

      (lower + upper) / 2.0
    end

    def chi_squared_cumulative(value, degrees_of_freedom)
      return 0.0 if value <= 0.0

      shape = degrees_of_freedom / 2.0
      x = value / 2.0
      log_prefix = -x + shape * Math.log(x) - Math.lgamma(shape)[0]

      if x < shape + 1.0
        term = 1.0 / shape
        total = term

        200.times do |index|
          term *= x / (shape + index + 1.0)
          total += term
          break if term.abs <= total.abs * Float::EPSILON
        end

        [0.0, [1.0, total * Math.exp(log_prefix)].min].max
      else
        b = x + 1.0 - shape
        c = 1.0e300
        d = 1.0 / b
        h = d
        tiny = 1.0e-300

        200.times do |index|
          m = index + 1.0
          numerator = -m * (m - shape)
          b += 2.0

          d = numerator * d + b
          d = tiny if d.abs < tiny
          c = b + numerator / c
          c = tiny if c.abs < tiny

          d = 1.0 / d
          delta = d * c
          h *= delta
          break if (delta - 1.0).abs < Float::EPSILON
        end

        upper_probability = Math.exp(log_prefix) * h
        [0.0, [1.0, 1.0 - upper_probability].min].max
      end
    end
  end
end
