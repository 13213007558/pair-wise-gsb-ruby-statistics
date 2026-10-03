module RubyStatistics
  module StatisticalTest
    class FisherExactTest
      DEFAULT_STATE_LIMIT = 10_000

      # Conditional two-sided Fisher test for a non-empty RxC table of non-negative integers.
      # The two-sided tail contains every fixed-margin table whose conditional probability
      # is no greater than the observed table. With tolerance 0, integer factorial weights
      # are compared exactly. A positive tolerance is interpreted as
      # candidate_probability <= observed_probability + tolerance * max(1, observed_probability.abs),
      # while the threshold itself is evaluated with Rational arithmetic to avoid underflow.
      #
      # max_states bounds the number of fixed-margin tables examined, including the observed
      # table. If the bound is reached first, p_value is nil and accumulated_probability is
      # the total conditional mass searched so far.
      def self.perform(table, positional_state_limit = nil, max_states: DEFAULT_STATE_LIMIT,
                       state_limit: nil, max_iterations: nil, tolerance: 0.0,
                       comparison_tolerance: nil)
        limit = state_limit || max_iterations || positional_state_limit || max_states
        tolerance = comparison_tolerance unless comparison_tolerance.nil?
        validated_table = validate_table(table)
        tolerance = validate_probability_tolerance(tolerance)
        validate_state_limit(limit)

        reduced_table, row_indices, column_indices = reduce_table(validated_table)
        total = reduced_table.sum { |row| row.sum }

        if total.zero?
          return result(
            observed_probability: 1r,
            p_value: 1r,
            accumulated_probability: 1r,
            visited_states: 1,
            complete: true,
            table: validated_table,
            reduced_table: reduced_table,
            row_indices: row_indices,
            column_indices: column_indices,
            tolerance: tolerance,
            limit: limit,
            total_states: 1
          )
        end

        observed_weight = table_weight(reduced_table)
        probability_scale = probability_scale(reduced_table)
        observed_probability = probability_scale * observed_weight
        tail_weight = observed_weight
        accumulated_weight = observed_weight
        visited_states = 1
        exhausted = false
        total_states = nil

        enumerator = feasible_tables(reduced_table)
        while visited_states < limit
          begin
            candidate = enumerator.next
          rescue StopIteration
            exhausted = true
            break
          end

          candidate_weight = table_weight(candidate)
          tail_weight += candidate_weight if include_probability?(candidate_weight, observed_weight, probability_scale, tolerance)
          accumulated_weight += candidate_weight
          visited_states += 1
        end

        unless exhausted
          begin
            enumerator.next
          rescue StopIteration
            exhausted = true
          end
        end

        complete = exhausted

        if complete
          total_states = visited_states
          result(
            observed_probability: observed_probability,
            p_value: probability_scale * tail_weight,
            accumulated_probability: 1r,
            visited_states: visited_states,
            complete: true,
            table: validated_table,
            reduced_table: reduced_table,
            row_indices: row_indices,
            column_indices: column_indices,
            tolerance: tolerance,
            limit: limit,
            total_states: total_states
          )
        else
          result(
            observed_probability: observed_probability,
            p_value: nil,
            accumulated_probability: probability_scale * accumulated_weight,
            visited_states: visited_states,
            complete: false,
            table: validated_table,
            reduced_table: reduced_table,
            row_indices: row_indices,
            column_indices: column_indices,
            tolerance: tolerance,
            limit: limit,
            total_states: nil
          )
        end
      end

      class << self
        alias_method :exact_test, :perform
        alias_method :test, :perform
        alias_method :exact, :perform
        alias_method :two_sided, :perform
      end

      def self.feasible_tables(table)
        return to_enum(:feasible_tables, table) unless block_given?
        return if table.empty?

        row_sums = table.map { |row| row.sum }
        column_sums = table.each_with_object(Array.new(table[0].size, 0)) do |row, sums|
          row.each_with_index { |value, index| sums[index] += value }
        end
        candidate = Array.new(table.size) { Array.new(table[0].size, 0) }

        generate_rows = lambda do |row_index, remaining_columns|
          if row_index == table.size
            yield candidate.map(&:dup) unless observed_table?(table, candidate)
            return
          end

          remaining_rows = row_sums.drop(row_index).sum
          return if remaining_columns.each_with_index.any? { |sum, column_index| sum > remaining_rows }

          generate_row = lambda do |column_index, remaining_row_sum|
            if column_index == table[0].size
              generate_rows.call(row_index + 1, remaining_columns) if remaining_row_sum.zero?
              return
            end

            upper_bound = [remaining_row_sum, remaining_columns[column_index]].min
            lower_bound = if column_index == table[0].size - 1
                            remaining_row_sum
                          elsif row_index == table.size - 1
                            remaining_columns[column_index]
                          else
                            0
                          end

            (lower_bound..upper_bound).each do |value|
              next unless value <= remaining_row_sum
              next unless value <= remaining_columns[column_index]

              candidate[row_index][column_index] = value
              remaining_columns[column_index] -= value
              generate_row.call(column_index + 1, remaining_row_sum - value)
              remaining_columns[column_index] += value
              candidate[row_index][column_index] = 0
            end
          end

          generate_row.call(0, row_sums[row_index])
        end

        generate_rows.call(0, column_sums)
      end

      private_class_method def self.validate_table(table)
        unless table.is_a?(Array) && !table.empty? && table.all? { |row| row.is_a?(Array) }
          raise ArgumentError, 'table must be a non-empty array of arrays'
        end

        widths = table.map(&:size)
        if widths.any?(&:zero?) || widths.uniq.size != 1
          raise ArgumentError, 'table must be a non-empty rectangular R x C matrix'
        end

        table.each do |row|
          row.each do |value|
            unless value.is_a?(Integer) && value >= 0
              raise ArgumentError, 'all table values must be non-negative integers'
            end
          end
        end

        table.map(&:dup)
      end

      private_class_method def self.reduce_table(table)
        row_indices = table.each_index.select { |index| table[index].sum.positive? }
        column_indices = table[0].each_index.select do |column_index|
          table.sum { |row| row[column_index] }.positive?
        end

        reduced_table = row_indices.map do |row_index|
          column_indices.map { |column_index| table[row_index][column_index] }
        end

        [reduced_table, row_indices, column_indices]
      end

      private_class_method def self.validate_state_limit(limit)
        valid = limit.is_a?(Integer) && limit.positive?
        valid ||= limit == Float::INFINITY
        return if valid

        raise ArgumentError, 'state limit must be a positive integer or Float::INFINITY'
      end

      private_class_method def self.validate_probability_tolerance(tolerance)
        unless tolerance.is_a?(Numeric) && tolerance >= 0 && tolerance.to_f.finite?
          raise ArgumentError, 'tolerance must be a finite non-negative number'
        end

        tolerance.to_f
      end

      private_class_method def self.table_weight(table)
        table.reduce(1r) do |weight, row|
          row.reduce(weight) { |row_weight, value| row_weight / factorial(value) }
        end
      end

      private_class_method def self.probability_scale(table)
        row_sums = table.map { |row| row.sum }
        column_sums = table[0].each_index.map do |column_index|
          table.sum { |row| row[column_index] }
        end
        total = row_sums.sum

        numerator = row_sums.reduce(1r) { |product, value| product * factorial(value) } *
                    column_sums.reduce(1r) { |product, value| product * factorial(value) }
        numerator / factorial(total)
      end

      private_class_method def self.include_probability?(candidate_weight, observed_weight, scale, tolerance)
        return candidate_weight <= observed_weight if tolerance.zero?

        observed_probability = scale * observed_weight
        allowed_probability = Rational(tolerance) * [1r, observed_probability.abs].max
        scale * candidate_weight <= observed_probability + allowed_probability
      end

      private_class_method def self.factorial(value)
        @factorial_cache ||= [1]
        @factorial_cache.length.upto(value) { |n| @factorial_cache << @factorial_cache.last * n }
        @factorial_cache[value]
      end

      private_class_method def self.observed_table?(observed, candidate)
        observed == candidate
      end

      private_class_method def self.result(observed_probability:, p_value:, accumulated_probability:,
                                           visited_states:, complete:, table:, reduced_table:,
                                           row_indices:, column_indices:, tolerance:, limit:,
                                           total_states:)
        {
          observed_table: table,
          reduced_table: reduced_table,
          row_indices: row_indices,
          column_indices: column_indices,
          active_row_indices: row_indices,
          active_column_indices: column_indices,
          observed_probability: observed_probability,
          probability: observed_probability,
          p_value: p_value,
          two_sided_p_value: p_value,
          tail_probability: p_value,
          accumulated_probability: accumulated_probability,
          accumulated_mass: accumulated_probability,
          searched_probability: accumulated_probability,
          visited_states: visited_states,
          states_visited: visited_states,
          state_limit: limit,
          total_states: total_states,
          complete: complete,
          completed: complete,
          status: complete ? :complete : :incomplete,
          state: complete ? :complete : :incomplete,
          comparison_tolerance: tolerance
        }
      end
    end
    FisherTest = FisherExactTest
  end
end
