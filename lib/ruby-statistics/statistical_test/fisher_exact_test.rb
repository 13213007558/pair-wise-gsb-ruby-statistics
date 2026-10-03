module RubyStatistics
  module StatisticalTest
    # Fisher's exact test for small RxC contingency tables with non-negative
    # integer entries.
    #
    # The test conditions on the observed row and column margins. Under the
    # null hypothesis of independence, the conditional probability of a table
    # T with the same margins follows the (multivariate) hypergeometric
    # distribution:
    #
    #   P(T) = (prod row_i! * prod col_j!) / (N! * prod x_ij!)
    #
    # The two-sided p-value is defined as the sum of the probabilities of
    # every table with the same margins whose conditional probability is
    # less than or equal to the probability of the observed table.
    #
    # All probabilities are computed through log-factorials, so no factorial
    # is ever materialised and the computation does not overflow. The
    # comparison "not greater than the observed table" is performed on the
    # log scale with a small absolute tolerance (LOG_PROBABILITY_TOLERANCE)
    # to absorb floating point noise: a table is included in the p-value
    # when log P(T) <= log P(observed) + 1e-10.
    #
    # All-zero rows and columns are dropped before enumerating (they do not
    # change the conditional distribution because 0! == 1), but the result
    # keeps the original margins plus the indices of the kept rows/columns
    # so the reduced table can be mapped back to the original one.
    class FisherExactTest
      # Absolute tolerance used when comparing log-probabilities against the
      # log-probability of the observed table.
      LOG_PROBABILITY_TOLERANCE = 1e-10

      # Default upper bound on the number of tables enumerated while
      # searching the state space defined by the fixed margins.
      DEFAULT_MAX_STATES = 1_000_000

      # Conditional probability of the observed table given its margins.
      def self.observed_probability(table)
        validate_table!(table)

        reduced_table, reduced_rows, reduced_columns = reduce_table(table)
        total = reduced_rows.sum
        log_fact = log_factorials(total)

        Math.exp(log_margin_factor(reduced_rows, reduced_columns, log_fact) -
                 cell_log_factorial_sum(reduced_table, log_fact))
      end

      # Two-sided Fisher exact test for an RxC table.
      #
      # Returns a hash with:
      #   observed_probability: conditional probability of the observed table.
      #   p_value:              two-sided p-value, or nil when the search was
      #                         cut short by max_states (no fake final value
      #                         is reported in that case).
      #   complete:             true when the whole state space was enumerated.
      #   states:               number of tables enumerated.
      #   accumulated_mass:     probability mass accumulated towards the
      #                         p-value so far (partial when not complete).
      #   total_mass:           probability mass of all enumerated tables
      #                         (1.0, up to float noise, when complete).
      #   row_margins/column_margins: margins of the original table.
      #   nonzero_rows/nonzero_columns: indices of the kept rows/columns.
      #   reduced_table:        table actually enumerated (zero rows/columns
      #                         removed).
      def self.exact_test(table, max_states: DEFAULT_MAX_STATES)
        validate_table!(table)

        unless max_states.is_a?(Integer) && max_states.positive?
          raise ArgumentError, 'max_states must be a positive integer'
        end

        row_margins = table.map { |row| row.sum }
        column_margins = table.transpose.map { |column| column.sum }

        reduced_table, reduced_rows, reduced_columns = reduce_table(table)
        total = reduced_rows.sum
        log_fact = log_factorials(total)
        log_margin_factor = log_margin_factor(reduced_rows, reduced_columns, log_fact)

        observed_log_probability =
          log_margin_factor - cell_log_factorial_sum(reduced_table, log_fact)

        states = 0
        accumulated_mass = 0.0
        total_mass = 0.0
        complete = true

        row_count = reduced_rows.size
        column_count = reduced_columns.size
        current = Array.new(row_count) { Array.new(column_count, 0) }
        column_left = reduced_columns.dup

        enumerate = lambda do |row_index, column_index, row_left|
          if column_index == column_count - 1
            # Last column of the row: the cell is forced by the row margin.
            return if row_left > column_left[column_index]

            current[row_index][column_index] = row_left
            column_left[column_index] -= row_left

            if row_index == row_count - 1
              if states >= max_states
                complete = false
                throw(:state_limit_reached)
              end

              states += 1
              log_probability =
                log_margin_factor - cell_log_factorial_sum(current, log_fact)
              probability = Math.exp(log_probability)

              total_mass += probability
              if log_probability <= observed_log_probability + LOG_PROBABILITY_TOLERANCE
                accumulated_mass += probability
              end
            else
              enumerate.call(row_index + 1, 0, reduced_rows[row_index + 1])
            end

            column_left[column_index] += row_left
            return
          end

          max_value = [row_left, column_left[column_index]].min
          (0..max_value).each do |value|
            current[row_index][column_index] = value
            column_left[column_index] -= value
            enumerate.call(row_index, column_index + 1, row_left - value)
            column_left[column_index] += value
          end
        end

        catch(:state_limit_reached) do
          enumerate.call(0, 0, reduced_rows[0])
        end

        {
          observed_probability: Math.exp(observed_log_probability),
          p_value: complete ? [accumulated_mass, 1.0].min : nil,
          complete: complete,
          states: states,
          accumulated_mass: accumulated_mass,
          total_mass: total_mass,
          row_margins: row_margins,
          column_margins: column_margins,
          nonzero_rows: row_margins.each_index.select { |i| row_margins[i].positive? },
          nonzero_columns: column_margins.each_index.select { |j| column_margins[j].positive? },
          reduced_table: reduced_table
        }
      end

      def self.validate_table!(table)
        unless table.is_a?(Array) && !table.empty? &&
               table.all? { |row| row.is_a?(Array) && !row.empty? }
          raise ArgumentError, 'table must be a non-empty array of non-empty rows'
        end

        width = table.first.size
        unless table.all? { |row| row.size == width }
          raise ArgumentError, 'table rows must all have the same length'
        end

        table.each_with_index do |row, row_index|
          row.each_with_index do |value, column_index|
            unless value.is_a?(Integer)
              raise ArgumentError,
                    "cell (#{row_index}, #{column_index}) is not an integer: #{value.inspect}"
            end

            if value.negative?
              raise ArgumentError,
                    "cell (#{row_index}, #{column_index}) is negative: #{value}"
            end
          end
        end

        if table.sum { |row| row.sum }.zero?
          raise ArgumentError, 'table margins are degenerate: total count is zero'
        end
      end
      private_class_method :validate_table!

      # Drops all-zero rows and columns. Returns the reduced table together
      # with its row and column margins.
      def self.reduce_table(table)
        row_margins = table.map { |row| row.sum }
        column_margins = table.transpose.map { |column| column.sum }

        kept_rows = row_margins.each_index.select { |i| row_margins[i].positive? }
        kept_columns = column_margins.each_index.select { |j| column_margins[j].positive? }

        reduced_table = kept_rows.map { |i| kept_columns.map { |j| table[i][j] } }
        reduced_rows = kept_rows.map { |i| row_margins[i] }
        reduced_columns = kept_columns.map { |j| column_margins[j] }

        [reduced_table, reduced_rows, reduced_columns]
      end
      private_class_method :reduce_table

      # log_fact[k] == log(k!), built by accumulation to avoid overflow and
      # to keep rounding error minimal.
      def self.log_factorials(n)
        log_fact = Array.new(n + 1, 0.0)
        (1..n).each { |k| log_fact[k] = log_fact[k - 1] + Math.log(k) }
        log_fact
      end
      private_class_method :log_factorials

      # log of (prod row_i! * prod col_j!) / N!, shared by every table with
      # the given margins.
      def self.log_margin_factor(row_margins, column_margins, log_fact)
        total = row_margins.sum
        (row_margins + column_margins).sum { |margin| log_fact[margin] } - log_fact[total]
      end
      private_class_method :log_margin_factor

      # Sum of log(x_ij!) over every cell of a table.
      def self.cell_log_factorial_sum(table, log_fact)
        table.sum { |row| row.sum { |value| log_fact[value] } }
      end
      private_class_method :cell_log_factorial_sum
    end
  end
end
