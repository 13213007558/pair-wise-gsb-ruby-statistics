module RubyStatistics
  module StatisticalTest
    class WilcoxonSignedRankTest
      ZERO_METHOD_ALIASES = {
        wilcox: :wilcox,
        discard: :wilcox,
        discard_zeros: :wilcox,
        drop_zeros: :wilcox,
        pratt: :pratt
      }.freeze

      ALTERNATIVE_ALIASES = {
        one_tail: :one_tail,
        two_tail: :two_sided,
        two_sided: :two_sided,
        "two-sided": :two_sided,
        greater: :greater,
        positive: :greater,
        less: :less,
        negative: :less
      }.freeze

      DIRECTION_ALIASES = {
        greater: :greater,
        positive: :greater,
        right: :greater,
        less: :less,
        negative: :less,
        left: :less
      }.freeze

      def self.perform(alpha, alternative, group_one, group_two = nil,
                       positional_zero_method = nil, direction: nil, **options)
        new.perform(alpha, alternative, group_one, group_two,
                    positional_zero_method, direction: direction, **options)
      end

      # When only one sample is supplied it is interpreted as the paired
      # differences. With two samples, differences are group_one - group_two.
      def perform(alpha, alternative, group_one, group_two = nil,
                  positional_zero_method = nil, direction: nil, **options)
        differences = build_differences(group_one, group_two)
        zero_method = normalize_zero_method(
          options[:zero_method] ||
            options[:zero_rule] ||
            options[:zero_handling] ||
            options[:zero_difference_method] ||
            positional_zero_method ||
            :wilcox
        )
        direction = infer_direction(differences) if direction.nil?
        alternative = normalize_alternative(alternative, direction)

        result = test_statistics(differences, alpha, alternative, zero_method)
        estimation_data = zero_method == :wilcox ? differences.reject(&:zero?) : differences

        point_estimate = median(walsh_averages(estimation_data))
        acceptance_set = if zero_method == :wilcox
                           inversion_acceptance_set(estimation_data, alpha, alternative, zero_method)
                         else
                           inversion_acceptance_set(differences, alpha, alternative, zero_method)
                         end
        confidence_interval = confidence_envelope(acceptance_set, alpha)

        result.merge(
          point_estimate: point_estimate,
          estimate: point_estimate,
          median: point_estimate,
          hodges_lehmann_estimate: point_estimate,
          acceptance_set: acceptance_set,
          acceptance_region: acceptance_set,
          confidence_interval: confidence_interval,
          confidence_level: 1 - alpha
        )
      end

      def signed_ranks(differences, zero_method = :wilcox)
        zero_method = normalize_zero_method(zero_method)
        values = zero_method == :wilcox ? differences.reject(&:zero?) : differences
        ranks = average_ranks(values.map(&:abs))

        values.zip(ranks).map do |value, rank|
          value.zero? ? 0r : value.positive? ? rank : -rank
        end
      end

      alias rank signed_ranks

      private

      def build_differences(group_one, group_two)
        values = Array(group_one).map { |value| Rational(value, 1) }

        if group_two.nil?
          values
        else
          paired = Array(group_two).map { |value| Rational(value, 1) }
          unless values.size == paired.size
            raise ArgumentError, 'both paired samples must have the same size'
          end

          values.zip(paired).map { |left, right| left - right }
        end
      end

      def normalize_alternative(alternative, direction)
        normalized = ALTERNATIVE_ALIASES[alternative.to_s.tr('-', '_').to_sym]
        raise ArgumentError, "unsupported alternative: #{alternative.inspect}" unless normalized
        return normalized unless normalized == :one_tail

        normalized_direction = DIRECTION_ALIASES[direction.to_s.to_sym]
        unless normalized_direction
          raise ArgumentError, "unsupported one-tail direction: #{direction.inspect}"
        end

        normalized_direction
      end

      def infer_direction(differences)
        positive_count = differences.count(&:positive?)
        negative_count = differences.count(&:negative?)

        positive_count >= negative_count ? :greater : :less
      end

      def normalize_zero_method(zero_method)
        normalized_zero_method = zero_method.to_s.tr('-', '_').to_sym
        ZERO_METHOD_ALIASES.fetch(normalized_zero_method) do
          raise ArgumentError, "unsupported zero-difference method: #{zero_method.inspect}"
        end
      end

      def test_statistics(differences, alpha, alternative, zero_method)
        values = zero_method == :wilcox ? differences.reject(&:zero?) : differences
        absolute_ranks = average_ranks(values.map(&:abs))
        nonzero = []
        zero_count = 0

        values.each_with_index do |value, index|
          if value.zero?
            zero_count += 1
          else
            nonzero << { sign: value > 0 ? 1 : -1, rank: absolute_ranks[index] }
          end
        end

        scaled_weights = nonzero.map { |item| (2 * item[:rank]).to_i }
        scaled_w_plus = 2 * nonzero.select { |item| item[:sign] == 1 }.sum(0r) { |item| item[:rank] }
        scaled_w_plus = scaled_w_plus.to_i
        scaled_total_rank = scaled_weights.sum
        distribution = signed_rank_distribution(scaled_weights)
        assignments = 1 << nonzero.size
        if zero_count.positive?
          assignments = 1 << values.size
          distribution = distribution.map { |count| count * (1 << zero_count) }
        end

        greater_count = distribution[scaled_w_plus..scaled_total_rank].sum
        less_count = distribution[0..scaled_w_plus].sum
        greater_p_value = Rational(greater_count, assignments)
        less_p_value = Rational(less_count, assignments)
        two_sided_p_value = [1r, 2 * [greater_p_value, less_p_value].min].min

        w_plus = Rational(scaled_w_plus, 2)
        w_minus = Rational(scaled_total_rank - scaled_w_plus, 2)
        p_value = case alternative
                  when :greater then greater_p_value
                  when :less then less_p_value
                  else two_sided_p_value
                  end

        {
          w: [w_plus, w_minus].min,
          w_plus: w_plus,
          w_minus: w_minus,
          statistic: w_plus,
          signed_ranks: values.zip(absolute_ranks).map do |value, rank|
                           value.zero? ? 0r : value.positive? ? rank : -rank
                         end,
          n: differences.size,
          nonzero_n: nonzero.size,
          zero_n: differences.count(&:zero?),
          p_value: p_value,
          greater_p_value: greater_p_value,
          less_p_value: less_p_value,
          two_sided_p_value: two_sided_p_value,
          alpha: alpha,
          null: alpha < p_value,
          alternative: p_value <= alpha,
          alternative_hypothesis: alternative,
          zero_method: zero_method,
          exact: true
        }
      end

      def average_ranks(values)
        ordered = values.each_with_index.sort_by { |value, _index| value }
        ranks = Array.new(values.size, 0r)

        index = 0
        while index < ordered.size
          value = ordered[index][0]
          last_index = index

          last_index += 1 while last_index + 1 < ordered.size && ordered[last_index + 1][0] == value

          average_rank = ((index + 1)..(last_index + 1)).sum / (last_index - index + 1).to_r
          (index..last_index).each { |position| ranks[ordered[position][1]] = average_rank }
          index = last_index + 1
        end

        ranks
      end

      # Ranks may be half-integers. Multiplying all ranks by two keeps the
      # subset-sum distribution on integers while retaining tied average ranks.
      def signed_rank_distribution(scaled_weights)
        total = scaled_weights.sum
        counts = Array.new(total + 1, 0)
        counts[0] = 1

        scaled_weights.each do |weight|
          total.downto(weight) do |rank_sum|
            counts[rank_sum] += counts[rank_sum - weight]
          end
        end

        counts
      end

      def walsh_averages(values)
        values.each_with_index.flat_map do |value, row|
          values[row..].map { |other| (value + other) / 2r }
        end.sort
      end

      def median(values)
        return nil if values.empty?

        middle = values.size / 2
        if values.size.odd?
          values[middle]
        else
          (values[middle - 1] + values[middle]) / 2r
        end
      end

      def inversion_acceptance_set(values, alpha, alternative, zero_method)
        return [] if values.empty?
        return [{ lower: 0r, upper: 0r, lower_inclusive: true, upper_inclusive: true }] if values.all?(&:zero?)

        candidates = walsh_averages(values).uniq
        return [] if candidates.empty?

        intervals = []
        before_first = candidates.first - (candidates.first.abs + 1)
        after_last = candidates.last + (candidates.last.abs + 1)

        if accepted?(values.map { |value| value - before_first }, alpha, alternative, zero_method)
          intervals << interval(-Float::INFINITY, candidates.first, false,
                               accepted_at?(values, candidates.first, alpha, alternative, zero_method))
        end

        candidates.each do |candidate|
          if accepted_at?(values, candidate, alpha, alternative, zero_method)
            intervals << interval(candidate, candidate, true, true)
          end
        end

        candidates.each_cons(2) do |left, right|
          midpoint = (left + right) / 2r
          next unless accepted?(values.map { |value| value - midpoint }, alpha, alternative, zero_method)

          intervals << interval(left, right, false, false)
        end

        if accepted?(values.map { |value| value - after_last }, alpha, alternative, zero_method)
          intervals << interval(candidates.last, Float::INFINITY, false, false)
        end

        merge_intervals(intervals.sort_by { |item| [item[:lower], item[:upper]] })
      end

      def accepted_at?(values, candidate, alpha, alternative, zero_method)
        accepted?(values.map { |value| value - candidate }, alpha, alternative, zero_method)
      end

      def accepted?(shifted_values, alpha, alternative, zero_method)
        test_statistics(shifted_values, alpha, alternative, zero_method)[:null]
      end

      def interval(lower, upper, lower_inclusive, upper_inclusive)
        {
          lower: lower,
          upper: upper,
          lower_inclusive: lower_inclusive,
          upper_inclusive: upper_inclusive
        }
      end

      def merge_intervals(intervals)
        intervals.each_with_object([]) do |candidate, merged|
          if merged.empty? || !touching_intervals?(merged.last, candidate)
            merged << candidate
          else
            previous = merged.pop
            merged << merged_interval(previous, candidate)
          end
        end
      end

      def touching_intervals?(left, right)
        return true if left[:upper] > right[:lower]
        return false unless left[:upper] == right[:lower]

        left[:upper_inclusive] || right[:lower_inclusive]
      end

      def merged_interval(left, right)
        interval(
          left[:lower],
          right[:upper],
          left[:lower_inclusive],
          right[:upper_inclusive]
        )
      end

      def confidence_envelope(acceptance_set, alpha)
        return nil if acceptance_set.empty?

        first = acceptance_set.first
        last = acceptance_set.last
        contains_rejected_gaps = acceptance_set.size > 1

        {
          lower: first[:lower],
          upper: last[:upper],
          lower_inclusive: first[:lower_inclusive],
          upper_inclusive: last[:upper_inclusive],
          envelope_contains_rejected_gaps: contains_rejected_gaps,
          conservative: true,
          exact: true,
          confidence_level: 1 - alpha
        }
      end
    end

    WilcoxonSignedRank = WilcoxonSignedRankTest
  end
end
