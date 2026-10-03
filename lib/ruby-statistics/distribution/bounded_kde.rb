module RubyStatistics
  module Distribution
    class BoundedKDE
      attr_reader :samples, :lower_bound, :upper_bound, :bandwidth, :mirror_tolerance,
                  :max_mirror_terms, :fit_report

      def initialize(samples:, lower_bound:, upper_bound:, bandwidth: nil,
                     bandwidth_bounds: nil, mirror_tolerance: 1e-12,
                     max_mirror_terms: 10_000, grid_size: 64,
                     search_boundary_tolerance: 1e-8)
        @lower_bound = Float(lower_bound)
        @upper_bound = Float(upper_bound)
        @mirror_tolerance = Float(mirror_tolerance)
        @max_mirror_terms = Integer(max_mirror_terms)
        @grid_size = Integer(grid_size)
        @search_boundary_tolerance = Float(search_boundary_tolerance)

        validate_bounds
        @samples = samples.map { |sample| Float(sample) }.tap do |converted|
          validate_samples(converted)
        end

        if bandwidth && bandwidth_bounds
          raise ArgumentError, "specify either bandwidth or bandwidth_bounds, not both"
        elsif bandwidth
          @bandwidth = validate_bandwidth(bandwidth)
          @fit_report = specified_bandwidth_report
        else
          unless bandwidth_bounds
            raise ArgumentError, "bandwidth is required unless bandwidth_bounds is supplied"
          end
          @bandwidth_bounds = validate_bandwidth_bounds(bandwidth_bounds)
          fit_by_leave_one_out
        end

        record_data_warnings
      end

      def self.fit(samples:, lower_bound:, upper_bound:, bandwidth_bounds:, **options)
        new(
          samples: samples,
          lower_bound: lower_bound,
          upper_bound: upper_bound,
          bandwidth_bounds: bandwidth_bounds,
          **options
        )
      end

      def density(value, error: false)
        value = Float(value)
        unless value >= lower_bound && value <= upper_bound
          return error ? result_hash(0.0, 0.0) : 0.0
        end

        relative_value = relative_position(value)
        relative_bandwidth = scaled_bandwidth(bandwidth)
        mirror_count = required_mirror_count(relative_bandwidth, :density, samples.size)
        estimate = reflected_sum(relative_value, relative_bandwidth, mirror_count) do |center, h|
          normal_density((relative_value - center) / h) / h
        end / (samples.size * interval_width)
        bound = tail_error_bound(relative_bandwidth, mirror_count, :density, samples.size)

        error ? result_hash(estimate, bound) : estimate
      end

      def cumulative_function(value, error: false)
        value = Float(value)
        return error ? result_hash(0.0, 0.0) : 0.0 if value <= lower_bound
        return error ? result_hash(1.0, 0.0) : 1.0 if value >= upper_bound

        interval_probability(lower_bound, value, error: error)
      end
      alias cdf cumulative_function

      def interval_probability(range_start, range_end, error: false)
        start_value = Float(range_start)
        end_value = Float(range_end)
        raise ArgumentError, "range_end must be greater than or equal to range_start" if end_value < start_value

        start_value = [start_value, lower_bound].max
        end_value = [end_value, upper_bound].min
        return error ? result_hash(0.0, 0.0) : 0.0 if end_value <= start_value
        return error ? result_hash(1.0, 0.0) : 1.0 if start_value == lower_bound && end_value == upper_bound

        relative_start = relative_position(start_value)
        relative_end = relative_position(end_value)
        relative_bandwidth = scaled_bandwidth(bandwidth)
        mirror_count = required_mirror_count(relative_bandwidth, :cdf, samples.size)
        estimate = reflected_sum(0.0, relative_bandwidth, mirror_count) do |center, h|
          upper_value = normal_cdf((relative_end - center) / h)
          lower_value = normal_cdf((relative_start - center) / h)
          [0.0, upper_value - lower_value].max
        end / samples.size
        estimate = [[estimate, 0.0].max, 1.0].min
        bound = tail_error_bound(relative_bandwidth, mirror_count, :cdf, samples.size)

        error ? result_hash(estimate, bound) : estimate
      end

      def leave_one_out_log_likelihood(selected_bandwidth = bandwidth, error: false)
        result = leave_one_out_result(selected_bandwidth)
        error ? result : result[:log_likelihood]
      end

      private

      def validate_bounds
        unless finite?(lower_bound) && finite?(upper_bound) && lower_bound < upper_bound
          raise ArgumentError, "lower_bound and upper_bound must be finite and ordered"
        end
      end

      def validate_samples(values)
        raise ArgumentError, "samples cannot be empty" if values.empty?
        unless values.all? { |value| finite?(value) && value >= lower_bound && value <= upper_bound }
          raise ArgumentError, "all samples must be finite and inside the bounded interval"
        end
      end

      def validate_bandwidth(value)
        converted = Float(value)
        unless finite?(converted) && converted.positive?
          raise ArgumentError, "bandwidth must be a finite positive number"
        end
        converted
      end

      def validate_bandwidth_bounds(bounds)
        unless bounds.is_a?(Array) && bounds.size == 2
          raise ArgumentError, "bandwidth_bounds must be an array of two positive numbers"
        end

        lower = validate_bandwidth(bounds[0])
        upper = validate_bandwidth(bounds[1])
        unless lower < upper
          raise ArgumentError, "bandwidth_bounds must be strictly ordered"
        end
        [lower, upper]
      end

      def finite?(value)
        value.to_f.finite?
      end

      def interval_width
        upper_bound - lower_bound
      end

      def relative_position(value)
        (value - lower_bound) / interval_width
      end

      def scaled_bandwidth(value)
        value / interval_width
      end

      def normal_density(value)
        Math.exp(-(value**2) / 2.0) / Math.sqrt(2.0 * Math::PI)
      end

      def normal_cdf(value)
        0.5 * (1.0 + Math.erf(value / Math.sqrt(2.0)))
      end

      def reflected_sum(_point, h, mirror_count, excluded_index: nil)
        total = 0.0
        samples.each_with_index do |sample, index|
          next if index == excluded_index

          relative_sample = relative_position(sample)
          (-mirror_count..mirror_count).each do |image_index|
            total += yield(2.0 * image_index + relative_sample, h)
            total += yield(2.0 * image_index - relative_sample, h)
          end
        end
        total
      end

      def required_mirror_count(relative_bandwidth, mode, sample_count)
        (0..max_mirror_terms).each do |mirror_count|
          if tail_error_bound(relative_bandwidth, mirror_count, mode, sample_count) <= mirror_tolerance
            return mirror_count
          end
        end

        raise RuntimeError,
              "could not reduce mirrored-tail error below #{mirror_tolerance} with #{max_mirror_terms} image pairs"
      end

      def tail_error_bound(relative_bandwidth, mirror_count, mode, sample_count)
        first_distance = (2.0 * mirror_count) / relative_bandwidth
        spacing = 2.0 / relative_bandwidth
        decay = Math.exp(-(first_distance * spacing) - (spacing**2) / 2.0)
        geometric_sum = 1.0 / (1.0 - decay)
        first_tail =
          if mode == :density
            normal_density(first_distance) / (relative_bandwidth * interval_width)
          else
            normal_cdf(-first_distance)
          end

        4.0 * first_tail * geometric_sum / sample_count
      end

      def leave_one_out_result(selected_bandwidth)
        selected_bandwidth = validate_bandwidth(selected_bandwidth)
        raise ArgumentError, "leave-one-out requires at least two samples" if samples.size < 2

        h = scaled_bandwidth(selected_bandwidth)
        sample_count = samples.size - 1
        mirror_count = required_mirror_count(h, :density, sample_count)
        densities = samples.each_with_index.map do |sample, excluded_index|
          point = relative_position(sample)
          reflected_sum(point, h, mirror_count, excluded_index: excluded_index) do |center, current_bandwidth|
            normal_density((point - center) / current_bandwidth) / current_bandwidth
          end / (sample_count * interval_width)
        end

        density_error = tail_error_bound(h, mirror_count, :density, sample_count)
        valid_densities = densities.all? { |value| value.finite? && value.positive? }
        log_likelihood =
          if valid_densities
            densities.sum { |value| Math.log(value) } / samples.size
          else
            -Float::INFINITY
          end

        log_error =
          if valid_densities && densities.all? { |value| value > density_error }
            densities.sum { |value| -Math.log(1.0 - density_error / value) } / samples.size
          else
            Float::INFINITY
          end

        {
          log_likelihood: log_likelihood,
          density_error_bound: density_error,
          log_likelihood_error_bound: log_error,
          mirror_count: mirror_count
        }
      end

      def fit_by_leave_one_out
        raise ArgumentError, "bandwidth selection requires at least two samples" if samples.size < 2

        lower, upper = @bandwidth_bounds
        candidates = bandwidth_candidates(lower, upper)
        scores = candidates.to_h { |candidate| [candidate, leave_one_out_result(candidate)] }
        best_index = best_score_index(scores)
        best_bandwidth = candidates[best_index]

        if best_index.positive? && best_index < candidates.size - 1
          best_bandwidth = golden_section_refinement(candidates, best_index, scores)
        end

        selected_result = scores[best_bandwidth] || leave_one_out_result(best_bandwidth)
        @bandwidth = best_bandwidth
        @fit_report = {
          method: :leave_one_out,
          bandwidth: best_bandwidth,
          search_bounds: [lower, upper],
          log_likelihood: selected_result[:log_likelihood],
          log_likelihood_error_bound: selected_result[:log_likelihood_error_bound],
          density_error_bound: selected_result[:density_error_bound],
          maximum_search_log_likelihood_error_bound: maximum_search_error_bound(scores),
          mirror_count: selected_result[:mirror_count],
          evaluations: scores.size,
          optimizer: :grid_search_with_golden_section_refinement,
          grid_size: @grid_size,
          global_maximum_guaranteed: false,
          unique_maximum_guaranteed: false,
          warnings: []
        }
      end

      def bandwidth_candidates(lower, upper)
        raise ArgumentError, "grid_size must be at least 3" if @grid_size < 3

        (0..@grid_size).map do |index|
          lower + (upper - lower) * (index.to_f / @grid_size)
        end
      end

      def best_score_index(scores)
        candidates = scores.keys
        finite_scores = candidates.select { |candidate| scores[candidate][:log_likelihood].finite? }
        if finite_scores.empty?
          raise RuntimeError, "all leave-one-out log-likelihood evaluations were non-finite"
        end

        candidates.index(finite_scores.max_by { |candidate| scores[candidate][:log_likelihood] })
      end

      def golden_section_refinement(candidates, best_index, scores)
        left = candidates[best_index - 1]
        right = candidates[best_index + 1]
        golden = (3.0 - Math.sqrt(5.0)) / 2.0

        50.times do
          inner_left = left + golden * (right - left)
          inner_right = right - golden * (right - left)
          scores[inner_left] ||= leave_one_out_result(inner_left)
          scores[inner_right] ||= leave_one_out_result(inner_right)

          if scores[inner_left][:log_likelihood] > scores[inner_right][:log_likelihood]
            right = inner_right
          else
            left = inner_left
          end
        end

        refined_candidates = [left, right, (left + right) / 2.0, candidates[best_index]]
        refined_candidates.each do |candidate|
          scores[candidate] ||= leave_one_out_result(candidate)
        end
        refined_candidates.max_by { |candidate| scores[candidate][:log_likelihood] }
      end

      def maximum_search_error_bound(scores)
        scores.values.map do |result|
          result[:log_likelihood].finite? ? result[:log_likelihood_error_bound] : Float::INFINITY
        end.max
      end

      def specified_bandwidth_report
        {
          method: :specified_bandwidth,
          bandwidth: bandwidth,
          search_bounds: nil,
          warnings: []
        }
      end

      def record_data_warnings
        warnings = @fit_report[:warnings]
        duplicates = samples.tally.select { |_value, count| count > 1 }
        lower_hits = samples.count { |value| value == lower_bound }
        upper_hits = samples.count { |value| value == upper_bound }

        unless duplicates.empty?
          warnings << :duplicate_observations
          @fit_report[:duplicate_observations] = duplicates
        end

        if lower_hits.positive?
          warnings << :observations_at_lower_bound
          @fit_report[:observations_at_lower_bound] = lower_hits
        end

        if upper_hits.positive?
          warnings << :observations_at_upper_bound
          @fit_report[:observations_at_upper_bound] = upper_hits
        end

        return unless @fit_report[:method] == :leave_one_out

        lower, upper = @fit_report[:search_bounds]
        threshold = @search_boundary_tolerance * (upper - lower)
        near_lower = bandwidth - lower <= threshold
        near_upper = upper - bandwidth <= threshold
        warnings << :bandwidth_near_lower_search_bound if near_lower
        warnings << :bandwidth_near_upper_search_bound if near_upper
        @fit_report[:interior_bandwidth] = !near_lower && !near_upper
      end

      def result_hash(value, error_bound)
        { value: value, truncation_error_bound: error_bound }
      end
    end
  end
end
