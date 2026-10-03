# frozen_string_literal: true

module RubyStatistics
  module Distribution
    class GeneralizedPareto
      SHAPE_EPSILON = 1.0e-10
      GOLDEN_SECTION = 0.3819660112501051

      attr_reader :shape, :scale

      def initialize(shape:, scale:)
        @shape = shape.to_f
        @scale = scale.to_f
      end

      def density_function(value)
        return unless scale.positive? && finite_parameters?
        return 0.0 if value.negative?

        if exponential_limit?
          return 0.0 if value.infinite?

          Math.exp(-value / scale) / scale
        else
          lower = 1.0 + (shape * value / scale)
          return 0.0 if lower < 0.0

          (1.0 / scale) * lower**(-1.0 / shape - 1.0)
        end
      end

      def cumulative_function(value)
        return unless scale.positive? && finite_parameters?
        return 0.0 if value <= 0.0

        if exponential_limit?
          return 1.0 if value.infinite?

          1.0 - Math.exp(-value / scale)
        else
          lower = 1.0 + (shape * value / scale)
          return 0.0 if lower < 0.0

          return 1.0 if shape.negative? && value >= support_endpoint

          1.0 - lower**(-1.0 / shape)
        end
      end

      def quantile_function(probability)
        return unless scale.positive? && finite_parameters?
        return if probability <= 0.0 || probability >= 1.0

        self.class.quantile_for(shape, scale, 1.0 / (1.0 - probability))
      end

      def support_endpoint
        return Float::INFINITY if shape >= 0.0

        scale / -shape
      end

      def self.fit(excesses:, threshold:, exceedance_probability:, return_period: nil, confidence_level: 0.95)
        samples = normalize_samples(excesses)
        invalid = validate_inputs(
          samples,
          Array(excesses).length,
          threshold,
          exceedance_probability,
          return_period,
          confidence_level
        )
        return invalid if invalid

        fit = maximize_likelihood(samples)
        result = {
          status: fit[:shape] ? :ok : fit[:status],
          threshold: threshold.to_f,
          exceedance_probability: exceedance_probability.to_f,
          number_of_excesses: samples.length,
          excesses: samples,
          shape: fit[:shape],
          scale: fit[:scale],
          log_likelihood: fit[:log_likelihood],
          negative_log_likelihood: fit[:negative_log_likelihood],
          confidence_level: confidence_level.to_f,
          termination_reason: fit[:termination_reason],
          iterations: fit[:iterations],
          diagnostics: {
            shape: fit[:shape],
            scale: fit[:scale],
            negative_log_likelihood: fit[:negative_log_likelihood],
            support_satisfied: fit[:shape]&.finite? ? satisfies_support?(samples, fit[:shape], fit[:scale]) : false
          }
        }

        if return_period && fit[:shape]
          add_return_level(result, samples, return_period.to_f, exceedance_probability.to_f)
        elsif return_period
          result[:confidence_interval] = { status: :unavailable, reason: result[:status] }
        end

        result
      end

      def self.profile_return_level(excesses:, target:, shape:, scale:, shape_bounds: nil)
        samples = normalize_samples(excesses)
        if samples.length != Array(excesses).length || samples.length < 2 || samples.any?(&:negative?)
          return failed_result(:data_insufficient, :non_numeric_or_negative_excess)
        end
        if samples.any?(&:zero?)
          return failed_result(:data_insufficient, :excess_must_be_strictly_positive)
        end

        shape = shape.to_f
        scale = scale.to_f
        target = target.to_f
        if !shape.finite? || !scale.finite? || !scale.positive? || !target.finite? || target <= 0.0
          return failed_result(:data_insufficient, :invalid_profile_parameters)
        end
        if shape < -0.5
          return failed_result(:data_insufficient, :shape_below_regularity_boundary)
        end
        unless satisfies_support?(samples, shape, scale)
          return failed_result(:no_finite_optimum, :profile_start_point_outside_support)
        end

        period = if shape.abs < SHAPE_EPSILON
                   Math.exp(target / scale)
                 else
                   (1.0 + (shape * target / scale))**(1.0 / shape)
                 end
        profile = minimize_target_profile(samples, target, period, shape_bounds)
        {
          status: profile[:negative_log_likelihood].finite? ? :ok : :numerical_error,
          target: target,
          conditional_return_period: period,
          shape: profile[:shape],
          scale: profile[:scale],
          log_likelihood: profile[:negative_log_likelihood] ? -profile[:negative_log_likelihood] : nil,
          negative_log_likelihood: profile[:negative_log_likelihood],
          termination_reason: profile[:termination_reason],
          iterations: profile[:iterations],
          diagnostics: {
            actual_target: target,
            support_satisfied: profile[:shape]&.finite? ?
              satisfies_support?(samples, profile[:shape], profile[:scale]) : false
          }
        }
      end

      def self.quantile_for(shape, scale, conditional_period)
        if shape.abs < SHAPE_EPSILON
          scale * Math.log(conditional_period)
        else
          scale * ((conditional_period**shape) - 1.0) / shape
        end
      end

      def self.satisfies_support?(samples, shape, scale)
        return false unless shape.finite? && scale.finite? && scale.positive?

        return samples.all?(&:positive?) if shape >= -SHAPE_EPSILON

        samples.all? { |value| value > 0.0 && value <= scale / -shape }
      end
      private_class_method :satisfies_support?

      def self.failed_result(status, reason)
        {
          status: status,
          shape: nil,
          scale: nil,
          log_likelihood: nil,
          negative_log_likelihood: nil,
          termination_reason: reason,
          iterations: 0,
          diagnostics: {
            actual_negative_log_likelihood: nil,
            support_satisfied: false
          }
        }
      end
      private_class_method :failed_result

      def self.maximize_likelihood(samples)
        minimization = minimize_shape_profile(samples) do |shape|
          profile_scale_result(samples, shape)
        end

        shape = minimization[:shape]
        scale = minimization[:scale]
        nll = minimization[:negative_log_likelihood]

        unless nll.finite? && scale.finite? && scale.positive? && satisfies_support?(samples, shape, scale)
          return failed_result(:no_finite_optimum, :optimizer_left_support_or_diverged)
        end

        if shape.abs < SHAPE_EPSILON
          shape = 0.0
          scale = arithmetic_mean(samples)
          nll = negative_log_likelihood(samples, shape, scale)
        end

        {
          shape: shape,
          scale: scale,
          log_likelihood: -nll,
          negative_log_likelihood: nll,
          termination_reason: minimization[:termination_reason],
          iterations: minimization[:iterations]
        }
      end
      private_class_method :maximize_likelihood

      def self.add_return_level(result, samples, requested_period, probability)
        conditional_period = requested_period * probability

        if samples.length < 3
          result[:return_level_diagnostics] = {
            termination_reason: :data_insufficient,
            actual_target: nil
          }
          result[:confidence_interval] = { status: :unavailable, reason: :need_at_least_three_excesses }
          return
        end

        shape = result[:shape]
        scale = result[:scale]
        excess_level = quantile_for(shape, scale, conditional_period)
        return_level = result[:threshold] + excess_level
        z = InverseStandardNormal.new.cumulative_function((1.0 + result[:confidence_level]) / 2.0)
        cutoff = z * z
        lower = profile_target_endpoint(:lower, samples, conditional_period, excess_level, result[:negative_log_likelihood], cutoff)
        upper = profile_target_endpoint(:upper, samples, conditional_period, excess_level, result[:negative_log_likelihood], cutoff)
        status = interval_status(lower, upper)

        result[:return_period] = requested_period
        result[:conditional_return_period] = conditional_period
        result[:exceedance_return_level] = excess_level
        result[:return_level] = return_level
        result[:return_level_diagnostics] = {
          actual_target: return_level,
          actual_excess_target: excess_level,
          likelihood_drop_cutoff: cutoff,
          termination_reason: status,
          lower_endpoint: lower,
          upper_endpoint: upper
        }
        result[:confidence_interval] = {
          level: result[:confidence_level],
          lower: result[:threshold] + lower[:target],
          upper: upper[:target].infinite? ? Float::INFINITY : result[:threshold] + upper[:target],
          lower_excess: lower[:target],
          upper_excess: upper[:target],
          status: status
        }
      end
      private_class_method :add_return_level

      def self.interval_status(lower, upper)
        if lower[:status] == :ok && upper[:status] == :ok
          :converged
        elsif upper[:status] == :unbounded
          :unbounded
        elsif lower[:status] == :unbounded
          :unbounded
        else
          :numerical_error
        end
      end
      private_class_method :interval_status

      def self.normalize_samples(excesses)
        Array(excesses).each_with_object([]) do |value, samples|
          begin
            numeric = Float(value)
          rescue ArgumentError, TypeError, NoMethodError
            next
          end
          samples << numeric if numeric.finite?
        end
      end
      private_class_method :normalize_samples

      def self.validate_inputs(samples, raw_count, threshold, probability, return_period, confidence_level)
        threshold = Float(threshold) rescue nil
        probability = Float(probability) rescue nil
        confidence_level = Float(confidence_level) rescue nil

        if threshold.nil? || !threshold.finite? || probability.nil? || !probability.finite? ||
           probability <= 0.0 || probability > 1.0
          return failed_result(:data_insufficient, :invalid_threshold_or_exceedance_probability)
        end

        if confidence_level.nil? || confidence_level <= 0.0 || confidence_level >= 1.0
          return failed_result(:data_insufficient, :invalid_confidence_level)
        end

        if raw_count != samples.size || samples.any?(&:negative?)
          return failed_result(:data_insufficient, :non_numeric_or_negative_excess)
        end

        if samples.any?(&:zero?)
          return failed_result(:data_insufficient, :excess_must_be_strictly_positive)
        end

        if samples.length < 2
          return failed_result(:data_insufficient, :need_at_least_two_positive_excesses)
        end

        if return_period
          period = Float(return_period) rescue nil
          if period.nil? || !period.finite? || period <= 0.0
            return failed_result(:data_insufficient, :invalid_return_period)
          end

          if period * probability <= 1.0
            return failed_result(:data_insufficient, :return_period_must_exceed_inverse_exceedance_probability)
          end
        end

        nil
      end
      private_class_method :validate_inputs

      def self.arithmetic_mean(samples)
        samples.sum / samples.length.to_f
      end
      private_class_method :arithmetic_mean

      def self.negative_log_likelihood(samples, shape, scale)
        return Float::INFINITY unless shape.finite? && scale.finite? && scale.positive?
        return Float::INFINITY unless satisfies_support?(samples, shape, scale)

        if shape.abs < SHAPE_EPSILON
          return samples.length * Math.log(scale) + samples.sum / scale
        end

        log_terms = 0.0
        samples.each do |value|
          lower = 1.0 + (shape * value / scale)
          return Float::INFINITY if lower.zero?
          log_terms += Math.log(lower)
        end

        samples.length * Math.log(scale) + ((1.0 / shape) + 1.0) * log_terms
      rescue Math::DomainError
        Float::INFINITY
      end
      private_class_method :negative_log_likelihood

      def self.profile_scale_result(samples, shape)
        if shape.abs < SHAPE_EPSILON
          scale = arithmetic_mean(samples)
          nll = negative_log_likelihood(samples, 0.0, scale)
          return evaluation(0.0, scale, nll)
        end

        scale = profile_scale(samples, shape)
        nll = scale.finite? ? negative_log_likelihood(samples, shape, scale) : Float::INFINITY
        evaluation(shape, scale, nll)
      end
      private_class_method :profile_scale_result

      def self.profile_scale(samples, shape)
        mean = arithmetic_mean(samples)
        maximum = samples.max

        if shape.negative?
          low = (-shape * maximum) * (1.0 + 1.0e-12)
          high = [mean * 2.0, -shape * maximum * 2.0].max
        else
          low = [mean * 0.25, maximum / 1.0e3].max
          high = [mean * 2.0, maximum, mean + maximum].max
        end

        low_score = scale_score(samples, shape, low)
        high_score = scale_score(samples, shape, high)
        return Float::NAN unless low_score.finite? && high_score.finite?

        16.times do
          break if (low_score.negative? && high_score.positive?) ||
                   (low_score.positive? && high_score.negative?)

          if shape.negative?
            low = (-shape * maximum) * (1.0 + 1.0e-10)
            high *= 2.0
          else
            low /= 2.0 if low_score.zero? || (low_score.negative? == high_score.negative?)
            high *= 2.0 if high_score.zero? || (low_score.negative? == high_score.negative?)
          end
          low_score = scale_score(samples, shape, low)
          high_score = scale_score(samples, shape, high)
          return Float::NAN unless low_score.finite? && high_score.finite?
        end
        return Float::NAN unless (low_score.negative? && high_score.positive?) ||
                                 (low_score.positive? && high_score.negative?)

        100.times do
          middle = Math.sqrt(low * high)
          score = scale_score(samples, shape, middle)
          return Float::NAN unless score.finite?

          if (score.negative? && low_score.negative?) ||
             (score.positive? && low_score.positive?)
            low = middle
            low_score = score
          else
            high = middle
            high_score = score
          end
        end

        Math.sqrt(low * high)
      end
      private_class_method :profile_scale

      def self.scale_score(samples, shape, scale)
        inverse_mean = samples.sum { |value|
          1.0 / (1.0 + (shape * value / scale))
        } / samples.length.to_f
        return Float::INFINITY unless inverse_mean.finite?

        ((1.0 / shape) + 1.0) * inverse_mean - (1.0 / shape)
      rescue Math::DomainError, ZeroDivisionError
        Float::INFINITY
      end
      private_class_method :scale_score

      def self.evaluation(shape, scale, nll)
        {
          shape: shape,
          scale: scale,
          negative_log_likelihood: nll
        }
      end
      private_class_method :evaluation

      def self.minimize_shape_profile(samples)
        grid = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
                0.02, 0.03, 0.05, 0.08, 0.15, 0.25, 0.35, 0.5, 0.75,
                1.1, 1.6, 2.3, 3.5, 5.2, 8.0, 12.0, 20.0, 32.0,
                64.0, 128.0, 256.0, 512.0, 1024.0]
        values = grid.map { |shape| [shape, profile_scale_result(samples, shape)] }
        best_shape, best = values.min_by { |_, value| value[:negative_log_likelihood] }

        if best_shape == grid.last
          20.times do
            grid << grid.last * 2.0
            candidate = profile_scale_result(samples, grid.last)
            values << [grid.last, candidate]
            break if candidate[:negative_log_likelihood] > best[:negative_log_likelihood] + 20.0

            best_shape, best = grid.last, candidate
          end
        end

        unless best[:negative_log_likelihood].finite?
          return best.merge(termination_reason: :non_finite_profile, iterations: 0)
        end

        index = grid.index(best_shape)
        left = index.zero? ? 0.0 : grid[index - 1]
        right = index == grid.length - 1 ? grid.last : grid[index + 1]
        minimized = golden_minimize(left, right) { |shape| profile_scale_result(samples, shape) }

        minimized
      end
      private_class_method :minimize_shape_profile

      def self.golden_minimize(left, right)
        inner_left = right - GOLDEN_SECTION * (right - left)
        inner_right = left + GOLDEN_SECTION * (right - left)
        value_left = yield(inner_left)
        value_right = yield(inner_right)
        iterations = 2

        150.times do
          iterations += 1
          if value_left[:negative_log_likelihood] < value_right[:negative_log_likelihood]
            right = inner_right
            inner_right = inner_left
            value_right = value_left
            inner_left = left + GOLDEN_SECTION * (right - left)
            value_left = yield(inner_left)
          else
            left = inner_left
            inner_left = inner_right
            value_left = value_right
            inner_right = right - GOLDEN_SECTION * (right - left)
            value_right = yield(inner_right)
          end

          width = right - left
          next unless width <= 1.0e-10 * [1.0, left.abs, right.abs].max

          best = [value_left, value_right].min_by { |value| value[:negative_log_likelihood] }
          return best.merge(termination_reason: :relative_convergence, iterations: iterations)
        end

        [value_left, value_right].min_by { |value| value[:negative_log_likelihood] }
          .merge(termination_reason: :iteration_limit, iterations: iterations)
      rescue Math::DomainError, ZeroDivisionError
        value_left || evaluation(inner_left, Float::NAN, Float::INFINITY)
      end
      private_class_method :golden_minimize

      def self.profile_target_endpoint(direction, samples, period, mle_target, mle_nll, cutoff)
        if direction == :lower
          bracket = scan_target_bracket(:lower, samples, period, mle_target, mle_nll, cutoff)
        else
          bracket = scan_target_bracket(:upper, samples, period, mle_target, mle_nll, cutoff)
        end

        return bracket unless bracket[:status] == :bracketed

        root_target = profile_target_root(
          bracket[:inner_target],
          bracket[:outer_target],
          direction,
          samples,
          period,
          mle_nll,
          cutoff
        )

        return root_target.merge(status: :numerical_error) unless root_target[:status] == :ok

        root_target
      end
      private_class_method :profile_target_endpoint

      def self.scan_target_bracket(direction, samples, period, mle_target, mle_nll, cutoff)
        factor = direction == :lower ? 0.5 : 2.0
        target = mle_target
        last_result = nil

        90.times do
          target *= factor

          result = minimize_target_profile(samples, target, period)
          if result[:negative_log_likelihood].finite?
            last_result = result.merge(target: target)
          else
            next
          end

          drop = 2.0 * (result[:negative_log_likelihood] - mle_nll)
          if direction == :lower
            next if drop < cutoff

            return {
              status: :bracketed,
              inner_target: target * 2.0,
              outer_target: target
            }
          else
            next if drop < cutoff

            return {
              status: :bracketed,
              inner_target: target / 2.0,
              outer_target: target
            }
          end
        end

        {
          status: :unbounded,
          target: Float::INFINITY,
          shape: nil,
          scale: nil,
          negative_log_likelihood: nil,
          likelihood_drop: nil,
          termination_reason: direction == :lower ? :lower_boundary_not_reached : :upper_interval_unbounded,
          last_evaluated_target: last_result&.dig(:target),
          last_evaluated_negative_log_likelihood: last_result&.dig(:negative_log_likelihood)
        }
      end
      private_class_method :scan_target_bracket

      def self.profile_target_root(inner_target, outer_target, direction, samples, period, mle_nll, cutoff)
        low = [inner_target, outer_target].min
        high = [inner_target, outer_target].max
        best = nil

        100.times do
          middle = Math.sqrt(low * high)
          best = minimize_target_profile(samples, middle, period)
          return best.merge(status: :numerical_error) unless best[:negative_log_likelihood].finite?

          drop = 2.0 * (best[:negative_log_likelihood] - mle_nll)
          below = drop < cutoff
          if direction == :lower
            below ? high = middle : low = middle
          else
            below ? low = middle : high = middle
          end
        end

        target = Math.sqrt(low * high)
        best = minimize_target_profile(samples, target, period)
        drop = 2.0 * (best[:negative_log_likelihood] - mle_nll)
        best.merge(
          status: :ok,
          target: target,
          likelihood_drop: drop,
          cutoff: cutoff,
          termination_reason: :profile_root_relative_convergence
        )
      end
      private_class_method :profile_target_root

      def self.minimize_target_profile(samples, target, period, shape_bounds = nil)
        lower_bound, upper_bound = if shape_bounds.nil?
                                     [nil, nil]
                                   else
                                     Array(shape_bounds).map(&:to_f)
                                   end
        grid = [-0.499, -0.49, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2,
                -0.15, -0.1, -0.08, -0.05, -0.03, -0.02, -0.01, -0.005,
                -0.002, -0.001, -0.0005, -0.0002, -0.0001, 0.0,
                0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01,
                0.02, 0.03, 0.05, 0.08, 0.15, 0.25, 0.35, 0.5, 0.75,
                1.1, 1.6, 2.3, 3.5, 5.2, 8.0, 12.0, 20.0,
                32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
        values = grid.select { |shape| shape >= (lower_bound || -Float::INFINITY) && shape <= (upper_bound || Float::INFINITY) }
                      .map { |shape| [shape, target_profile_result(samples, shape, target, period)] }
                      .select { |_, value| value[:negative_log_likelihood].finite? }
        if lower_bound && lower_bound.finite? && !values.any? { |candidate_shape, _| (candidate_shape - lower_bound).abs < 1.0e-8 }
          values << [lower_bound, target_profile_result(samples, lower_bound, target, period)]
        end
        return evaluation(0.0, Float::NAN, Float::INFINITY).merge(target: target) if values.empty?

        best_shape, best = values.min_by { |_, value| value[:negative_log_likelihood] }

        if best_shape == values.last.first
          20.times do
            next_shape = (values.last.first || 1024.0) * 2.0
            break unless next_shape.finite? && next_shape > values.last.first
            candidate = target_profile_result(samples, next_shape, target, period)
            if candidate[:negative_log_likelihood].finite?
              values << [next_shape, candidate]
              if candidate[:negative_log_likelihood] <= best[:negative_log_likelihood] + 20.0
                best_shape, best = next_shape, candidate
                next
              end
            end
            break
          end
        end

        candidates = values.map(&:first)
        index = candidates.index(best_shape)
        left = index.zero? ? candidates.first : candidates[index - 1]
        right = index == candidates.length - 1 ? candidates.last : candidates[index + 1]
        left = [left, lower_bound].max if lower_bound
        right = [right, upper_bound].min if upper_bound
        result = golden_minimize(left, right) do |shape|
          target_profile_result(samples, shape, target, period)
        end

        unless result[:shape] >= -0.499 &&
               result[:negative_log_likelihood].finite? &&
               result[:scale].finite? && result[:scale].positive? &&
               satisfies_support?(samples, result[:shape], result[:scale])
          return result.merge(termination_reason: :target_profile_left_support_or_diverged)
        end

        result
      end
      private_class_method :minimize_target_profile

      def self.target_profile_result(samples, shape, target, period)
        scale = if shape.abs < SHAPE_EPSILON
                  target / Math.log(period)
                else
                  target * shape / (period**shape - 1.0)
                end
        nll = negative_log_likelihood(samples, shape, scale)
        evaluation(shape, scale, nll).merge(target: target)
      rescue Math::DomainError, ZeroDivisionError
        evaluation(shape, Float::NAN, Float::INFINITY).merge(target: target)
      end
      private_class_method :target_profile_result

      private

      def exponential_limit?
        shape.abs < SHAPE_EPSILON
      end

      def finite_parameters?
        shape.finite? && scale.finite?
      end
    end
  end
end
