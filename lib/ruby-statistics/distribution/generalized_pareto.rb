# frozen_string_literal: true

module RubyStatistics
  module Distribution
    class GeneralizedPareto
      OPTIMIZATION_ITERATIONS = 100
      INTERVAL_ITERATIONS = 80
      BRACKET_ITERATIONS = 120
      NEAR_ZERO = 1.0e-10

      FitResult = Data.define(:converged, :status, :shape, :scale, :log_likelihood,
                              :negative_log_likelihood, :sample_size, :threshold,
                              :exceedance_proportion, :termination_reason,
                              :iterations, :excesses, :diagnostics) do
        def to_h
          {
            converged: converged,
            status: status,
            shape: shape,
            scale: scale,
            log_likelihood: log_likelihood,
            negative_log_likelihood: negative_log_likelihood,
            sample_size: sample_size,
            threshold: threshold,
            exceedance_proportion: exceedance_proportion,
            termination_reason: termination_reason,
            iterations: iterations,
            diagnostics: diagnostics
          }
        end

        def return_level(return_period)
          GeneralizedPareto.send(
            :calculate_return_level, shape, scale, threshold,
            exceedance_proportion, Float(return_period)
          )
        end

        def return_level_profile_interval(return_period:, confidence_level: 0.95)
          period = Float(return_period) rescue nil
          level = Float(confidence_level) rescue nil
          if period.nil? || level.nil?
            return failed_return_level(period, level, :invalid_input,
                                      :invalid_numeric_input)
          end
          unless converged
            return failed_return_level(period, level, :fit_not_converged,
                                      :fit_not_converged)
          end
          if !period.finite? || period <= 0.0
            return failed_return_level(period, level, :invalid_input,
                                      :invalid_return_period)
          end
          unless level.positive? && level < 1.0
            return failed_return_level(period, level, :invalid_input,
                                      :confidence_level_out_of_range)
          end

          multiplier = 1.0 / (exceedance_proportion * period)
          unless multiplier.positive? && multiplier < 1.0
            return failed_return_level(period, level, :invalid_input,
                                      :return_period_too_small)
          end

          fitted_return = GeneralizedPareto.send(
            :calculate_return_level, shape, scale, threshold,
            exceedance_proportion, period
          )
          critical = chi_squared_critical_value(level)
          target_nll = negative_log_likelihood + critical / 2.0
          fitted_profile = constrained_profile(fitted_return, multiplier)
          if fitted_profile.nil? ||
             fitted_profile.negative_log_likelihood > negative_log_likelihood + 1.0e-7
            return failed_return_level(period, level, :optimizer_failed,
                                      :profile_at_estimate_failed,
                                      fitted_return, target_nll)
          end

          lower_endpoint = bracket_and_solve_lower(fitted_return, multiplier,
                                                   target_nll)
          upper_endpoint = bracket_and_solve_upper(fitted_return, multiplier,
                                                   target_nll)

          status = lower_endpoint && upper_endpoint ? :converged : :unbounded_interval
          reason = status == :converged ? :profile_likelihood :
                     (lower_endpoint.nil? ? :lower_endpoint_unbounded :
                      :upper_endpoint_unbounded)

          ReturnLevelResult.new(
            converged: status == :converged,
            status: status,
            return_period: period,
            return_level: fitted_return,
            lower_return_level: lower_endpoint&.return_level || threshold,
            upper_return_level: upper_endpoint&.return_level || Float::INFINITY,
            confidence_level: level,
            critical_value: critical,
            target_negative_log_likelihood: target_nll,
            actual_return_level: fitted_return,
            target_log_likelihood: -target_nll,
            termination_reason: reason,
            fit: self,
            lower_endpoint: lower_endpoint,
            upper_endpoint: upper_endpoint,
            diagnostics: {
              exceedance_multiplier: multiplier,
              profile_negative_log_likelihood: fitted_profile.negative_log_likelihood,
              profile_log_likelihood: fitted_profile.log_likelihood,
              log_likelihood_drop: target_nll - fitted_profile.negative_log_likelihood
            }
          )
        end

        private

        def failed_return_level(period, level, status, reason,
                                return_level = nil, target_nll = nil)
          ReturnLevelResult.new(
            false, status, period, return_level, nil, nil, level, nil,
            target_nll, return_level, target_nll.nil? ? nil : -target_nll,
            reason, self, nil, nil, {}
          )
        end

        def chi_squared_critical_value(level)
          quantile = Distribution::InverseStandardNormal.new.cumulative_function(
            1.0 - (1.0 - level) / 2.0
          )
          quantile ** 2
        end

        def profile_nll(return_level, multiplier)
          point = constrained_profile(return_level, multiplier)
          point&.negative_log_likelihood || Float::INFINITY
        end

        def bracket_and_solve_lower(fitted_return, multiplier, target_nll)
          step = [1.0e-8 * fitted_return, 1.0e-8].max
          lower = threshold + step
          return nil unless profile_nll(lower, multiplier) > target_nll

          GeneralizedPareto::BRACKET_ITERATIONS.times do
            next_step = step * 2.0
            middle_return = threshold + next_step
            break if middle_return >= fitted_return

            if profile_nll(middle_return, multiplier) > target_nll
              step = next_step
              lower = middle_return
            else
              return solve_return_interval(lower, middle_return, multiplier,
                                          target_nll, :lower)
            end
          end

          solve_return_interval(lower, fitted_return, multiplier,
                                target_nll, :lower)
        end

        def bracket_and_solve_upper(fitted_return, multiplier, target_nll)
          step = [1.0e-8 * fitted_return, 1.0e-8].max
          GeneralizedPareto::BRACKET_ITERATIONS.times do
            lower = fitted_return + step
            next_step = step * 2.0
            upper = fitted_return + next_step
            if profile_nll(upper, multiplier) >= target_nll &&
               profile_nll(lower, multiplier) <= target_nll
              return solve_return_interval(lower, upper, multiplier,
                                          target_nll, :upper)
            end

            step = next_step
          end

          nil
        end

        def solve_return_interval(lower, upper, multiplier, target_nll, side)
          low = lower
          high = upper
          GeneralizedPareto::INTERVAL_ITERATIONS.times do
            middle = (low + high) / 2.0
            point = constrained_profile(middle, multiplier)
            return nil if point.nil?

            if side == :lower
              point.negative_log_likelihood > target_nll ? low = middle : high = middle
            else
              point.negative_log_likelihood > target_nll ? high = middle : low = middle
            end
          end

          value = (low + high) / 2.0
          endpoint = constrained_profile(value, multiplier)
          return nil if endpoint.nil?

          deviation = endpoint.negative_log_likelihood - target_nll
          reason = deviation.abs <= 1.0e-5 ? :target_reached : :max_iterations
          Endpoint.new(value, endpoint.shape, endpoint.scale,
                       endpoint.negative_log_likelihood,
                       endpoint.log_likelihood, deviation,
                       GeneralizedPareto::INTERVAL_ITERATIONS, reason)
        end

        def constrained_profile(return_level, multiplier)
          excess_return = return_level - threshold
          return nil unless excess_return.finite? && excess_return.positive?

          lower_shape = constrained_lower_shape(excess_return)
          points = constrained_shape_grid(lower_shape).filter_map do |candidate|
            constrained_point(candidate, excess_return, multiplier)
          end
          best_grid = points.min_by(&:negative_log_likelihood)
          return nil if best_grid.nil?

          refined = refine_constrained(lower_shape, best_grid.shape,
                                       excess_return, multiplier)
          [best_grid, refined].compact.min_by(&:negative_log_likelihood)
        end

        def constrained_lower_shape(excess_return)
          return -0.999999 + 1.0e-8 if excess_return >= excesses.max

          boundary = -excess_return / (excesses.max - excess_return)
          [-0.999999 + 1.0e-8, boundary].max
        end

        def constrained_shape_grid(lower_shape)
          seeds = [shape, -0.99, -0.9, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0,
                   5.0, 10.0, 20.0]
          regular = (-99..400).map { |index| (index + 1) * 0.05 }
          (seeds + regular).select { |value| value > lower_shape }.uniq.sort
        end

        def refine_constrained(lower_shape, center, excess_return, multiplier)
          radius = [0.08, center.abs * 0.35 + 0.08].max
          lower = [lower_shape + 1.0e-7, center - radius].max
          upper = center + radius
          golden_constrained(lower, upper, excess_return, multiplier)
        end

        def golden_constrained(lower, upper, excess_return, multiplier)
          phi = (Math.sqrt(5.0) - 1.0) / 2.0
          inner_low = upper - phi * (upper - lower)
          inner_high = lower + phi * (upper - lower)
          point_low = constrained_point(inner_low, excess_return, multiplier)
          point_high = constrained_point(inner_high, excess_return, multiplier)
          return [point_low, point_high].compact.min_by(&:negative_log_likelihood) if
            point_low.nil? || point_high.nil?

          GeneralizedPareto::OPTIMIZATION_ITERATIONS.times do
            if point_low.negative_log_likelihood < point_high.negative_log_likelihood
              upper = inner_high
              inner_high = inner_low
              point_high = point_low
              inner_low = upper - phi * (upper - lower)
              point_low = constrained_point(inner_low, excess_return, multiplier)
              break if point_low.nil?
            else
              lower = inner_low
              inner_low = inner_high
              point_low = point_high
              inner_high = lower + phi * (upper - lower)
              point_high = constrained_point(inner_high, excess_return, multiplier)
              break if point_high.nil?
            end
          end

          [point_low, point_high].compact.min_by(&:negative_log_likelihood)
        end

        def constrained_point(candidate_shape, excess_return, multiplier)
          scale = GeneralizedPareto.send(
            :scale_for_return, candidate_shape, excess_return, multiplier
          )
          return nil unless scale.finite? && scale.positive?
          return nil unless GeneralizedPareto.send(
            :supports_values?, excesses, candidate_shape, scale
          )

          nll = GeneralizedPareto.send(
            :nll_values, excesses, candidate_shape, scale
          )
          return nil unless nll.finite?

          ProfilePoint.new(candidate_shape, scale, nll, -nll,
                           GeneralizedPareto::OPTIMIZATION_ITERATIONS,
                           :fixed_return_profile)
        end
      end

      ReturnLevelResult = Data.define(:converged, :status, :return_period,
                                      :return_level, :lower_return_level,
                                      :upper_return_level, :confidence_level,
                                      :critical_value,
                                      :target_negative_log_likelihood,
                                      :actual_return_level,
                                      :target_log_likelihood,
                                      :termination_reason, :fit,
                                      :lower_endpoint, :upper_endpoint,
                                      :diagnostics) do
        def to_h
          {
            converged: converged,
            status: status,
            return_period: return_period,
            return_level: return_level,
            lower_return_level: lower_return_level,
            upper_return_level: upper_return_level,
            confidence_level: confidence_level,
            critical_value: critical_value,
            target_negative_log_likelihood: target_negative_log_likelihood,
            actual_return_level: actual_return_level,
            target_log_likelihood: target_log_likelihood,
            termination_reason: termination_reason,
            fit: fit&.to_h,
            lower_endpoint: lower_endpoint&.to_h,
            upper_endpoint: upper_endpoint&.to_h,
            diagnostics: diagnostics
          }
        end
      end

      Endpoint = Data.define(:return_level, :shape, :scale,
                             :negative_log_likelihood, :log_likelihood,
                             :deviation, :iterations, :termination_reason) do
        def to_h
          {
            return_level: return_level,
            shape: shape,
            scale: scale,
            negative_log_likelihood: negative_log_likelihood,
            log_likelihood: log_likelihood,
            deviation: deviation,
            iterations: iterations,
            termination_reason: termination_reason
          }
        end
      end

      ProfilePoint = Data.define(:shape, :scale, :negative_log_likelihood,
                                 :log_likelihood, :iterations,
                                 :termination_reason)

      def initialize(shape:, scale:)
        @shape = Float(shape)
        @scale = Float(scale)
        raise ArgumentError, 'scale must be positive' unless @scale.positive?
        raise ArgumentError, 'shape must be finite' unless @shape.finite?
      end

      attr_reader :shape, :scale

      def self.fit(excesses:, threshold:, exceedance_proportion:)
        new(shape: 0.0, scale: 1.0).send(:fit_gpd, excesses, threshold,
                                         exceedance_proportion)
      end

      def density_function(value)
        x = Float(value)
        return 0.0 unless supported?(x)
        return Math.exp(-x / scale) / scale if shape.abs < NEAR_ZERO

        factor = 1.0 + shape * x / scale
        return 0.0 if factor <= 0.0

        factor ** (-1.0 - 1.0 / shape) / scale
      end

      def cumulative_function(value)
        x = Float(value)
        return 0.0 if x.negative?
        return 1.0 - Math.exp(-x / scale) if shape.abs < NEAR_ZERO

        factor = 1.0 + shape * x / scale
        return 1.0 if factor <= 0.0

        1.0 - factor ** (-1.0 / shape)
      end

      def return_level(return_period:, threshold:, exceedance_proportion:)
        self.class.send(:calculate_return_level, shape, scale, threshold,
                        exceedance_proportion, return_period)
      end

      private

      def fit_gpd(excesses, threshold, proportion)
        values = Array(excesses).map { |value| Float(value) } rescue nil
        threshold = Float(threshold) rescue nil
        proportion = Float(proportion) rescue nil
        if values.nil? || threshold.nil? || proportion.nil?
          safe_values = values || []
          safe_threshold = threshold&.finite? ? threshold : nil
          safe_proportion = proportion&.finite? ? proportion : nil
          return failed_fit(safe_values, safe_threshold.to_f, safe_proportion.to_f,
                            :invalid_input, :invalid_numeric_input)
        end
        unless threshold.finite?
          return failed_fit(values, threshold, proportion, :invalid_input,
                            :threshold_must_be_finite)
        end

        unless proportion.positive? && proportion <= 1.0
          return failed_fit(values, threshold, proportion, :invalid_input,
                            :exceedance_proportion_out_of_range)
        end
        if values.size < 2
          return failed_fit(values, threshold, proportion, :data_insufficient,
                            :fewer_than_two_excesses)
        end
        unless values.all? { |value| value.finite? && value.positive? }
          return failed_fit(values, threshold, proportion, :invalid_input,
                            :excess_must_be_finite_positive)
        end

        points = fit_shape_grid(values).filter_map do |candidate|
          profile_scale(values, candidate)
        end
        best_grid = points.min_by(&:negative_log_likelihood)
        return failed_fit(values, threshold, proportion, :optimizer_failed,
                          :no_profile_point) if best_grid.nil?

        best = refine_shape(values, best_grid.shape) || best_grid
        boundary = values.size * Math.log(values.max)
        boundary_point = ProfilePoint.new(
          -1.0, values.max, boundary, -boundary, OPTIMIZATION_ITERATIONS,
          :closed_support_boundary
        )
        best = [best, boundary_point].min_by(&:negative_log_likelihood)

        if best.shape <= -1.0 + 1.0e-6 && best.negative_log_likelihood >= boundary - 1.0e-6
          return fit_result(best, values, threshold, proportion, false,
                            :no_finite_optimum, :lower_support_boundary)
        end

        fit_result(best, values, threshold, proportion, true, :converged,
                   :bounded_profile_mle)
      end

      def failed_fit(values, threshold, proportion, status, reason)
        FitResult.new(
          converged: false, status: status, shape: nil, scale: nil,
          log_likelihood: nil, negative_log_likelihood: nil,
          sample_size: values.size, threshold: threshold,
          exceedance_proportion: proportion, termination_reason: reason,
          iterations: 0, excesses: values, diagnostics: {}
        )
      end

      def fit_result(point, values, threshold, proportion, converged, status,
                     reason)
        FitResult.new(
          converged: converged, status: status, shape: point.shape,
          scale: point.scale, log_likelihood: point.log_likelihood,
          negative_log_likelihood: point.negative_log_likelihood,
          sample_size: values.size, threshold: threshold,
          exceedance_proportion: proportion, termination_reason: reason,
          iterations: point.iterations, excesses: values,
          diagnostics: {
          objective: point.negative_log_likelihood,
          target: point.log_likelihood,
          scale_termination_reason: point.termination_reason,
          iterations: point.iterations
          }
        )
      end

      def fit_shape_grid(values)
        mean = values.sum / values.size
        variance = values.sum { |value| (value - mean) ** 2 } / values.size
        moment = variance.zero? ? 0.0 : 0.5 * (mean ** 2 / variance - 1.0)
        seeds = [moment, -0.99, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5,
                 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        regular = (-99..300).map { |index| (index + 1) * 0.02 }
        (seeds + regular).select { |value| value > -1.0 }.uniq.sort
      end

      def refine_shape(values, center)
        radius = [0.05, (center.abs + 1.0) * 0.15].max
        golden_minimize(values, center - radius, center + radius)
      end

      def golden_minimize(values, lower, upper)
        lower = [-0.999999, lower].max
        upper = [1.0e-5, upper].max if upper <= lower
        phi = (Math.sqrt(5.0) - 1.0) / 2.0
        inner_low = upper - phi * (upper - lower)
        inner_high = lower + phi * (upper - lower)
        point_low = profile_scale(values, inner_low)
        point_high = profile_scale(values, inner_high)
        return [point_low, point_high].compact.min_by(&:negative_log_likelihood) if
          point_low.nil? || point_high.nil?

        OPTIMIZATION_ITERATIONS.times do
          if point_low.negative_log_likelihood < point_high.negative_log_likelihood
            upper = inner_high
            inner_high = inner_low
            point_high = point_low
            inner_low = upper - phi * (upper - lower)
            point_low = profile_scale(values, inner_low)
          else
            lower = inner_low
            inner_low = inner_high
            point_low = point_high
            inner_high = lower + phi * (upper - lower)
            point_high = profile_scale(values, inner_high)
          end
        end

        [point_low, point_high].compact.min_by(&:negative_log_likelihood)
      end

      def profile_scale(values, candidate_shape)
        scale = optimal_scale(values, candidate_shape)
        return nil unless finite_positive?(scale)
        return nil unless supports?(values, candidate_shape, scale)

        nll = negative_log_likelihood(values, candidate_shape, scale)
        return nil unless nll.finite?

        ProfilePoint.new(candidate_shape, scale, nll, -nll,
                         OPTIMIZATION_ITERATIONS, :scale_score_bisection)
      end

      def optimal_scale(values, candidate_shape)
        return values.sum / values.size if candidate_shape.abs < NEAR_ZERO

        if candidate_shape.negative?
          boundary = -candidate_shape * values.max
          if scale_score(values, candidate_shape, values.max).negative?
            lower = boundary
            upper = values.max
          else
            lower = values.max
            upper = boundary - 1.0e-12 * [boundary, values.max, 1.0].max
          end
        else
          lower = 1.0e-12 * [values.sum / values.size, values.max].min
          upper = values.sum / values.size
          upper *= 2.0 until scale_score(values, candidate_shape, upper).positive?
        end

        lower_score = lower == boundary ? Float::INFINITY : scale_score(values, candidate_shape, lower)
        upper_score = scale_score(values, candidate_shape, upper)
        if candidate_shape.negative?
          return nil unless lower_score.positive? && upper_score.negative?
        else
          return nil unless lower_score.negative? && upper_score.positive?
        end

        OPTIMIZATION_ITERATIONS.times do
          middle = (lower + upper) / 2.0
          score = scale_score(values, candidate_shape, middle)
          if candidate_shape.negative?
            if middle <= boundary
              lower = middle
              next
            end
            score.positive? ? lower = middle : upper = middle
          else
            score.positive? ? upper = middle : lower = middle
          end
        end

        (lower + upper) / 2.0
      end

      def scale_score(values, candidate_shape, scale)
        return 1.0 - values.sum { |value| value / scale } / values.size if
          candidate_shape.abs < NEAR_ZERO

        (1.0 + candidate_shape) *
          values.sum { |value| 1.0 / (1.0 + candidate_shape * value / scale) } /
          values.size - 1.0
      end

      def negative_log_likelihood(values, candidate_shape, scale)
        return nil unless supports?(values, candidate_shape, scale)

        if candidate_shape.abs < NEAR_ZERO
          return values.size * Math.log(scale) +
                 values.sum { |value| value / scale } +
                 candidate_shape * values.sum { |value| (value / scale) ** 2 } / 2.0
        end

        values.size * Math.log(scale) +
          (1.0 + 1.0 / candidate_shape) *
            values.sum { |value| stable_log_one_plus(candidate_shape * value / scale) }
      end

      def supports?(values, candidate_shape, scale)
        return false unless finite_positive?(scale)

        values.all? do |value|
          next false unless value.finite? && value.positive?
          next true if candidate_shape.abs < NEAR_ZERO

          1.0 + candidate_shape * value / scale > 0.0
        end
      end

      def supported?(value)
        return false if value.negative?
        return true if shape.abs < NEAR_ZERO || shape.positive?

        value < -scale / shape
      end

      def stable_log_one_plus(value)
        raise Math::DomainError if value <= -1.0
        return value - value ** 2 / 2.0 + value ** 3 / 3.0 if value.abs < 1.0e-4

        Math.log(1.0 + value)
      end

      def self.calculate_return_level(candidate_shape, scale, threshold,
                                      exceedance_proportion, return_period)
        product = Float(exceedance_proportion) * Float(return_period)
        return nil unless product.finite? && product > 1.0

        multiplier = 1.0 / product
        if candidate_shape.abs < NEAR_ZERO
          Float(threshold) - scale * Math.log(multiplier)
        else
          Float(threshold) + scale / candidate_shape *
            (multiplier ** (-candidate_shape) - 1.0)
        end
      end

      def self.scale_for_return(candidate_shape, excess_return, multiplier)
        return excess_return / multiplier if candidate_shape.abs < NEAR_ZERO

        survival = multiplier ** (-candidate_shape)
        candidate_shape * excess_return / (survival - 1.0)
      end

      def self.supports_values?(values, candidate_shape, scale)
        return false unless scale.respond_to?(:finite?) && scale.finite? && scale.positive?

        values.all? do |value|
          next false unless value.finite? && value.positive?
          next true if candidate_shape.abs < NEAR_ZERO

          1.0 + candidate_shape * value / scale > 0.0
        end
      end

      def self.nll_values(values, candidate_shape, scale)
        return nil unless supports_values?(values, candidate_shape, scale)

        if candidate_shape.abs < NEAR_ZERO
          return values.size * Math.log(scale) +
                 values.sum { |value| value / scale } +
                 candidate_shape * values.sum { |value| (value / scale) ** 2 } / 2.0
        end

        values.size * Math.log(scale) +
          (1.0 + 1.0 / candidate_shape) *
            values.sum do |value|
              stable_log_one_plus(candidate_shape * value / scale)
            end
      end

      def self.stable_log_one_plus(value)
        raise Math::DomainError if value <= -1.0
        return value - value ** 2 / 2.0 + value ** 3 / 3.0 if value.abs < 1.0e-4

        Math.log(1.0 + value)
      end

      def finite_positive?(value)
        value.respond_to?(:finite?) && value.finite? && value.positive?
      end
    end
  end
end
