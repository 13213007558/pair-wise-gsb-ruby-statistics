require "spec_helper"

describe RubyStatistics::Distribution::MultivariateNormal do
  let(:infinity) { Float::INFINITY }

  describe ".rectangular_probability" do
    it "supports an instance-level interface with the same result" do
      distribution = described_class.new([0.0], [[1.0]])
      class_result = described_class.rectangular_probability(
        [0.0], [[1.0]], [-1.0], [1.0], error_budget: 1e-10
      )
      instance_result = distribution.rectangular_probability(
        [-1.0], [1.0], error_budget: 1e-10
      )

      expect(instance_result.probability).to eq(class_result.probability)
      expect(instance_result).to be_a(described_class::Result)
    end

    it "calculates a univariate interval" do
      result = described_class.rectangular_probability(
        [1.0], [[4.0]], [0.0], [3.0], error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-10).of(0.5328072073)
      expect(result.algorithm).to eq(:univariate)
      expect(result.strict_error_bound).to be(true)
      expect(result.budget_exhausted).to be(false)
    end

    it "multiplies univariate probabilities when correlations are zero" do
      one_dimensional = 0.6826894921370859
      result = described_class.rectangular_probability(
        [0.0, 0.0, 0.0],
        Array.new(3) { |row| Array.new(3) { |column| row == column ? 1.0 : 0.0 } },
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-12).of(one_dimensional ** 3)
      expect(result.algorithm).to eq(:independent_product)
      expect(result.strict_error_bound).to be(true)
      expect(result.error_estimate).to eq(0.0)
    end

    it "handles completely positively correlated variables" do
      covariance = [[1.0, 1.0], [1.0, 1.0]]
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        covariance,
        [-1.0, -0.5],
        [1.0, 1.5],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-10).of(0.5328072073)
      expect(result.algorithm).to eq(:rank_one)
      expect(result.rank).to eq(1)
      expect(result.rank_deficient).to be(true)
    end

    it "handles completely negatively correlated variables" do
      covariance = [[1.0, -1.0], [-1.0, 1.0]]
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        covariance,
        [-1.0, -1.0],
        [1.0, 1.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-12).of(0.6826894921370859)
      expect(result.algorithm).to eq(:rank_one)
      expect(result.rank_deficient).to be(true)
    end

    it "detects impossible dependencies in rank-deficient covariance" do
      covariance = [[1.0, 1.0], [1.0, 1.0]]
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        covariance,
        [-1.0, 1.0],
        [1.0, 2.0],
        error_budget: 1e-10
      )

      expect(result.probability).to eq(0.0)
      expect(result.strict_error_bound).to be(true)
    end

    it "evaluates the centered two-dimensional quadrant formula" do
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 0.5], [0.5, 1.0]],
        [-infinity, -infinity],
        [0.0, 0.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-10).of(1.0 / 3.0)
      expect(result.algorithm).to eq(:bivariate_quadrant)
      expect(result.strict_error_bound).to be(false)
      expect(result.error_estimate).to eq(0.0)
    end

    it "calculates a finite correlated rectangle in two dimensions" do
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 0.5], [0.5, 1.0]],
        [-1.0, -1.0],
        [1.0, 1.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-8).of(0.4979717778)
      expect(result.algorithm).to eq(:bivariate_quadrant)
      expect(result.budget_exhausted).to be(false)
    end

    it "matches a non-symmetric two-dimensional rectangle" do
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 0.5], [0.5, 1.0]],
        [-1.0, -1.0],
        [1.0, 0.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-8).of(0.2489858889)
    end

    it "uses randomized integration for dimensions above two" do
      covariance = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
      ]
      result = described_class.rectangular_probability(
        [0.0, 0.0, 0.0],
        covariance,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        error_budget: 0.001,
        max_evaluations: 50_000,
        seed: 7,
        confidence_level: 0.95
      )

      expect(result.probability).to be_within(0.01).of(0.3487)
      expect(result.algorithm).to eq(:randomized_qmc)
      expect(result.seed).to eq(7)
      expect(result.standard_error).to be > 0.0
      expect(result.error_estimate).to be > result.standard_error
      expect(result.confidence_level).to eq(0.95)
      expect(result.strict_error_bound).to be(false)
      expect(result.budget_exhausted).to be(false)
    end

    it "is reproducible for a fixed random seed" do
      covariance = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
      ]
      arguments = [
        [0.0, 0.0, 0.0],
        covariance,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0]
      ]
      first = described_class.rectangular_probability(
        *arguments, error_budget: 0.01, max_evaluations: 1024, seed: 42
      )
      second = described_class.rectangular_probability(
        *arguments, error_budget: 0.01, max_evaluations: 1024, seed: 42
      )

      expect(second.probability).to eq(first.probability)
      expect(second.evaluations_used).to eq(first.evaluations_used)
    end

    it "marks randomized integration as budget-exhausted when the target is not reached" do
      covariance = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0]
      ]
      result = described_class.rectangular_probability(
        [0.0, 0.0, 0.0],
        covariance,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        error_budget: 1e-8,
        max_evaluations: 1024,
        seed: 42
      )

      expect(result.budget_exhausted).to be(true)
      expect(result.error_estimate).to be > 1e-8
      expect(result.strict_error_bound).to be(false)
    end

    it "accepts infinite bounds for an entire space" do
      result = described_class.rectangular_probability(
        [1.0, -2.0],
        [[2.0, 0.5], [0.5, 1.0]],
        [-infinity, -infinity],
        [infinity, infinity],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-12).of(1.0)
    end

    it "uses randomized integration and marks near-singular covariance" do
      gap = 1e-12
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 1.0 - gap], [1.0 - gap, 1.0]],
        [-1.0, -1.0],
        [1.0, 1.0],
        error_budget: 0.01,
        max_evaluations: 1024,
        seed: 3
      )

      expect(result.algorithm).to eq(:randomized_qmc)
      expect(result.near_singular).to be(true)
      expect(result.strict_error_bound).to be(false)
      expect(result.probability).to be_within(1e-6).of(0.6826894921)
    end

    it "supports a feasible deterministic zero-variance coordinate" do
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0]],
        [-1.0, 0.0],
        [1.0, 0.0],
        error_budget: 1e-10
      )

      expect(result.probability).to be_within(1e-12).of(0.6826894921370859)
      expect(result.zero_variance_dimensions).to eq([1])
      expect(result.rank_deficient).to be(true)
      expect(result.near_singular).to be(false)
    end

    it "returns zero for an infeasible deterministic coordinate" do
      result = described_class.rectangular_probability(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 0.0]],
        [-1.0, 1.0],
        [1.0, 2.0],
        error_budget: 1e-10
      )

      expect(result.probability).to eq(0.0)
      expect(result.algorithm).to eq(:degenerate_constraint)
      expect(result.zero_variance_dimensions).to eq([1])
    end

    it "rejects dimensions above six" do
      expect do
        described_class.new(Array.new(7, 0.0), Array.new(7) { Array.new(7, 1.0) })
      end.to raise_error(ArgumentError, /1 to 6/)
    end

    it "rejects non-positive-semidefinite covariance without regularizing it" do
      expect do
        described_class.new([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])
      end.to raise_error(ArgumentError, "covariance must be positive semidefinite")
    end

    it "rejects asymmetric covariance" do
      expect do
        described_class.new([0.0, 0.0], [[1.0, 0.5], [0.2, 1.0]])
      end.to raise_error(ArgumentError, "covariance must be symmetric")
    end

    it "rejects a zero variance that is still correlated with another coordinate" do
      expect do
        described_class.new([0.0, 0.0], [[0.0, 1.0], [1.0, 1.0]])
      end.to raise_error(ArgumentError, "covariance must be positive semidefinite")
    end

    it "rejects non-positive error budgets" do
      expect do
        described_class.rectangular_probability(
          [0.0], [[1.0]], [-1.0], [1.0], error_budget: 0
        )
      end.to raise_error(ArgumentError, "error_budget must be a finite positive number")
    end

    it "rejects confidence levels outside the open unit interval" do
      expect do
        described_class.rectangular_probability(
          [0.0, 0.0, 0.0],
          Array.new(3) { |row| Array.new(3) { |column| row == column ? 1.0 : 0.0 } },
          [-1.0] * 3,
          [1.0] * 3,
          error_budget: 1e-8,
          confidence_level: 1.0
        )
      end.to raise_error(ArgumentError, "confidence_level must be between 0 and 1")
    end
  end
end
