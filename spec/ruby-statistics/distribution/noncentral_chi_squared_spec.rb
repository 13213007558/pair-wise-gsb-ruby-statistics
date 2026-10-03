require 'spec_helper'

describe RubyStatistics::Distribution::NoncentralChiSquared do
  FIXTURE_DIR = File.join(__dir__, '..', '..', 'fixtures')

  def load_fixture(name)
    lines = File.readlines(File.join(FIXTURE_DIR, name))
                .map(&:strip)
                .reject { |line| line.empty? || line.start_with?('#') }
    lines.shift # column header
    lines.map { |line| line.split(',').map { |field| Float(field) } }
  end

  def expect_close(got, ref, rel: 1.0e-9, abs: 1.0e-12)
    expect((got - ref).abs).to be <= [abs, rel * ref.abs].max
  end

  describe '#initialize' do
    it 'accepts positive, possibly non-integer degrees of freedom' do
      expect { described_class.new(0.5, 1.0) }.not_to raise_error
      expect { described_class.new(10.25, 0.0) }.not_to raise_error
    end

    it 'rejects non-positive degrees of freedom' do
      expect { described_class.new(0, 1.0) }.to raise_error(ArgumentError)
      expect { described_class.new(-2, 1.0) }.to raise_error(ArgumentError)
    end

    it 'rejects negative noncentrality' do
      expect { described_class.new(2, -0.5) }.to raise_error(ArgumentError)
    end
  end

  describe '#mean and #variance' do
    it 'follows the k + lambda and 2(k + 2 lambda) identities' do
      distribution = described_class.new(3.7, 2.5)

      expect(distribution.mean).to eq 6.2
      expect(distribution.variance).to eq 17.4
    end
  end

  describe 'offline reference vectors' do
    # Vectors come from spec/fixtures/noncentral_chi_squared_cdf.csv and
    # spec/fixtures/noncentral_chi_squared_quantiles.csv; see the header
    # comments in those files for source (mpmath 1.3.0, 50 significant
    # digits, cross-checked against scipy 1.17.1) and precision.
    let(:cdf_rows) { load_fixture('noncentral_chi_squared_cdf.csv') }
    let(:quantile_rows) { load_fixture('noncentral_chi_squared_quantiles.csv') }

    it 'matches the reference CDF values' do
      cdf_rows.each do |k, lambda, x, cdf, _sf, _logsf|
        got = described_class.new(k, lambda).cumulative_function(x)

        expect_close(got, cdf, rel: 1.0e-9, abs: 1.0e-12)
      end
    end

    it 'matches the reference survival values, including the extreme upper tail' do
      cdf_rows.each do |k, lambda, x, _cdf, sf, _logsf|
        got = described_class.new(k, lambda).survival_function(x)

        expect_close(got, sf, rel: 1.0e-8, abs: 1.0e-12)
      end
    end

    it 'matches the reference log-survival values, including probabilities that underflow a double' do
      cdf_rows.each do |k, lambda, x, _cdf, _sf, logsf|
        got = described_class.new(k, lambda).log_survival_function(x)

        expect(got).to be_within(1.0e-8).of(logsf)
      end
    end

    it 'matches the reference quantiles' do
      quantile_rows.each do |k, lambda, p, quantile|
        got = described_class.new(k, lambda).inverse_cumulative_function(p)

        expect_close(got, quantile, rel: 1.0e-7, abs: 1.0e-9)
      end
    end
  end

  describe 'degenerate noncentrality' do
    it 'coincides with the central chi-squared distribution when lambda is zero' do
      [1, 2, 3.5, 10].each do |k|
        [0.1, 0.5, 2.0, 8.0, 30.0].each do |x|
          noncentral = described_class.new(k, 0).cumulative_function(x)
          central = RubyStatistics::Distribution::ChiSquared.new(k).cumulative_function(x)

          expect(noncentral).to be_within(1.0e-12).of(central)
        end
      end
    end

    it 'approaches the central chi-squared distribution as lambda tends to zero' do
      [1, 2, 3.5, 10].each do |k|
        [0.5, 2.0, 8.0].each do |x|
          noncentral = described_class.new(k, 1.0e-10).cumulative_function(x)
          central = RubyStatistics::Distribution::ChiSquared.new(k).cumulative_function(x)

          expect(noncentral).to be_within(1.0e-8).of(central)
        end
      end
    end

    it 'reduces to the closed form 1 - exp(-x/2) for k = 2 and lambda = 0' do
      distribution = described_class.new(2, 0)

      [0.3, 1.0, 4.0, 12.0].each do |x|
        expect(distribution.cumulative_function(x)).to be_within(1.0e-12).of(1.0 - Math.exp(-x / 2.0))
      end
    end
  end

  describe 'mathematical identities' do
    it 'satisfies CDF + survival = 1 with both tails evaluated independently' do
      [[2, 0.5], [3.7, 2.5], [10.25, 7.75], [4, 100], [6, 500]].each do |k, lambda|
        distribution = described_class.new(k, lambda)

        [0.2, 1.0, 5.0, 20.0, 60.0].each do |x|
          sum = distribution.cumulative_function(x) + distribution.survival_function(x)

          expect(sum).to be_within(1.0e-12).of(1.0)
        end
      end
    end

    it 'satisfies log_survival = log(survival) whenever the survival is representable' do
      [[2, 0.5], [3.7, 2.5], [10.25, 7.75]].each do |k, lambda|
        distribution = described_class.new(k, lambda)

        [0.2, 1.0, 5.0, 20.0].each do |x|
          survival = distribution.survival_function(x)

          expect(distribution.log_survival_function(x)).to be_within(1.0e-10).of(Math.log(survival))
        end
      end
    end

    it 'inverts the CDF on a grid of probabilities' do
      [[2, 0.5], [3.7, 2.5], [10.25, 7.75], [4, 100]].each do |k, lambda|
        distribution = described_class.new(k, lambda)

        [0.001, 0.1, 0.5, 0.9, 0.999].each do |p|
          x = distribution.inverse_cumulative_function(p)

          expect(distribution.cumulative_function(x)).to be_within(1.0e-9).of(p)
        end
      end
    end

    it 'recovers x from the CDF through the inverse CDF' do
      [[2, 0.5], [3.7, 2.5], [6, 500]].each do |k, lambda|
        distribution = described_class.new(k, lambda)

        [0.5, 2.0, 10.0, 50.0].each do |x|
          p = distribution.cumulative_function(x)
          # In the far tails p saturates at 0.0/1.0 or carries ~1e-16
          # absolute float noise, which dominates any inverse roundtrip.
          next if p < 1.0e-6 || p > 1.0 - 1.0e-6

          recovered = distribution.inverse_cumulative_function(p)

          expect(recovered).to be_within(1.0e-7 * [x, 1.0].max).of(x)
        end
      end
    end
  end

  describe 'upper tail behaviour' do
    it 'does not compute the survival function as 1 - CDF' do
      # For x = 150 with k = 5, lambda = 10 the CDF saturates at 1.0 within
      # double precision, so 1 - CDF would report (at best) a meaningless
      # ~1e-16 while the true survival probability is ~7.7e-19.
      distribution = described_class.new(5, 10)

      expect(distribution.cumulative_function(150)).to be >= 0.99999999999999
      expect(distribution.survival_function(150)).to be_within(1.0e-8 * 7.7004889477864627e-19).of(7.7004889477864627e-19)
      expect(distribution.log_survival_function(150)).to be_within(1.0e-8).of(-41.707832940330724)
    end

    it 'keeps a finite log-survival where the survival itself underflows a double' do
      # True survival probability here is ~1.4e-35, still representable; push
      # further to check the log stays finite well past 1e-308.
      distribution = described_class.new(5, 10)

      expect(distribution.log_survival_function(250)).to be_within(1.0e-8).of(-80.254028473017115)
      expect(distribution.log_survival_function(1000)).to be < -300.0
      expect(distribution.log_survival_function(1000)).to be > -Float::INFINITY
    end
  end

  describe 'support boundaries' do
    it 'returns zero CDF, one survival and zero log-survival at or below zero' do
      distribution = described_class.new(3, 1.5)

      expect(distribution.cumulative_function(0)).to eq 0.0
      expect(distribution.cumulative_function(-1)).to eq 0.0
      expect(distribution.survival_function(0)).to eq 1.0
      expect(distribution.survival_function(-1)).to eq 1.0
      expect(distribution.log_survival_function(0)).to eq 0.0
      expect(distribution.log_survival_function(-1)).to eq 0.0
    end
  end

  describe '#inverse_cumulative_function' do
    let(:distribution) { described_class.new(4, 2) }

    it 'returns the support endpoints for probabilities 0 and 1' do
      expect(distribution.inverse_cumulative_function(0)).to eq 0.0
      expect(distribution.inverse_cumulative_function(1)).to eq Float::INFINITY
    end

    it 'returns nil for probabilities outside [0, 1]' do
      expect(distribution.inverse_cumulative_function(-0.1)).to be_nil
      expect(distribution.inverse_cumulative_function(1.1)).to be_nil
      expect(distribution.inverse_cumulative_function(Float::NAN)).to be_nil
    end

    it 'distinguishes endpoints from algorithmic failure' do
      # Endpoints return values; exhausting the series term cap raises an
      # explicit non-convergence error instead of silently returning a
      # wrong quantile.
      stub_const('RubyStatistics::Distribution::NoncentralChiSquared::MIXTURE_MAX_TERMS', 50)

      expect { described_class.new(2, 400).inverse_cumulative_function(0.5) }
        .to raise_error(described_class::NotConvergedError)
    end
  end

  describe 'series non-convergence' do
    it 'raises NotConvergedError when the Poisson mixture exceeds the term cap' do
      stub_const('RubyStatistics::Distribution::NoncentralChiSquared::MIXTURE_MAX_TERMS', 50)

      expect { described_class.new(2, 400).cumulative_function(200) }
        .to raise_error(described_class::NotConvergedError)
    end
  end
end
