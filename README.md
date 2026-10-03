# Ruby Statistics
![](https://github.com/estebanz01/ruby-statistics/actions/workflows/ruby.yml/badge.svg)

## Note regarding Versions 3.x and 4.x
Hola! :wave: right now the gem got an update that introduced a breaking change in master where the Outermost namespace was changed to be `ruby-statistics` instead of `statistics`. This change is in _master_ already and released in version 4.0. If you're relying on branch instead of gem version, you can checkout branch `3.x`. This branch will be supported but no additional features will be added.

## Note regarding Big Decimal support
This gem also provides basic support for `BigDecimal` and the gem is listed as an optional `development` dependency on the gemspec. Since it's no longer supported as a bundled feature since 3.4.0, the functionality around this gem is going to be limited as well. It is not a required dependency to run or use any of the features developed so far. :grin:

---

A basic ruby gem that implements some statistical methods, functions and concepts to be used in any ruby environment without depending on any mathematical software like `R`, `Matlab`, `Octave` or similar.

Unit test runs under the following ruby versions:
* Ruby 3.2.
* Ruby 3.3.
* Ruby 3.4.
* Ruby 4.0.

We got the inspiration from the folks at [JStat](https://github.com/jstat/jstat) and some interesting lectures about [Keystroke dynamics](http://www.biometric-solutions.com/keystroke-dynamics.html).

Some logic and algorithms are extractions or adaptations from other authors, which are referenced in the comments.
This software is released under the MIT License.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'ruby-statistics'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install ruby-statistics

## Basic Usage

just require the `ruby-statistics` gem in order to load it. If you don't have defined the `Distribution` namespace, the gem will assign an alias, reducing the number of namespaces needed to use a class.

Right now you can load:

* The whole statistics gem. `require 'ruby-statistics'`
* A namespace. `require 'ruby-statistics/distribution'`
* A class. `require 'ruby-statistics/distribution/normal'`

Feel free to use the one that is more convenient to you.

### Hello-World Example
```ruby
require 'ruby-statistics'

poisson = Distribution::Poisson.new(l) # Using Distribution alias.
normal = RubyStatistics::Distribution::StandardNormal.new # Using all namespaces.
```

### Multivariate Normal Rectangular Probability

`MultivariateNormal#rectangular_probability` computes `P(lower_bound <= X <= upper_bound)` for multivariate normal vectors. The current supported dimension is one through six; no external linear-algebra service is required.

```ruby
require 'ruby-statistics'

distribution = RubyStatistics::Distribution::MultivariateNormal.new(
  [0.0, 0.0],
  [[1.0, 0.5], [0.5, 1.0]]
)

result = distribution.rectangular_probability(
  lower_bound: [0.0, 0.0],
  upper_bound: [Float::INFINITY, Float::INFINITY],
  error_budget: 0.0001,
  confidence_level: 0.95,
  seed: 12345,
  max_samples: 100_000
)

result.probability        # Estimated or exact probability
result.estimated_error    # Statistical error convention for randomized results
result.confidence_level   # Wilson score confidence level for randomized results
result.budget_exhausted   # true when max_samples is reached before the budget is met
result.strict_error_bound # true only for deterministic closed-form/special-case results
result.random_seed        # Seed used by randomized integration
result.samples_used       # Number of Monte Carlo samples used
result.rank               # Numerical rank of the covariance matrix
result.rank_deficient     # true when rank is below dimension
result.near_singular      # true for rank deficiency or a condition number above 1e10
```

Boundaries may be finite or `Float::INFINITY`/`-Float::INFINITY`. Zero-variance coordinates are handled as point constraints: the probability is zero when the mean is outside the corresponding interval. Tiny positive variance remains a valid near-singular full-rank coordinate. Rank-deficient covariance matrices are supported without adding diagonal noise; the covariance is decomposed with an internal Jacobi eigendecomposition and probability is computed on the Gaussian support. Tiny negative eigenvalues below the numerical PSD tolerance are treated as zero; materially negative eigenvalues raise `MultivariateNormal::NonPositiveSemidefiniteError`.

Independent variables use the product of univariate normal CDFs, fully correlated rank-one distributions use an interval on the single random direction, and two-dimensional events use the Gaussian quadrant formula plus deterministic adaptive quadrature. Other low-dimensional cases use seeded Monte Carlo on the linear Gaussian representation. Monte Carlo errors are Wilson score-interval half-widths at the requested confidence level, not deterministic or strict upper bounds; `budget_exhausted` is set whenever the available sample count cannot meet `error_budget`.

## Documentation
You can find a bit more detailed documentation of all available distributions, tests and functions in the [Documentation Index](https://github.com/estebanz01/ruby-statistics/wiki)

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/estebanz01/ruby-statistics. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the RubyStatistics project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/estebanz01/ruby-statistics/blob/master/CODE_OF_CONDUCT.md).

## Contact

You can contact me via:
* [Github](https://github.com/estebanz01)
* [Twitter](https://twitter.com/estebanz01)
