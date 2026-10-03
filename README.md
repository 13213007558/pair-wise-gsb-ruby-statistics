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

### Multivariate normal rectangle probabilities

`RubyStatistics::Distribution::MultivariateNormal#rectangular_probability` computes
`P(lower <= X <= upper)` for a Gaussian vector. The supported dimension is 1 through 6.

```ruby
require "ruby-statistics"

mean = [0.0, 0.0, 0.0]
covariance = [
  [1.0, 0.5, 0.2],
  [0.5, 1.0, 0.3],
  [0.2, 0.3, 1.0]
]
infinity = Float::INFINITY

result = RubyStatistics::Distribution::MultivariateNormal.rectangular_probability(
  mean,
  covariance,
  [-1.0, -1.0, -infinity],
  [1.0, 1.0, infinity],
  error_budget: 0.001,
  max_evaluations: 50_000,
  seed: 123,
  confidence_level: 0.95
)

result.probability
result.error_estimate       # Randomized confidence half-width, not a strict bound.
result.standard_error
result.confidence_level
result.strict_error_bound
result.budget_exhausted
result.seed
result.rank
result.rank_deficient
result.near_singular
result.zero_variance_dimensions
```

Bounds may be finite numbers or `-Float::INFINITY` / `Float::INFINITY`. A zero
variance coordinate is treated as an exact equality constraint against its mean.
Rank-deficient covariance is accepted when it is positive semidefinite; impossible
linear dependencies have probability zero. Near-singular covariance is not modified
and is reported with `near_singular`.

Asymmetric, non-finite, non-positive-semidefinite, or dimensionally inconsistent
covariance matrices are rejected. The implementation never adds diagonal noise or
any other regularizing term. Small matrix operations are implemented in Ruby and do
not require an external linear algebra service.

Univariate, independent, completely correlated, and ordinary two-dimensional cases
use deterministic formulas. Higher-rank cases use seeded randomized Halton quasi-Monte
Carlo estimation. For randomized results, `error_estimate` is the symmetric confidence
half-width at `confidence_level` and `strict_error_bound` is always false. If the
evaluation budget is consumed before reaching the requested error, `budget_exhausted`
is true.

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
