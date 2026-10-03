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

### Bernoulli confidence sequence (anytime-valid)

For a dashboard that refreshes every time a success or a failure arrives,
`RubyStatistics::ConfidenceSequence::BernoulliConfidenceSequence` reports an
interval that is valid simultaneously for every observed prefix, not just for
one fixed sample size.

```ruby
require 'ruby-statistics'

cs = RubyStatistics::ConfidenceSequence::BernoulliConfidenceSequence.new(
  alpha: 0.05, # miscoverage level, paid once for the whole path
  prior: [     # pre-specified Beta mixture (weights are normalized once)
    { alpha_prior: 1.0, beta_prior: 1.0, weight: 1.0 },
    { alpha_prior: 0.5, beta_prior: 0.5, weight: 1.0 }
  ]
)

cs.success!          # one success, then read the refreshed interval
cs.failure!(2)       # two failures
result = cs.result   # recomputes for the current prefix, never reuses an old interval
result.lower         # => lower endpoint
result.upper         # => upper endpoint
result.sample_count  # => 3
result.config        # => { alpha:, prior: } frozen, cannot change after observing data
```

How it works: the pre-specified Beta mixture builds a likelihood-ratio
e-process that is a non-negative (super)martingale under any fixed Bernoulli
probability. Ville's inequality gives

```
P( exists n >= 0 : true probability is outside the n-th interval ) <= alpha
```

so the guarantee is on the **entire time path** and peeking at the dashboard
at arbitrary times costs nothing extra. Keep in mind:

* Only the success/failure counts are stored; all computations run in the log
  domain, and all successes, all failures and the boundary probabilities
  `p = 0` / `p = 1` are handled explicitly.
* The Beta mixture is only a device to construct the martingale. This is a
  **frequentist confidence sequence**, not a Bayesian credible interval; do
  not interpret the prior or the output as posterior probabilities.
* The configuration (`alpha` and the mixture) must be chosen **before**
  seeing data. It is normalized once at construction and frozen; it cannot
  silently change as observations arrive.
* A fixed-seed coverage simulation is included in the specs as supplementary
  evidence only. The mathematical guarantee comes from the supermartingale,
  and the endpoint checks use high precision direct formulas.

---

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
