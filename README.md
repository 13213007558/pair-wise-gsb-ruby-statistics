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

## Documentation
You can find a bit more detailed documentation of all available distributions, tests and functions in the [Documentation Index](https://github.com/estebanz01/ruby-statistics/wiki)

### Bounded kernel density estimation

Samples that live on a finite interval make a plain Gaussian KDE leak
probability mass outside the support. `Distribution::BoundedKernelDensity`
avoids this with the reflection (method of images) technique: every
observation `x` on `[lower, upper]` (of length `L`) is mirrored to
`x + 2*k*L` and `2*lower - x + 2*k*L` for integer `k`, and the truncated
mirror sum is renormalised so the density integrates to exactly one on the
interval and is zero outside of it.

```ruby
kde = Distribution::BoundedKernelDensity.new(
  [0.2, 0.4, 0.5, 0.7, 0.9],
  lower: 0.0, upper: 1.0,
  bandwidth: 0.15 # or bandwidth: :loo, bandwidth_interval: [0.01, 0.5]
)

kde.density_function(0.5)   # density, 0.0 outside [lower, upper]
kde.cumulative_function(2)  # => 1.0, exactly 0 below lower and 1 above upper
kde.probability(0.2, 0.6)   # interval probability, clipped to the support
```

The infinite mirror series is truncated to `|k| <= reflections` and the
omitted tail is bounded explicitly. `kde.truncation_error` returns the
bounds: with `e_d` the omitted density per observation, `e_m` the omitted
mass, `Z` the normalisation and `f`/`F` the returned density/CDF, the
infinite-mirror values `f*`/`F*` satisfy
`|f* - f| <= (e_d + f * e_m) / Z` and `|F* - F| <= 2 * e_m / Z`.
Interval probabilities inherit twice the cumulative bound. The number of
reflections is chosen automatically from `tolerance:` (default `1e-10`) or
can be forced with `reflections:`.

With `bandwidth: :loo`, the bandwidth maximises the leave-one-out
log-likelihood over the caller supplied positive `bandwidth_interval`.
Each left-out observation is removed together with all of its mirror
images. The search (golden section plus an explicit comparison against the
interval edges) does not guarantee a unique interior optimum:
`kde.diagnostics` and the returned `BandwidthSelection` report duplicate
observations (ties inflate the score at small bandwidths), observations
exactly on the boundaries, and optima lying at the edges of the search
interval (`at_lower_bound?` / `at_upper_bound?`, `interior_optimum?`).

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
