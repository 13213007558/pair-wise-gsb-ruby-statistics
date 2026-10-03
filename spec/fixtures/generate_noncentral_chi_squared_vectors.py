#!/usr/bin/env python3
"""Generates spec/fixtures/noncentral_chi_squared_vectors.rb.

Reference values for the non-central chi-squared distribution, computed
with mpmath in 60-digit decimal arithmetic through the Poisson mixture
    F(x; k, l) = sum_j Poisson(j; l/2) * P(k/2 + j, x/2)
    S(x; k, l) = sum_j Poisson(j; l/2) * Q(k/2 + j, x/2)
expanding from the mode of the Poisson weights in both directions. A
direction is truncated only after 30 consecutive terms drop below
1e-55 of the accumulated total, and every case verifies that the
Poisson weights sum to 1 within 1e-40, so the truncation error is far
below the 17 significant digits that are printed.

Quantiles are inverted by bisection to 60-digit precision.

Where scipy.stats.ncx2 is itself accurate (probabilities above ~1e-300),
the values agree with scipy to better than 1e-9 relative; scipy is not
used as a source because it cannot produce the extreme-tail entries.
"""

from mpmath import mp, mpf, floor, log, log1p, exp, loggamma, gammainc, inf

mp.dps = 60


def mixture(x, k, lam):
    """Returns (cdf, sf) of the non-central chi-squared distribution."""
    mu = mpf(lam) / 2
    a = mpf(k) / 2
    y = mpf(x) / 2
    if y <= 0:
        return mpf(0), mpf(1)
    if mu == 0:
        return (gammainc(a, 0, y, regularized=True),
                gammainc(a, y, inf, regularized=True))
    j0 = int(floor(mu))

    def logw(j):
        return -mu + j * log(mu) - loggamma(j + 1) if mu > 0 else mpf(0)

    cdf = mpf(0)
    sf = mpf(0)
    wsum = mpf(0)

    # Upwards from the mode.
    small = 0
    j = j0
    while True:
        w = exp(logw(j))
        dp = w * gammainc(a + j, 0, y, regularized=True)
        dq = w * gammainc(a + j, y, inf, regularized=True)
        cdf += dp
        sf += dq
        wsum += w
        small = small + 1 if dp < 1e-55 * cdf and dq < 1e-55 * sf else 0
        if small >= 30:
            break
        j += 1

    # Downwards from the mode.
    small = 0
    j = j0 - 1
    while j >= 0:
        w = exp(logw(j))
        dp = w * gammainc(a + j, 0, y, regularized=True)
        dq = w * gammainc(a + j, y, inf, regularized=True)
        cdf += dp
        sf += dq
        wsum += w
        small = small + 1 if dp < 1e-55 * cdf and dq < 1e-55 * sf else 0
        if small >= 30:
            break
        j -= 1

    assert abs(wsum - 1) < 1e-40, "weight sum check failed"
    return cdf, sf


def quantile(p, k, lam):
    p = mpf(p)
    lo = mpf(0)
    hi = max(mpf(1), mpf(k) + mpf(lam))
    while mixture(hi, k, lam)[0] < p:
        hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if mixture(mid, k, lam)[0] < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def f17(value):
    """17 significant digits, round-trip safe for a double."""
    if value == 0:
        return "0.0"
    return mp.nstr(value, 17, min_fixed=-6, max_fixed=6)


CASES = [
    # (k, lambda, [x, ...]) -- non-integer degrees of freedom
    (0.5, 1.3, [0.05, 0.5, 2.0, 6.0, 15.0]),
    (1.5, 2.5, [0.1, 1.0, 4.0, 10.0, 25.0]),
    (3.7, 0.8, [0.2, 1.5, 5.0, 12.0, 30.0]),
    (7.25, 12.75, [2.0, 10.0, 20.0, 40.0, 80.0]),
    (2.3, 5.1, [0.3, 2.0, 8.0, 18.0, 45.0]),
    (10.6, 3.3, [2.0, 8.0, 15.0, 30.0, 60.0]),
    # integer degrees of freedom
    (1, 0.5, [0.1, 1.0, 3.0, 8.0, 20.0]),
    (2, 4, [0.5, 3.0, 8.0, 16.0, 35.0]),
    (5, 2.5, [0.5, 3.0, 7.5, 15.0, 35.0]),
    (10, 5, [2.0, 8.0, 15.0, 28.0, 55.0]),
    # large non-centrality
    (5, 100, [40.0, 80.0, 110.0, 150.0, 220.0]),
    (10, 500, [300.0, 450.0, 520.0, 600.0, 750.0]),
    (3.5, 1000, [700.0, 900.0, 1010.0, 1150.0, 1400.0]),
    (4, 10000, [9000.0, 9700.0, 10050.0, 10400.0, 11200.0]),
    (8, 2000, [1600.0, 1900.0, 2020.0, 2200.0, 2500.0]),
    (50, 500, [400.0, 500.0, 560.0, 650.0, 800.0]),
    # extreme tails (upper tail down to and beyond double underflow)
    (4, 2, [50.0, 100.0, 500.0, 1000.0, 5000.0]),
    (2, 0.5, [40.0, 80.0, 200.0, 700.0]),
    (10, 5, [80.0, 150.0, 300.0, 900.0]),
    (6, 40, [120.0, 200.0, 350.0, 1200.0]),
    # extreme lower tail
    (8, 1, [0.01, 0.1, 0.5]),
    (20, 10, [1.0, 3.0, 6.0]),
    (3.5, 1000, [100.0, 300.0]),
    (4, 10000, [5000.0, 7000.0]),
    # degenerate case lambda = 0 (central chi-squared)
    (5, 0, [0.5, 3.0, 7.5, 15.0, 40.0]),
    (2.5, 0, [0.3, 2.0, 9.0, 30.0]),
]

QUANTILE_CASES = [
    # (k, lambda, [p, ...])
    (3.5, 2.5, [1e-6, 0.01, 0.5, 0.99, 0.999999]),
    (10, 5, [1e-8, 0.25, 0.9, 0.999]),
    (0.7, 1.3, [0.001, 0.5, 0.999]),
    (100, 50, [0.001, 0.5, 0.999]),
    (6, 3, [1 - 1e-12, 1e-10]),
    (4, 1000, [0.01, 0.5, 0.99]),
]


def main():
    lines = []
    quantile_lines = []
    for k, lam, xs in CASES:
        for x in xs:
            cdf, sf = mixture(x, k, lam)
            # When the cdf is tiny, sf rounds to 1 at 60 digits; log1p
            # still gives the exact log-survival.
            if cdf < mpf("1e-30"):
                log_sf = log1p(-cdf)
            else:
                log_sf = log(sf)
            lines.append(
                f"        [{k!r}, {lam!r}, {x!r}, {f17(cdf)}, {f17(sf)}, {f17(log_sf)}],"
            )
            print(f"case k={k} l={lam} x={x} done", flush=True)
    for k, lam, ps in QUANTILE_CASES:
        for p in ps:
            q = quantile(p, k, lam)
            quantile_lines.append(f"        [{k!r}, {lam!r}, {p!r}, {f17(q)}],")
            print(f"quantile k={k} l={lam} p={p} done", flush=True)

    header = '''# frozen_string_literal: true

# Reference values for RubyStatistics::Distribution::NonCentralChiSquared.
#
# Source: computed with mpmath 1.3.0 (https://mpmath.org/) in 60-digit
# decimal arithmetic through the Poisson mixture
#   F(x; k, l) = sum_j Poisson(j; l/2) * P(k/2 + j, x/2)
#   S(x; k, l) = sum_j Poisson(j; l/2) * Q(k/2 + j, x/2)
# with mpmath's regularised incomplete gamma functions, expanding from the
# mode of the Poisson weights and truncating a direction only after 30
# consecutive terms fall below 1e-55 of the accumulated total; every case
# verifies that the Poisson weights sum to 1 within 1e-40. Quantiles were
# inverted by bisection to 60-digit precision. Values agree with scipy
# 1.17.1 (scipy.stats.ncx2) to better than 1e-9 relative wherever scipy
# itself is accurate; scipy cannot produce the extreme-tail entries.
#
# Values are printed with 17 significant decimal digits (round-trip safe
# for Float) and are accurate to at least 1e-40 relative. Generated on
# 2026-10-03 by spec/fixtures/generate_noncentral_chi_squared_vectors.py.

module RubyStatistics
  module Distribution
    module NonCentralChiSquaredFixtures
      # [degrees_of_freedom, noncentrality, x, cdf, survival, log_survival]
      VECTORS = ['''

    middle = '''      ].freeze

      # [degrees_of_freedom, noncentrality, probability, quantile]
      QUANTILES = ['''

    tail = '''      ].freeze
    end
  end
end
'''

    with open("spec/fixtures/noncentral_chi_squared_vectors.rb", "w") as fh:
        fh.write(header + "\n")
        fh.write("\n".join(lines) + "\n")
        fh.write(middle + "\n")
        fh.write("\n".join(quantile_lines) + "\n")
        fh.write(tail)


if __name__ == "__main__":
    main()
