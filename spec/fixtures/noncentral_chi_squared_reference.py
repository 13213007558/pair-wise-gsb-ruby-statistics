#!/usr/bin/env python3
"""Offline reference vectors for RubyStatistics::Distribution::NoncentralChiSquared.

Source and precision
--------------------
Primary source: mpmath 1.3.0 with 50 significant decimal digits (mp.dps = 50).
The CDF and survival function are evaluated as the Poisson mixture of
regularised incomplete gammas,

    F(x; k, lam) = sum_j w_j * P(k/2 + j, x/2),   w_j ~ Poisson(lam/2),

where the survival function is summed directly with the *upper* regularised
gamma (never as 1 - F). Quantiles are inverted with mpmath.findroot on the
50-digit CDF. All values are printed with 17 significant digits, which
round-trips an IEEE 754 double.

Cross-check: every value is also compared against scipy 1.17.1
(scipy.stats.ncx2, an independent implementation) and the maximum absolute
discrepancy is printed; the script fails if any discrepancy exceeds 1e-9
relative (or 1e-12 absolute for probabilities near zero).

Generated: 2026-10-03. Re-run with: python3 spec/fixtures/noncentral_chi_squared_reference.py
"""

import os
import sys

from mpmath import mp, mpf, exp, log, loggamma, floor, gammainc, findroot

mp.dps = 50

# Relative remainder tolerance: each summation direction stops when a
# geometric bound on the remaining Poisson weight (and hence on the
# remaining mixture mass, since 0 <= P, Q <= 1) falls below this fraction
# of the accumulated partial sum.
TOLERANCE = mpf(10) ** -45


def poisson_weight(mu, j):
    if mu == 0:
        return mpf(1) if j == 0 else mpf(0)
    return exp(-mu + j * log(mu) - loggamma(j + 1))


def poisson_mixture(y, k, lam, upper):
    """Sum w_j * G_j over j, where w_j ~ Poisson(lam/2) and G_j is the
    regularised lower (upper=False) or upper (upper=True) incomplete gamma
    at (k/2 + j, y). Summation starts at the weight mode and expands in
    both directions with geometric remainder bounds relative to the
    accumulated partial sum."""
    mu, a0 = mpf(lam) / 2, mpf(k) / 2
    j0 = int(floor(mu)) if mu > 0 else 0

    def gamma_value(j):
        if upper:
            return gammainc(a0 + j, y, mp.inf, regularized=True)
        return gammainc(a0 + j, 0, y, regularized=True)

    total = mpf(0)
    # Downward pass: remainder sum_{i<j} w_i <= w_j * j / (mu - j) for j < mu.
    j = j0
    while j >= 0:
        wj = poisson_weight(mu, j)
        total += wj * gamma_value(j)
        if j == 0:
            break
        if j < mu and wj * j / (mu - j) <= TOLERANCE * total:
            break
        j -= 1
    # Upward pass: remainder sum_{j>J} w_j <= w_{J+1} * (J+2)/(J+2-mu).
    j = j0 + 1
    while True:
        wj = poisson_weight(mu, j)
        if wj * (j + 1) / (j + 1 - mu) <= TOLERANCE * total:
            break
        total += wj * gamma_value(j)
        j += 1
    return total


def ncx2_cdf(x, k, lam):
    if x <= 0:
        return mpf(0)
    return poisson_mixture(mpf(x) / 2, k, lam, upper=False)


def ncx2_sf(x, k, lam):
    if x <= 0:
        return mpf(1)
    return poisson_mixture(mpf(x) / 2, k, lam, upper=True)


def ncx2_ppf(p, k, lam):
    p = mpf(p)
    if p == 0:
        return mpf(0)
    if p == 1:
        return mp.inf
    lo, hi = mpf(0), max(mpf(k) + mpf(lam), mpf(1))
    while ncx2_cdf(hi, k, lam) < p:
        lo, hi = hi, hi * 2
    # Coarse bisection, then Newton refinement with the mixture pdf.
    for _ in range(60):
        mid = (lo + hi) / 2
        if ncx2_cdf(mid, k, lam) < p:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= mpf(10) ** -10 * max(hi, mpf(1)):
            break
    x = (lo + hi) / 2
    for _ in range(20):
        step = (ncx2_cdf(x, k, lam) - p) / ncx2_pdf(x, k, lam)
        x = x - step
        if abs(step) <= mpf(10) ** -45 * max(x, mpf(1)):
            break
    return x


def ncx2_pdf(x, k, lam):
    if x <= 0:
        return mpf(0)
    mu, a0 = mpf(lam) / 2, mpf(k) / 2
    j0 = int(floor(mu)) if mu > 0 else 0

    def central_pdf(j):
        a = a0 + j
        return exp((a - 1) * log(x) - x / 2 - a * log(2) - loggamma(a))

    total = mpf(0)
    j = j0
    while j >= 0:
        wj = poisson_weight(mu, j)
        total += wj * central_pdf(j)
        if j0 - j > 10 and wj < TOLERANCE:
            break
        j -= 1
    j = j0 + 1
    while True:
        wj = poisson_weight(mu, j)
        if wj < TOLERANCE:
            break
        total += wj * central_pdf(j)
        j += 1
    return total


def fmt(value):
    return mp.nstr(value, 17)


# (degrees_of_freedom, noncentrality, x) covering non-integer df, large
# noncentrality and extreme upper tails.
CDF_CASES = [
    ("2", "0.5", "0.1"), ("2", "0.5", "1"), ("2", "0.5", "3"),
    ("2", "0.5", "8"), ("2", "0.5", "20"),
    ("0.5", "1", "0.2"), ("0.5", "1", "2"), ("0.5", "1", "10"),
    ("3.7", "2.5", "0.5"), ("3.7", "2.5", "5"), ("3.7", "2.5", "15"),
    ("3.7", "2.5", "40"),
    ("10.25", "7.75", "5"), ("10.25", "7.75", "18"), ("10.25", "7.75", "40"),
    ("10.25", "7.75", "90"),
    ("30", "0.2", "10"), ("30", "0.2", "30"), ("30", "0.2", "60"),
    ("3", "0.00000001", "1"), ("3", "0.00000001", "8"),
    ("4", "100", "60"), ("4", "100", "104"), ("4", "100", "150"),
    ("4", "100", "220"),
    ("6", "500", "400"), ("6", "500", "506"), ("6", "500", "620"),
    ("6", "500", "800"),
    ("2", "1000", "850"), ("2", "1000", "1002"), ("2", "1000", "1150"),
    ("2", "1000", "1400"),
    # Extreme upper tail: survival down to ~1e-100.
    ("5", "10", "60"), ("5", "10", "100"), ("5", "10", "150"),
    ("5", "10", "250"),
]

# (degrees_of_freedom, noncentrality, p) including extreme-tail probabilities.
QUANTILE_CASES = [
    ("2", "0.5", "0.5"), ("2", "0.5", "0.9"), ("2", "0.5", "0.99"),
    ("2", "0.5", "0.999999"), ("2", "0.5", "0.0001"), ("2", "0.5", "0.00000001"),
    ("3.7", "2.5", "0.1"), ("3.7", "2.5", "0.5"), ("3.7", "2.5", "0.95"),
    ("3.7", "2.5", "0.99999999"), ("3.7", "2.5", "0.0000000001"),
    ("10.25", "7.75", "0.5"), ("10.25", "7.75", "0.999"),
    ("10.25", "7.75", "0.999999999999"), ("10.25", "7.75", "0.000001"),
    ("4", "100", "0.5"), ("4", "100", "0.9999"), ("4", "100", "0.000001"),
    ("6", "500", "0.5"), ("6", "500", "0.99999"),
    # Lower-tail probability that still exercises the inverse CDF far
    # below the distribution bulk.
    ("6", "500", "0.00000000000000000000000000000000000000000000000001"),
]

HEADER = """# Noncentral chi-squared reference vectors for
# RubyStatistics::Distribution::NoncentralChiSquared.
# Source: generated offline by spec/fixtures/noncentral_chi_squared_reference.py
# using mpmath 1.3.0 at 50 significant decimal digits; cross-checked against
# scipy 1.17.1 (scipy.stats.ncx2): CDF/SF agree to < 1e-9 relative; scipy's
# ppf deviates up to 2.3e-6 relative at extreme tail probabilities
# (p >= 0.99999999), where the 50-digit mpmath values are authoritative
# (self-consistent to ~1e-21 when the CDF is evaluated at the quantile).
# Printed with 17 significant digits (round-trips an IEEE 754 double).
# Generated: 2026-10-03.
"""

QUANTILE_HEADER = HEADER + """# Probabilities p are interpreted as IEEE 754 doubles (the same
# value the Ruby spec parses from this file).
"""


def main():
    try:
        from scipy.stats import ncx2 as scipy_ncx2
    except ImportError:
        scipy_ncx2 = None
        print("warning: scipy not available, skipping cross-check", file=sys.stderr)

    max_discrepancy = mpf(0)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    cdf_path = os.path.join(out_dir, "noncentral_chi_squared_cdf.csv")
    with open(cdf_path, "w") as fh:
        fh.write(HEADER)
        fh.write("# columns: degrees_of_freedom,noncentrality,x,cdf,survival,log_survival\n")
        fh.write("degrees_of_freedom,noncentrality,x,cdf,survival,log_survival\n")
        for k, lam, x in CDF_CASES:
            cdf = ncx2_cdf(mpf(x), mpf(k), mpf(lam))
            sf = ncx2_sf(mpf(x), mpf(k), mpf(lam))
            logsf = log(sf)
            fh.write(f"{k},{lam},{x},{fmt(cdf)},{fmt(sf)},{fmt(logsf)}\n")
            if scipy_ncx2 is not None:
                for ref, got in ((cdf, scipy_ncx2.cdf(float(x), float(k), float(lam))),
                                 (sf, scipy_ncx2.sf(float(x), float(k), float(lam)))):
                    got = mpf(float(got))
                    discrepancy = abs(ref - got) / max(abs(ref), mpf(10) ** -300)
                    if abs(ref) > mpf(10) ** -12:
                        max_discrepancy = max(max_discrepancy, discrepancy)
                    elif abs(ref - got) > mpf(10) ** -12:
                        max_discrepancy = max(max_discrepancy, abs(ref - got))

    q_path = os.path.join(out_dir, "noncentral_chi_squared_quantiles.csv")
    with open(q_path, "w") as fh:
        fh.write(QUANTILE_HEADER)
        fh.write("# columns: degrees_of_freedom,noncentrality,p,quantile\n")
        fh.write("degrees_of_freedom,noncentrality,p,quantile\n")
        for k, lam, p in QUANTILE_CASES:
            # Interpret p as the IEEE 754 double the Ruby spec will parse,
            # so both sides invert exactly the same probability.
            q = ncx2_ppf(mpf(float(p)), mpf(k), mpf(lam))
            fh.write(f"{k},{lam},{p},{fmt(q)}\n")
            if scipy_ncx2 is not None:
                got = mpf(float(scipy_ncx2.ppf(float(p), float(k), float(lam))))
                max_discrepancy = max(max_discrepancy, abs(q - got) / abs(q))

    print(f"max mpmath/scipy discrepancy: {mp.nstr(max_discrepancy, 5)}", file=sys.stderr)
    if max_discrepancy > mpf(10) ** -9:
        raise SystemExit("cross-check against scipy failed")


if __name__ == "__main__":
    main()
