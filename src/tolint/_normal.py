"""Normal-distribution tolerance intervals (free core).

One-sided factors are exact, via the noncentral t distribution (Wald and
Wolfowitz, 1946). Two-sided factors use Howe's approximation (1969), the
Hahn-Meeker form Minitab reports. The exact two-sided Witkovsky engine
is paid-only and refuses loudly when requested.

All code below is original; the formulas are classical.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy import stats

from ._validate import as_1d_float, canon_method, canon_side, check_n, check_prob

_PAID_URL = "https://groundplane0.gumroad.com/l/zxhmnx"


class TolInterval(NamedTuple):
    """Result of a tolerance-interval computation."""

    lower: float    # interval lower bound (-inf when open below)
    upper: float    # interval upper bound (+inf when open above)
    conf: float     # achieved confidence level
    n: int          # sample size used
    P: float        # target population proportion covered
    alpha: float    # 1 - conf target
    side: str       # 'two-sided', 'upper' or 'lower'
    method: str     # 'exact', 'howe' or 'wilks'
    k: Optional[float]  # k-factor (normal method); None for nonparametric


def _k_onesided(n, P, alpha):
    """Exact one-sided factor via the noncentral t distribution."""
    import warnings

    delta = stats.norm.ppf(P) * np.sqrt(n)
    # scipy's nct.ppf emits a benign divide-by-zero RuntimeWarning from an
    # internal starting-value guess at tiny df; the returned value is
    # correct (checked against Lieberman's table, Monte Carlo coverage,
    # and direct CDF inversion), so it is suppressed here, narrowly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stats.nct.ppf(1.0 - alpha, n - 1, delta) / np.sqrt(n)


def _k_howe(n, P, alpha):
    """Howe (1969) two-sided approximation (the form Minitab uses)."""
    nu = n - 1
    z = stats.norm.ppf((1.0 + P) / 2.0)
    return z * np.sqrt(nu * (1.0 + 1.0 / n) / stats.chi2.ppf(alpha, nu))


def _refuse_exact_twosided():
    raise ValueError(
        "method='exact' two-sided (Witkovsky) is not in the free core. "
        "The full version ($30, one-time) adds the exact two-sided "
        "Witkovsky interval, two-sided nonparametric Wilks intervals, "
        f"and sample-size planning: {_PAID_URL}"
    )


def _refuse_wilks():
    raise ValueError(
        "method='wilks' (nonparametric Wilks intervals) is not in the "
        "free core. The full version ($30, one-time) adds the exact "
        "two-sided Witkovsky interval, two-sided nonparametric Wilks "
        f"intervals, and sample-size planning: {_PAID_URL}"
    )


def norm_k_factor(n, alpha=0.05, P=0.95, side="two-sided", method=None):
    """Tolerance factor k for a normal population (interval = xbar +/- k*s).

    Parameters
    ----------
    n : int  sample size (>= 2)
    alpha : float  1 - confidence level, strictly inside (0, 1)
    P : float  population proportion to cover, strictly inside (0, 1)
    side : 'two-sided' (default), 'upper' or 'lower'
    method : 'howe' or 'exact'. Defaults to 'howe' two-sided and
      'exact' one-sided. Two-sided 'exact' (Witkovsky) is paid-only
      and refuses loudly.

    One-sided factors are always exact (noncentral t). Two-sided 'howe'
    is Howe's closed-form approximation, the one Minitab reports.
    """
    n = check_n(n, 2)
    alpha = check_prob(alpha, "alpha")
    P = check_prob(P, "P")
    side = canon_side(side)
    if method is None:
        method = "howe" if side == "two-sided" else "exact"
    method = canon_method(method)
    if method == "wilks":
        _refuse_wilks()
    if side == "two-sided":
        if method == "exact":
            _refuse_exact_twosided()
        return _k_howe(n, P, alpha)
    if method != "exact":
        raise ValueError("method='howe' only applies to two-sided intervals")
    return _k_onesided(n, P, alpha)


def norm_tol_int(x, alpha=0.05, P=0.95, side="two-sided", method=None,
                 lognormal=False):
    """Tolerance interval for normal data (or lognormal via lognormal=True).

    Returns a TolInterval. For one-sided intervals the open side is
    reported as -inf / +inf. With lognormal=True the data must be
    positive; the interval is computed on log(x) and exponentiated back,
    which is the standard lognormal tolerance interval. method defaults
    to 'howe' two-sided and 'exact' one-sided; two-sided 'exact'
    (Witkovsky) is paid-only and refuses loudly.
    """
    arr = as_1d_float(x)
    n = arr.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    alpha = check_prob(alpha, "alpha")
    P = check_prob(P, "P")
    side = canon_side(side)
    if method is None:
        method = "howe" if side == "two-sided" else "exact"
    method = canon_method(method)

    if lognormal:
        if np.any(arr <= 0):
            raise ValueError("lognormal=True needs all x > 0")
        arr = np.log(arr)

    xbar = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    if s == 0.0 or not np.isfinite(s):
        raise ValueError(
            "sample variance is zero or overflows: a normal tolerance "
            "interval needs a finite s > 0 to estimate spread"
        )

    k = norm_k_factor(n, alpha=alpha, P=P, side=side, method=method)
    if side == "two-sided":
        lower, upper = xbar - k * s, xbar + k * s
    elif side == "upper":
        lower, upper = -np.inf, xbar + k * s
    else:
        lower, upper = xbar - k * s, np.inf
    if lognormal:
        lower, upper = float(np.exp(lower)), float(np.exp(upper))

    return TolInterval(
        lower=lower, upper=upper, conf=1.0 - alpha, n=n, P=P, alpha=alpha,
        side=side, method=method, k=k,
    )
