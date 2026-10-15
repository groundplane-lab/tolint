"""Input validation shared by the tolint routines.

Every public function routes its arguments through here first, so bad
input fails loudly with a plain-English message instead of producing a
number that looks fine but is wrong.
"""

import numpy as np

_SIDE_ALIASES = {
    "two-sided": "two-sided",
    "two_sided": "two-sided",
    "twosided": "two-sided",
    "two": "two-sided",
    "2": "two-sided",
    "both": "two-sided",
    "upper": "upper",
    "one-sided-upper": "upper",
    "one_sided_upper": "upper",
    "onesidedupper": "upper",
    "one-sided upper": "upper",
    "up": "upper",
    "1": "upper",
    "lower": "lower",
    "one-sided-lower": "lower",
    "one_sided_lower": "lower",
    "onesidedlower": "lower",
    "one-sided lower": "lower",
    "low": "lower",
    "-1": "lower",
}

_METHOD_ALIASES = {
    "exact": "exact",
    "witkovsky": "exact",
    "howe": "howe",
    "he": "howe",
    "wilks": "wilks",
}


def as_1d_float(x, name="x"):
    """Coerce array-like to a 1-D float array, rejecting junk loudly."""
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric array-like") from None
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return arr


def check_prob(value, name):
    """Validate a probability strictly inside (0, 1)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number strictly between 0 and 1") from None
    if not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return v


def _as_int_at_least(value, minimum, what):
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be an integer >= {minimum}, got {value!r}") from None
    if not f.is_integer() or f < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return int(f)


def check_n(value, minimum, name="n"):
    """Validate a sample size as an integer >= minimum."""
    return _as_int_at_least(value, minimum, name)


def check_m(value):
    """Validate the order-statistic trim count m (>= 1 integer)."""
    return _as_int_at_least(value, 1, "m")


def canon_side(side):
    """Canonicalize the side argument, rejecting anything ambiguous."""
    if not isinstance(side, str):
        raise ValueError(
            f"side must be 'two-sided', 'upper' or 'lower', got {side!r}"
        )
    key = side.strip().lower()
    if key not in _SIDE_ALIASES:
        raise ValueError(
            f"side must be 'two-sided', 'upper' or 'lower', got {side!r}"
        )
    return _SIDE_ALIASES[key]


def canon_method(method):
    """Canonicalize the method argument ('exact', 'howe', or 'wilks')."""
    if not isinstance(method, str):
        raise ValueError(f"method must be 'exact' or 'howe', got {method!r}")
    key = method.strip().lower()
    if key not in _METHOD_ALIASES:
        raise ValueError(f"method must be 'exact' or 'howe', got {method!r}")
    return _METHOD_ALIASES[key]
