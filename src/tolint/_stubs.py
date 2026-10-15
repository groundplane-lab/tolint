"""Paid-only API stubs for the tolint free core.

np_tol_int and sample_size_tol exist here with their real signatures so
code written against the full package imports cleanly, but they refuse
loudly with the paid URL instead of computing. Their implementations
(the Wilks order-statistic engine and the sample-size search) ship only
in the paid product.
"""

_PAID_URL = "https://groundplane0.gumroad.com/l/zxhmnx"

_PAID_ADDS = (
    "the exact two-sided Witkovsky interval, two-sided nonparametric "
    "Wilks intervals, and sample-size planning"
)


def np_tol_int(x, alpha=0.05, P=0.95, side="two-sided", m=1):
    """Distribution-free tolerance interval (paid-only stub)."""
    raise ValueError(
        "np_tol_int is not in the free core. The full version ($30, "
        f"one-time) adds {_PAID_ADDS}: {_PAID_URL}"
    )


def sample_size_tol(alpha=0.05, P=0.95, side="two-sided", m=1):
    """Nonparametric sample-size planning (paid-only stub)."""
    raise ValueError(
        "sample_size_tol is not in the free core. The full version ($30, "
        f"one-time) adds {_PAID_ADDS}: {_PAID_URL}"
    )
