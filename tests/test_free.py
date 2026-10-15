"""Tests for the tolint free core.

Correctness is checked against the paid package (imported from its own
source tree, never copied). Premium APIs (np_tol_int, sample_size_tol,
method='exact' two-sided) must refuse loudly with the paid URL.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import tolint

HERE = Path(__file__).resolve().parent
PAID_SRC = HERE.parent.parent.parent / "tol-int" / "src" / "tolint"

_PAID_URL = "https://groundplane0.gumroad.com/l/zxhmnx"


def _load_paid():
    spec = importlib.util.spec_from_file_location(
        "paid_tolint",
        PAID_SRC / "__init__.py",
        submodule_search_locations=[str(PAID_SRC)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paid_tolint"] = mod
    spec.loader.exec_module(mod)
    return mod


paid = _load_paid()

rng = np.random.default_rng(11)
X = rng.normal(10.0, 2.0, size=40)


def test_norm_tol_int_matches_paid_howe():
    free = tolint.norm_tol_int(X, method="howe")
    ref = paid.norm_tol_int(X, method="howe")
    assert free.lower == pytest.approx(ref.lower, rel=1e-9)
    assert free.upper == pytest.approx(ref.upper, rel=1e-9)
    assert free.k == pytest.approx(ref.k, rel=1e-12)
    for side in ("upper", "lower"):
        free = tolint.norm_tol_int(X, side=side)
        ref = paid.norm_tol_int(X, side=side)
        assert free.lower == pytest.approx(ref.lower, rel=1e-9)
        assert free.upper == pytest.approx(ref.upper, rel=1e-9)


def test_norm_tol_int_matches_paid_onesided():
    for side in ("upper", "lower"):
        free = tolint.norm_tol_int(X, side=side, method="exact")
        ref = paid.norm_tol_int(X, side=side, method="exact")
        assert free.lower == pytest.approx(ref.lower, rel=1e-9)
        assert free.upper == pytest.approx(ref.upper, rel=1e-9)


def test_norm_k_factor_matches_paid():
    for side, method in (("two-sided", "howe"), ("upper", "exact"), ("lower", "exact")):
        free = tolint.norm_k_factor(40, side=side, method=method)
        ref = paid.norm_k_factor(40, side=side, method=method)
        assert free == pytest.approx(ref, rel=1e-12)


def test_lognormal_matches_paid():
    xl = np.exp(X)
    free = tolint.norm_tol_int(xl, lognormal=True)
    ref = paid.norm_tol_int(xl, lognormal=True, method="howe")
    assert free.lower == pytest.approx(ref.lower, rel=1e-9)
    assert free.upper == pytest.approx(ref.upper, rel=1e-9)


def _refuses_loudly(fn, *args, **kwargs):
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="groundplane0.gumroad.com/l/zxhmnx"):
        fn(*args, **kwargs)
    assert time.perf_counter() - t0 < 1.0


def test_exact_twosided_refuses():
    _refuses_loudly(tolint.norm_tol_int, X, method="exact")
    _refuses_loudly(tolint.norm_tol_int, X, side="two-sided", method="witkovsky")
    _refuses_loudly(tolint.norm_k_factor, 40, method="exact")


def test_stubs_refuse():
    _refuses_loudly(tolint.np_tol_int, X)
    _refuses_loudly(tolint.np_tol_int, X, side="upper", m=2)
    _refuses_loudly(tolint.sample_size_tol)
    _refuses_loudly(tolint.sample_size_tol, alpha=0.05, P=0.99)


def test_premium_symbols_absent():
    import tolint._normal as normal

    for name in ("_twosided_confidence", "_k_exact_twosided", "N_MAX_EXACT"):
        assert not hasattr(normal, name), name
    with pytest.raises(ImportError):
        import tolint._nonparam  # noqa


def test_garbage_inputs_raise():
    with pytest.raises((ValueError, TypeError)):
        tolint.norm_tol_int([1.0])
    with pytest.raises(ValueError):
        tolint.norm_tol_int(X, alpha=2.0)
    with pytest.raises(ValueError):
        tolint.norm_tol_int(X, side="sideways")
    with pytest.raises(ValueError):
        tolint.norm_tol_int(X, method="bogus")
    with pytest.raises(ValueError):
        tolint.norm_tol_int(np.zeros(10))
    with pytest.raises(ValueError):
        tolint.norm_k_factor(1)


def test_interval_sanity():
    iv = tolint.norm_tol_int(X, P=0.95)
    assert iv.lower < np.mean(X) < iv.upper
    assert iv.conf == pytest.approx(0.95)
    assert iv.method == "howe"
    up = tolint.norm_tol_int(X, side="upper")
    assert up.lower == -np.inf and np.isfinite(up.upper)
