"""Tests: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import ccf, data_generator as dg, features as ft


@pytest.fixture(scope="module")
def feats():
    return ft.build_features(dg.generate(seed=3))


def test_ccf_recovers_known_lag_on_toy_series():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = np.roll(x, 3) + 0.3 * rng.normal(size=400)      # y[t] = x[t-3] -> x precede a y por 3
    r = ccf.ccf(x[10:], y[10:], 10)
    assert r.idxmax() == 3 and r.max() > 0.8


def test_prewhitening_reduces_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=600)
    x = np.zeros(600)
    for t in range(1, 600):
        x[t] = 0.9 * x[t - 1] + e[t]
    xw, _ = ccf.prewhiten(x, x.copy(), p=3)
    ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    assert abs(ac(xw)) < 0.2 < ac(x)


def test_generator_shape_and_flags():
    d = dg.generate(seed=1)
    assert len(d) == 365 and d["is_projection"].sum() == 46
    ft.validate(d)


def test_shock_anchors_are_in_reported_ballpark(feats):
    k = ft.shock_kpis(feats).set_index("Sector")
    r = k.loc["Comercio minorista"]
    assert 70 <= r["Participacion digital pre (%)"] <= 90
    assert 30 <= r["Caida maxima de ventas (%)"] <= 75
    assert 20 <= r["Caida promedio sem. 2 (%)"] <= 40


def test_sector_heterogeneity_retail_hit_hardest(feats):
    k = ft.shock_kpis(feats).set_index("Sector")["Caida maxima de ventas (%)"]
    assert k["Comercio minorista"] > k["Gastronomia"] > k["Servicios"]


def test_no_effect_before_shock(feats):
    pre = feats.set_index("date").loc["2026-01-05":"2026-10-28", "sales_gap_comercio"]
    assert abs(pre.mean()) < 0.02


def test_post_shock_ccf_has_stronger_link_than_pre(feats):
    y = feats.set_index("date")["sales_gap_comercio"]
    x = ft.driver_series(feats, "digital_gap", "comercio")
    post = ccf.analyze(x, y, "2026-10-15", "2026-12-31", "prewhite")
    pre = ccf.analyze(x, y, "2026-01-05", "2026-10-28", "prewhite")
    assert post["peak_lag"] >= 0 and abs(post["peak_r"]) > 3 * abs(pre["peak_r"])


def test_faster_trust_recovery_loses_fewer_sales():
    slow = ft.shock_kpis(ft.build_features(dg.generate(seed=3, trust_halflife_days=90)))
    fast = ft.shock_kpis(ft.build_features(dg.generate(seed=3, trust_halflife_days=20)))
    col = "Dias de ventas perdidos (a 31-dic, proy.)"
    assert (fast[col] < slow[col]).all()
