"""
ccf.py
------
Correlacion cruzada con rezagos entre conductores del shock (pagos digitales, escasez de
efectivo, dinero disponible, confianza) y las ventas sectoriales.

Convencion: lag k > 0  =>  el conductor X PRECEDE a las ventas Y por k dias
            r(k) = corr( X[t], Y[t+k] )

Buenas practicas incluidas (evitan correlaciones espurias entre series con tendencia/persistencia):
  1. Diferenciacion de ambas series (cambios diarios), y
  2. Pre-blanqueo (Box-Jenkins): se ajusta un AR(p) a X y se filtran X e Y con ese mismo filtro,
  3. Banda de significancia ±1,96/sqrt(n) y bootstrap de bloques para el rezago del maximo.

IMPORTANTE: es evidencia de ASOCIACION temporal y de orden de precedencia, no de causalidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_LAG = 14


def _ar_fit(x: np.ndarray, p: int) -> np.ndarray:
    """Coeficientes AR(p) por minimos cuadrados (x ya sin media)."""
    X = np.column_stack([x[p - j - 1: len(x) - j - 1] for j in range(p)])
    y = x[p:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _ar_filter(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    p = len(coef)
    pred = sum(coef[j] * x[p - j - 1: len(x) - j - 1] for j in range(p))
    return x[p:] - pred


def prewhiten(x: np.ndarray, y: np.ndarray, p: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    y = y - y.mean()
    coef = _ar_fit(x, p)
    return _ar_filter(x, coef), _ar_filter(y, coef)


def ccf(x: np.ndarray, y: np.ndarray, max_lag: int = MAX_LAG) -> pd.Series:
    """r(k) = corr(x[t], y[t+k]) para k en [-max_lag, max_lag]."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = x[: len(x) - k], y[k:]
        else:
            a, b = x[-k:], y[: len(y) + k]
        out[k] = np.corrcoef(a, b)[0, 1] if len(a) > 3 and a.std() > 0 and b.std() > 0 else np.nan
    return pd.Series(out, name="r")


def band(n: int) -> float:
    return 1.96 / np.sqrt(n)


def _prep(x: pd.Series, y: pd.Series, mode: str) -> tuple[np.ndarray, np.ndarray]:
    xa, ya = x.values.astype(float), y.values.astype(float)
    if mode in ("diff", "prewhite"):
        xa, ya = np.diff(xa), np.diff(ya)
    if mode == "prewhite":
        xa, ya = prewhiten(xa, ya)
    return xa, ya


def analyze(x: pd.Series, y: pd.Series, start: str, end: str, mode: str = "prewhite",
            max_lag: int = MAX_LAG) -> dict:
    """CCF en la ventana [start, end]. modes: 'level' | 'diff' | 'prewhite'."""
    xs, ys = x.loc[start:end], y.loc[start:end]
    xa, ya = _prep(xs, ys, mode)
    r = ccf(xa, ya, max_lag)
    lag = int(r.abs().idxmax())
    return {"ccf": r, "n": len(xa), "band": band(len(xa)), "peak_lag": lag, "peak_r": float(r[lag]),
            "peak_significant": bool(abs(r[lag]) > band(len(xa))), "x": xa, "y": ya}


def bootstrap_peak_lag(xa: np.ndarray, ya: np.ndarray, max_lag: int = MAX_LAG, block: int = 10,
                       n_boot: int = 300, seed: int = 1) -> pd.Series:
    """Bootstrap de bloques moviles de pares (x,y): distribucion del rezago del maximo |r|."""
    rng = np.random.default_rng(seed)
    n = len(xa)
    nblocks = int(np.ceil(n / block))
    peaks = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nblocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r = ccf(xa[idx], ya[idx], max_lag)
        peaks.append(int(r.abs().idxmax()))
    return pd.Series(peaks, name="peak_lag")


def summary_table(f: pd.DataFrame, drivers: dict[str, str], sectors: dict[str, dict], start: str, end: str,
                  mode: str = "prewhite", driver_series=None) -> pd.DataFrame:
    rows = []
    for s, sp in sectors.items():
        y = f.set_index("date")[f"sales_gap_{s}"]
        for d, label in drivers.items():
            x = driver_series(f, d, s)
            a = analyze(x, y, start, end, mode)
            rows.append({"Sector": sp["label"], "Conductor": label, "driver": d, "sector_key": s,
                         "Rezago del maximo (dias)": a["peak_lag"], "r en el maximo": a["peak_r"],
                         "Significativo (95%)": a["peak_significant"], "n": a["n"]})
    return pd.DataFrame(rows)
