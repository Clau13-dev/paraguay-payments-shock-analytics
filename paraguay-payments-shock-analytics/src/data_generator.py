"""
data_generator.py
-----------------
Genera una serie DIARIA SINTETICA (1-ene a 31-dic de 2026) de pagos, efectivo y ventas por
sector, alrededor de un shock de confianza y de liquidez en el sistema de pagos digital.

Los datos NO son oficiales. Son un "gemelo sintetico" con un mecanismo causal explicito y
REZAGOS CONOCIDOS, lo que permite validar si el analisis de correlacion cruzada recupera la
estructura que se sembro. Las magnitudes del shock se ANCLAN a cifras publicas reportadas
(prensa, nov-2026), usadas solo como referencia de orden de magnitud:
  - ~80% de las compras en multitiendas era digital antes del shock; despues ~70% en efectivo
  - caida de ventas de hasta ~70% en los dias mas criticos y ~30% a la semana siguiente
  - pago de garantia de depositos: ~41% de clientes a los 4-5 dias, ~93% al 12-nov (parcial)

Los dias posteriores al 15-nov-2026 son PROYECCION sintetica (`is_projection = 1`).

Mecanismo (verdad del generador)
  confianza_perdida(t)  = exp(-ln2 * dias_desde_shock / vida_media)          [persistente, vida media base 45 dias]
  friccion F(t)         = 0.25 * acceso_bloqueado(t) + 0.75 * confianza_perdida(t)
  participacion digital = base * (1 - k * F)                                  [por sector]
  demanda extra de ATM  = 0.9 * F^1.5  ->  escasez de efectivo (se repone en ~7 dias)
  dinero disponible     = 100 * (1 - 0.24 * acceso_bloqueado)                 [saldos congelados]
  caida de ventas       = b1*L1(k*F) + b2*L3(escasez) + b3*L7(confianza_perdida) + b4*L0(1-M/100)
con L_k = rezago de k dias.

Uso:  python -m src.data_generator
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

START, END = "2026-01-01", "2026-12-31"
OBS_END = "2026-11-15"   # ultimo dia "observado"; despues es proyeccion sintetica
T0 = "2026-11-03"        # fecha del shock

# Cobertura de pago de garantia (% de clientes afectados con algun reintegro), estilizada
PAYOUT_ANCHORS = {"2026-11-03": 0.0, "2026-11-05": 0.15, "2026-11-07": 0.41,
                  "2026-11-12": 0.93, "2026-11-20": 0.97, "2026-12-31": 0.99}

# rezagos verdaderos (dias) de cada canal sobre las ventas: el analisis debe intentar recuperarlos
TRUE_LAGS = {"digital": 1, "scarcity": 3, "trust": 7, "money": 0}

SECTORS = {
    #              etiqueta              base digital  k     (b1_digital, b2_escasez, b3_confianza, b4_dinero)   amp estacional
    "comercio":    dict(label="Comercio minorista", base=80.0, k=0.85, b=(0.22, 0.20, 0.10, 1.00), amp=1.00),
    "servicios":   dict(label="Servicios",          base=70.0, k=0.60, b=(0.15, 0.10, 0.08, 0.45), amp=0.45),
    "gastronomia": dict(label="Gastronomia",        base=55.0, k=0.70, b=(0.20, 0.14, 0.10, 0.60), amp=0.80),
}
DOW = np.array([0.92, 0.92, 0.95, 0.98, 1.08, 1.22, 0.93])  # lun..dom (comercio)


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x.copy()
    out = np.empty_like(x)
    out[:k] = x[0]
    out[k:] = x[:-k]
    return out


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x, eps = np.zeros(n), rng.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _season_raw(idx: pd.DatetimeIndex) -> np.ndarray:
    """Estacionalidad de calendario del comercio: dia de semana, inicio de mes, Black Friday, navidad, aguinaldo."""
    s = DOW[idx.dayofweek.values].copy()
    s *= np.where(idx.day.values <= 5, 1.08, 1.0)
    bf = {pd.Timestamp("2026-11-27"): 1.25, pd.Timestamp("2026-11-28"): 1.20, pd.Timestamp("2026-11-29"): 1.10}
    s *= np.array([bf.get(d, 1.0) for d in idx])
    dec = idx.month.values == 12
    ramp = np.where(dec & (idx.day.values <= 24), 1.0 + 0.35 * (idx.day.values - 1) / 23, 1.0)
    s *= ramp
    s *= np.where(dec & (idx.day.values == 25), 0.5, 1.0)
    s *= np.where(dec & (idx.day.values >= 26), 1.05, 1.0)
    s *= np.where(dec & (idx.day.values >= 15) & (idx.day.values <= 24), 1.06, 1.0)  # aguinaldo
    return s


def generate(seed: int = 7, trust_halflife_days: float = 45.0, payout_speed: float = 1.0,
             out_dir: Path | None = None) -> pd.DataFrame:
    """payout_speed > 1 acelera el reintegro de depositos; trust_halflife_days controla la persistencia."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="D")
    n = len(idx)
    t0 = int((pd.Timestamp(T0) - idx[0]).days)
    since = np.arange(n) - t0
    post = since >= 0

    # --- estados latentes ---
    pa = pd.Series(PAYOUT_ANCHORS, dtype=float)
    pa.index = pd.to_datetime(pa.index)
    payout = pa.reindex(pa.index.union(idx)).interpolate(method="time").reindex(idx).fillna(0.0).values
    if payout_speed != 1.0:  # reescala el tiempo desde el shock
        s = np.clip(since * payout_speed, 0, None)
        payout = np.interp(s, since[post], payout[post], left=0.0) * post
    blocked = np.where(post, 1 - payout, 0.0)
    trust_loss = np.where(post, np.exp(-np.log(2) * np.clip(since, 0, None) / trust_halflife_days), 0.0)
    F = 0.25 * blocked + 0.75 * trust_loss

    excess = 0.9 * F ** 1.5
    repl = np.where(post, 1 - np.exp(-np.clip(since, 0, None) / 7.0), 1.0)
    scarcity = 100 * np.clip(excess * (1 - repl), 0, 1)
    money = 100 * (1 - 0.24 * blocked)

    seas = _season_raw(idx)
    payday = np.where(idx.day.values <= 5, 1.0, 0.0)
    atm = 100 * (1 + excess) * (1 + 0.06 * payday) * np.exp(_ar1(n, 0.5, 0.02, rng))
    trust_idx = 100 * (1 - 0.35 * trust_loss) * np.exp(_ar1(n, 0.8, 0.006, rng))

    df = pd.DataFrame({
        "date": idx,
        "is_projection": (idx > pd.Timestamp(OBS_END)).astype(int),
        "blocked_access_pct": 100 * blocked,
        "guarantee_payout_pct": 100 * payout,
        "money_availability_idx": money * np.exp(_ar1(n, 0.5, 0.003, rng)),
        "atm_withdrawals_idx": atm,
        "cash_scarcity_idx": np.clip(scarcity + rng.normal(0, 0.6, n), 0, 100),
        "digital_trust_idx": trust_idx,
    })

    trend_adopt = np.clip(1 - 0.04 * (1 - np.arange(n) / max(t0, 1)), 0.9, 1.0)  # adopcion creciente antes del shock
    for key, sp in SECTORS.items():
        b1, b2, b3, b4 = sp["b"]
        base_t = sp["base"] * np.where(post, 1.0, trend_adopt)
        share = base_t * (1 - sp["k"] * F) + rng.normal(0, 0.5, n)
        drop = (b1 * _lag(sp["k"] * F, TRUE_LAGS["digital"])
                + b2 * _lag(scarcity / 100, TRUE_LAGS["scarcity"])
                + b3 * _lag(trust_loss, TRUE_LAGS["trust"])
                + b4 * _lag(1 - money / 100, TRUE_LAGS["money"]))
        drop = np.clip(drop, 0, 0.9)
        season_s = 1 + sp["amp"] * (seas - 1)
        noise = np.exp(_ar1(n, 0.6, 0.03, rng))
        sales = 100 * season_s * noise * (1 - drop)
        hist = season_s * np.exp(rng.normal(0, 0.02, n))  # indice estacional "historico" (con error de estimacion)
        df[f"digital_share_{key}"] = np.clip(share, 1, 99)
        df[f"sales_{key}"] = sales
        df[f"seasonal_hist_{key}"] = hist

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "payments_daily.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trust-halflife", type=float, default=45.0)
    a = ap.parse_args()
    d = generate(a.seed, a.trust_halflife, out_dir=RAW_DIR)
    print(f"payments_daily.csv: {len(d)} dias ({d['is_projection'].sum()} de proyeccion) -> data/raw/")
