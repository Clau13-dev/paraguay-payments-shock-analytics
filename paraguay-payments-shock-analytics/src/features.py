"""
features.py
-----------
Capa de preparacion analitica: validacion, contrafactual de ventas y brechas.

  brecha_ventas(t)   = ln( ventas(t) / (nivel_base * estacional_historico(t)) )
  brecha_digital(t)  = participacion_digital(t) - media pre-shock
  dias_de_ventas_perdidos = suma de (1 - exp(brecha)) desde el shock

`nivel_base` = promedio pre-shock (1-ene a 28-oct) de ventas / estacional historico.
El estacional historico se toma como dato exogeno (en datos reales: promedio de los mismos
dias de anios previos), lo que evita confundir estacionalidad con efecto del shock.

Uso:  python -m src.features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_generator import OBS_END, RAW_DIR, SECTORS, T0

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRE_END = "2026-10-28"      # fin de la ventana pre-shock usada como linea base
DRIVERS = {
    "digital_gap": "Brecha de pagos digitales (pp)",
    "cash_scarcity_idx": "Escasez de efectivo (indice)",
    "atm_gap": "Retiros en cajeros (vs. normal)",
    "money_gap": "Dinero disponible (vs. normal)",
    "trust_gap": "Confianza en pagos digitales",
}

REQUIRED = ["date", "is_projection", "blocked_access_pct", "guarantee_payout_pct", "money_availability_idx",
            "atm_withdrawals_idx", "cash_scarcity_idx", "digital_trust_idx"] + \
    [f"{p}_{s}" for s in SECTORS for p in ("digital_share", "sales", "seasonal_hist")]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "payments_daily.csv"
    if not Path(path).exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta `python -m src.data_generator`.")
    df = pd.read_csv(path, parse_dates=["date"])
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    if df["date"].duplicated().any():
        raise ValueError("Fechas duplicadas")
    if not (df["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("La serie debe ser diaria y continua")
    for s in SECTORS:
        if (df[f"sales_{s}"] <= 0).any():
            raise ValueError(f"Ventas no positivas en {s}")
        if not df[f"digital_share_{s}"].between(0, 100).all():
            raise ValueError(f"Participacion digital fuera de [0, 100] en {s}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy().set_index("date")
    pre = f.index <= pd.Timestamp(PRE_END)
    for s in SECTORS:
        level = (f[f"sales_{s}"] / f[f"seasonal_hist_{s}"])[pre].mean()
        f[f"sales_gap_{s}"] = np.log(f[f"sales_{s}"] / (level * f[f"seasonal_hist_{s}"]))
        f[f"digital_gap_{s}"] = f[f"digital_share_{s}"] - f[f"digital_share_{s}"][pre].iloc[-60:].mean()
    f["atm_gap"] = np.log(f["atm_withdrawals_idx"] / f["atm_withdrawals_idx"][pre].mean())
    f["money_gap"] = np.log(f["money_availability_idx"] / f["money_availability_idx"][pre].mean())
    f["trust_gap"] = np.log(f["digital_trust_idx"] / f["digital_trust_idx"][pre].mean())
    return f.reset_index()


def driver_series(f: pd.DataFrame, driver: str, sector: str) -> pd.Series:
    """Serie del conductor; la brecha digital es especifica del sector."""
    col = f"digital_gap_{sector}" if driver == "digital_gap" else driver
    return f.set_index("date")[col]


def shock_kpis(f: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.Timestamp(T0)
    rows = []
    g = f.set_index("date")
    for s, sp in SECTORS.items():
        gap = g[f"sales_gap_{s}"]
        post = gap[gap.index >= t0]
        share = g[f"digital_share_{s}"]
        lost = (1 - np.exp(post[post.index <= pd.Timestamp(OBS_END)])).sum()
        lost_dec = (1 - np.exp(post)).sum()
        rec = post[(post > np.log(0.95)) & (post.index > post.idxmin())]
        rows.append({
            "Sector": sp["label"],
            "Participacion digital pre (%)": share[: pd.Timestamp(PRE_END)].iloc[-60:].mean(),
            "Participacion digital minima (%)": share[share.index >= t0].min(),
            "Caida maxima de ventas (%)": (1 - np.exp(post.min())) * 100,
            "Caida promedio sem. 2 (%)": (1 - np.exp(post[t0 + pd.Timedelta(days=6): t0 + pd.Timedelta(days=12)].mean())) * 100,
            "Dias de ventas perdidos (a 15-nov)": lost,
            "Dias de ventas perdidos (a 31-dic, proy.)": lost_dec,
            "Recuperacion a 95% del nivel": rec.index.min().date() if len(rec) else None,
        })
    return pd.DataFrame(rows)


def recovery_table(f: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Dias desde el shock hasta recuperar 50% y 90% de la desviacion maxima, por variable.

    Descriptivo y libre de modelo: contrasta la velocidad de normalizacion de la liquidez
    (dinero disponible, escasez de efectivo) con la de los pagos digitales y las ventas.
    """
    g = f.set_index("date")
    t0 = pd.Timestamp(T0)
    series = {
        "Escasez de efectivo": g["cash_scarcity_idx"] - g["cash_scarcity_idx"][:PRE_END].mean(),
        "Dinero disponible": g["money_gap"],
        "Retiros en cajeros": g["atm_gap"],
        f"Pagos digitales ({SECTORS[sector]['label']})": g[f"digital_gap_{sector}"],
        "Confianza en pagos digitales": g["trust_gap"],
        f"Ventas ({SECTORS[sector]['label']})": g[f"sales_gap_{sector}"],
    }
    rows = []
    for name, x in series.items():
        post = x[x.index >= t0]
        peak_day = post.abs().idxmax()
        peak = post[peak_day]
        after = post[post.index >= peak_day].abs()
        def first(th):
            hit = after[after <= th * abs(peak)]
            return int((hit.index.min() - t0).days) if len(hit) else None
        rows.append({"Variable": name, "Desviacion maxima": peak, "Dia del maximo": int((peak_day - t0).days),
                     "Dias a recuperar 50%": first(0.5), "Dias a recuperar 90%": first(0.1)})
    return pd.DataFrame(rows)


def run(raw: Path | None = None, proc_dir: Path = PROC_DIR) -> pd.DataFrame:
    f = build_features(load_raw(raw))
    proc_dir.mkdir(parents=True, exist_ok=True)
    f.to_parquet(proc_dir / "features_daily.parquet", index=False)
    shock_kpis(f).to_csv(proc_dir / "shock_kpis.csv", index=False)
    print(f"Features OK: {len(f)} dias -> {proc_dir}")
    return f


if __name__ == "__main__":
    run()
    print(shock_kpis(pd.read_parquet(PROC_DIR / "features_daily.parquet")).round(1).to_string(index=False))
