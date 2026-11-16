"""
Paraguay Payments Shock Analytics
Del pago digital al efectivo: impacto de un shock de liquidez y confianza sobre las ventas.
streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import ccf as cc
from src import data_generator as dg
from src import features as ft

st.set_page_config(page_title="Payments Shock Analytics | Paraguay", page_icon="💳", layout="wide")
BLUE, RED, GREEN, GREY, AMBER = "#1f4e79", "#c0392b", "#2e8b57", "#7f8c8d", "#e69f00"
T0, OBS_END = pd.Timestamp(dg.T0), pd.Timestamp(dg.OBS_END)
MODES = {"Pre-blanqueada (recomendado)": "prewhite", "Primeras diferencias": "diff", "Niveles (puede ser espurio)": "level"}


@st.cache_data(show_spinner="Preparando datos...")
def get_data():
    if not (dg.RAW_DIR / "payments_daily.csv").exists():
        dg.generate(out_dir=dg.RAW_DIR)
    return ft.build_features(ft.load_raw())


@st.cache_data(show_spinner=False)
def simulate(halflife: float, payout_speed: float):
    return ft.build_features(dg.generate(trust_halflife_days=halflife, payout_speed=payout_speed))


def shade(fig, **kw):
    fig.add_vrect(x0=OBS_END, x1=pd.Timestamp(dg.END), fillcolor=GREY, opacity=0.12, line_width=0,
                  annotation_text="Proyeccion sintetica", annotation_position="top left", **kw)
    fig.add_vline(x=T0, line_dash="dash", line_color=RED)


f = get_data()
SECT = {k: v["label"] for k, v in dg.SECTORS.items()}

st.title("💳 Paraguay Payments Shock Analytics")
st.caption("Del pago digital al efectivo: como un shock de liquidez y confianza se transmite a las ventas · "
           "Datos SINTETICOS anclados a cifras publicas reportadas · Serie diaria ene-dic 2026")
st.sidebar.title("Parametros")
sector = st.sidebar.selectbox("Sector", list(SECT), format_func=SECT.get)
st.sidebar.markdown("---")
st.sidebar.caption("El analisis es de asociacion temporal; no demuestra causalidad. "
                   "Los dias posteriores al 15-nov son proyeccion sintetica.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen", "Linea de tiempo", "Correlacion cruzada", "Recuperacion y simulador", "Datos y metodologia"])
kp = ft.shock_kpis(f)

# ------------------------------------------------------------------ resumen --
with tab1:
    row = kp[kp["Sector"] == SECT[sector]].iloc[0]
    c = st.columns(5)
    c[0].metric("Pagos digitales: pre → minimo", f"{row['Participacion digital pre (%)']:.0f}% → {row['Participacion digital minima (%)']:.0f}%")
    c[1].metric("Caida maxima de ventas", f"{row['Caida maxima de ventas (%)']:.0f}%")
    c[2].metric("Caida promedio, semana 2", f"{row['Caida promedio sem. 2 (%)']:.0f}%")
    c[3].metric("Dias de ventas perdidos (a 15-nov)", f"{row['Dias de ventas perdidos (a 15-nov)']:.1f}")
    rec = row["Recuperacion a 95% del nivel"]
    c[4].metric("Recuperacion a 95% del nivel", str(rec) if rec is not None and rec == rec else "No antes del 31-dic")
    st.dataframe(kp.style.format({c: "{:.1f}" for c in kp.columns if c not in ("Sector", "Recuperacion a 95% del nivel")}),
                 hide_index=True, width="stretch")
    rt = ft.recovery_table(f, sector)
    liq = rt[rt["Variable"].isin(["Dinero disponible", "Escasez de efectivo"])]["Dias a recuperar 50%"].max()
    dig = rt[rt["Variable"].str.startswith("Pagos digitales")]["Dias a recuperar 50%"].iloc[0]
    st.info(f"**Lectura (en este gemelo sintetico).** La liquidez se normaliza rapido: el dinero disponible y la escasez de efectivo "
            f"recuperan la mitad de su desvio en ≈ {liq:.0f} dias, mientras los pagos digitales tardan ≈ {dig:.0f} dias. "
            f"Es decir, el dano persistente no viene de la falta de circulante sino de la **perdida de confianza en el pago digital**. "
            f"Este resultado refleja el mecanismo sembrado en el generador; con datos reales habria que contrastarlo.")

# ------------------------------------------------------------ linea de tiempo --
with tab2:
    start = st.slider("Desde", pd.Timestamp("2026-09-01").date(), pd.Timestamp(dg.END).date(), pd.Timestamp("2026-10-01").date())
    g = f[f["date"] >= pd.Timestamp(start)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Participacion de pagos digitales (%)", "Brecha de ventas vs. contrafactual (%)",
                                        "Liquidez: escasez de efectivo, retiros en cajeros y dinero disponible"),
                        specs=[[{}], [{}], [{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=g["date"], y=g[f"digital_share_{sector}"], name="Pagos digitales", line=dict(color=BLUE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=(np.exp(g[f"sales_gap_{sector}"]) - 1) * 100, name="Ventas vs. normal", line=dict(color=RED)), row=2, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["cash_scarcity_idx"], name="Escasez de efectivo", line=dict(color=AMBER)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=g["date"], y=g["atm_withdrawals_idx"], name="Retiros en cajeros (idx)", line=dict(color=GREY, dash="dot")), row=3, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=g["date"], y=g["money_availability_idx"], name="Dinero disponible (idx)", line=dict(color=GREEN, dash="dot")), row=3, col=1, secondary_y=True)
    shade(fig)
    fig.update_layout(height=760, hovermode="x unified", legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig, width="stretch")
    st.caption("Linea roja punteada: dia del shock. Zona gris: proyeccion sintetica. Brecha de ventas = ventas vs. nivel base × estacionalidad historica.")

# ------------------------------------------------------------------ CCF --
with tab3:
    a, b, c = st.columns(3)
    driver = a.selectbox("Conductor (X)", list(ft.DRIVERS), format_func=ft.DRIVERS.get)
    mode_label = b.selectbox("Tratamiento de la serie", list(MODES))
    window = c.radio("Ventana", ["Post-shock (15-oct a 31-dic)", "Pre-shock (5-ene a 28-oct)"], horizontal=True)
    start_w, end_w = ("2026-10-15", "2026-12-31") if window.startswith("Post") else ("2026-01-05", "2026-10-28")
    max_lag = st.slider("Rezago maximo (dias)", 5, 21, 14)
    mode = MODES[mode_label]

    x = ft.driver_series(f, driver, sector)
    y = f.set_index("date")[f"sales_gap_{sector}"]
    res = cc.analyze(x, y, start_w, end_w, mode, max_lag)
    r = res["ccf"]
    fig = go.Figure(go.Bar(x=r.index, y=r.values, marker_color=[RED if abs(v) > res["band"] else GREY for v in r.values]))
    fig.add_hline(y=res["band"], line_dash="dash", line_color=BLUE)
    fig.add_hline(y=-res["band"], line_dash="dash", line_color=BLUE)
    fig.update_layout(height=380, title=f"CCF: {ft.DRIVERS[driver]} → Ventas ({SECT[sector]}) · n = {res['n']}",
                      xaxis_title="Rezago k (dias): k > 0 = el conductor precede a las ventas", yaxis_title="Correlacion r(k)")
    st.plotly_chart(fig, width="stretch")
    k = st.columns(3)
    k[0].metric("Rezago del maximo", f"{res['peak_lag']} dias")
    k[1].metric("r en el maximo", f"{res['peak_r']:.2f}")
    k[2].metric("Significativo al 95%", "Si" if res["peak_significant"] else "No")

    if len(res["x"]) > 30:
        boot = cc.bootstrap_peak_lag(res["x"], res["y"], max_lag, n_boot=200)
        d = boot.value_counts(normalize=True).sort_index()
        figb = go.Figure(go.Bar(x=d.index, y=d.values * 100, marker_color=BLUE))
        figb.update_layout(height=260, title="Estabilidad del rezago del maximo (bootstrap de bloques)", xaxis_title="Rezago (dias)", yaxis_title="% de remuestras")
        st.plotly_chart(figb, width="stretch")

    st.subheader("Los conductores se mueven juntos: por eso la CCF no separa canales")
    drv = pd.DataFrame({ft.DRIVERS[d]: ft.driver_series(f, d, sector) for d in ft.DRIVERS}).loc["2026-10-15":"2026-12-31"].corr()
    hm = go.Figure(go.Heatmap(z=drv.values, x=drv.columns, y=drv.index, zmin=-1, zmax=1, colorscale="RdBu", text=np.round(drv.values, 2), texttemplate="%{text}"))
    hm.update_layout(height=380)
    st.plotly_chart(hm, width="stretch")
    st.warning("**Limite metodologico.** Con un unico shock comun, todos los conductores saltan el mismo dia y la CCF detecta "
               "el *momento* de transmision (≈ 1 dia) pero **no puede atribuir el efecto a un canal especifico**. "
               "Para separar canales se necesita variacion independiente (varios episodios, diferencias entre sectores o regiones, o microdatos).")

# ------------------------------------------------- recuperacion y simulador --
with tab4:
    rt = ft.recovery_table(f, sector)
    figr = go.Figure(go.Bar(x=rt["Dias a recuperar 50%"], y=rt["Variable"], orientation="h", marker_color=BLUE))
    figr.update_layout(height=320, title="Dias hasta recuperar 50% de la desviacion maxima", yaxis=dict(autorange="reversed"))
    st.plotly_chart(figr, width="stretch")
    st.dataframe(rt.style.format({"Desviacion maxima": "{:.2f}"}, na_rep="no recupera"), hide_index=True, width="stretch")

    st.subheader("Simulador de recuperacion (escenarios sobre el gemelo sintetico)")
    a, b = st.columns(2)
    hl = a.slider("Persistencia de la desconfianza: vida media (dias)", 10, 120, 45, 5)
    ps = b.slider("Velocidad de reintegro de depositos (x)", 0.5, 3.0, 1.0, 0.25)
    sim = simulate(float(hl), float(ps))
    base = f
    fig = go.Figure()
    for name, d, colr in [("Escenario base (45 dias)", base, GREY), (f"Tu escenario ({hl} dias, {ps}x)", sim, RED)]:
        dd = d[d["date"] >= pd.Timestamp("2026-10-20")]
        fig.add_trace(go.Scatter(x=dd["date"], y=(np.exp(dd[f"sales_gap_{sector}"]) - 1) * 100, name=name, line=dict(color=colr)))
    shade(fig)
    fig.update_layout(height=360, title=f"Ventas vs. normal (%) · {SECT[sector]}", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    col_lost = "Dias de ventas perdidos (a 31-dic, proy.)"
    lb = kp[kp["Sector"] == SECT[sector]].iloc[0][col_lost]
    ksim = ft.shock_kpis(sim)
    ls = ksim[ksim["Sector"] == SECT[sector]].iloc[0][col_lost]
    k = st.columns(3)
    k[0].metric("Dias de ventas perdidos, base", f"{lb:.1f}")
    k[1].metric("Dias de ventas perdidos, tu escenario", f"{ls:.1f}", f"{ls - lb:+.1f}", delta_color="inverse")
    k[2].metric("Efecto de restaurar la confianza mas rapido", f"{(lb - ls) / lb * 100:+.0f}%" if lb else "n/a")
    st.caption("Escenario hipotetico sobre el mecanismo del generador: sirve para dimensionar sensibilidades, no para pronosticar.")

# ----------------------------------------------------------- datos / metodo --
with tab5:
    st.markdown(f"""
**Datos.** Serie diaria sintetica (1-ene a 31-dic de 2026; proyeccion desde el {OBS_END:%d-%m}), con un mecanismo causal explicito y
rezagos conocidos. Magnitudes ancladas a cifras publicas reportadas: ~80% de compras digitales en multitiendas antes del shock, ~70% en
efectivo despues, caidas de ventas de hasta ~70% en los dias criticos y ~30% a la semana siguiente.

**Metodo.** Brecha de ventas = ln(ventas / (nivel base × estacionalidad historica)); CCF con diferenciacion y pre-blanqueo (Box-Jenkins),
banda ±1,96/√n y bootstrap de bloques; comparacion pre vs. post-shock; velocidad de recuperacion por variable.

**Rezagos sembrados en el generador (verdad):** digital {dg.TRUE_LAGS['digital']} d · escasez {dg.TRUE_LAGS['scarcity']} d · confianza {dg.TRUE_LAGS['trust']} d · dinero {dg.TRUE_LAGS['money']} d.
La CCF recupera el momento comun de transmision pero **no** los rezagos por canal (ver pestana de correlacion cruzada).
""")
    st.dataframe(f.tail(10).style.format(precision=2), width="stretch")
