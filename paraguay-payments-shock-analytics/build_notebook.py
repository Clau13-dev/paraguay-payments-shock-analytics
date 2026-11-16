"""Construye y ejecuta notebooks/01_explicacion.ipynb (python build_notebook.py)."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

nb.cells = [
md("""# Del pago digital al efectivo: ¿cómo se transmitió el shock a las ventas?

**Pregunta:** ¿cuánto de la caída de ventas se asocia a la falta de efectivo y al retroceso de los pagos digitales, y con qué rezago?

> **Aviso.** Los datos son **sintéticos**: un gemelo con mecanismo causal explícito y magnitudes ancladas a cifras públicas reportadas (≈ 80% de compras digitales antes del shock, ≈ 70% en efectivo después, caídas de ventas de hasta ≈ 70% en los días críticos y ≈ 30% a la semana siguiente). Los días posteriores al 15-nov-2026 son proyección sintética. Todo lo que aparece aquí es **asociación temporal**, no causalidad."""),
code("""import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src import data_generator as dg, features as ft, ccf

plt.rcParams.update({"figure.figsize": (10, 3.6), "axes.grid": True, "grid.alpha": .25})
raw = dg.generate(seed=7)
f = ft.build_features(raw)
T0 = pd.Timestamp(dg.T0); OBS = pd.Timestamp(dg.OBS_END)
raw[["date","is_projection","digital_share_comercio","sales_comercio","cash_scarcity_idx"]].tail(3)"""),
md("""## 1. Construir el contrafactual de ventas
Comparar ventas de noviembre con las de octubre confunde el efecto del shock con la estacionalidad (Black Friday, Navidad, aguinaldo).
Por eso se define una **brecha**: `ln(ventas / (nivel base × estacionalidad histórica))`, con el nivel base estimado solo con datos **previos al shock**."""),
code("""g = f.set_index("date")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for s in dg.SECTORS:
    ax[0].plot(g.index[g.index >= "2026-10-01"], g.loc["2026-10-01":, f"digital_share_{s}"], label=dg.SECTORS[s]["label"])
    ax[1].plot(g.index[g.index >= "2026-10-01"], (np.exp(g.loc["2026-10-01":, f"sales_gap_{s}"]) - 1) * 100, label=dg.SECTORS[s]["label"])
for a, t in zip(ax, ["Participación de pagos digitales (%)", "Ventas vs. normal (%)"]):
    a.axvline(T0, color="crimson", ls="--"); a.axvspan(OBS, g.index[-1], color="grey", alpha=.12); a.set_title(t); a.legend()
plt.tight_layout(); plt.show()
ft.shock_kpis(f).round(1)"""),
md("""**Lectura.** El comercio minorista —el más digitalizado— cae más; servicios, menos. La zona gris es proyección sintética."""),
md("""## 2. Correlación cruzada con rezagos
Convención: `k > 0` significa que el conductor **precede** a las ventas por `k` días. Para evitar correlaciones espurias entre series persistentes se usan **primeras diferencias** y **pre-blanqueo (Box-Jenkins)**; la banda es ±1,96/√n.

Primero, una verificación de la implementación con una serie de juguete cuyo rezago conocemos (3 días):"""),
code("""rng = np.random.default_rng(0)
x = rng.normal(size=400); y = np.roll(x, 3) + 0.3 * rng.normal(size=400)
r = ccf.ccf(x[10:], y[10:], 10)
print("Rezago recuperado:", int(r.idxmax()), "| r =", round(r.max(), 2))"""),
code("""y = g["sales_gap_comercio"]
x = ft.driver_series(f, "digital_gap", "comercio")
post = ccf.analyze(x, y, "2026-10-15", "2026-12-31", "prewhite")
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.bar(post["ccf"].index, post["ccf"].values, color=["crimson" if abs(v) > post["band"] else "grey" for v in post["ccf"].values])
ax.axhline(post["band"], ls="--"); ax.axhline(-post["band"], ls="--")
ax.set_title("CCF pre-blanqueada: brecha de pagos digitales → ventas (comercio minorista), ventana post-shock")
ax.set_xlabel("rezago k (días)"); plt.show()
print({k: round(v, 2) if isinstance(v, float) else v for k, v in post.items() if k in ("peak_lag","peak_r","peak_significant","n")})"""),
md("""### Contraste pre-shock vs. post-shock
Si la relación es específica del evento, antes del shock la correlación debería ser mucho más débil."""),
code("""pre = ccf.analyze(x, y, "2026-01-05", "2026-10-28", "prewhite")
print(f"Pre-shock : |r| máx = {abs(pre['peak_r']):.2f} (n={pre['n']})")
print(f"Post-shock: |r| máx = {abs(post['peak_r']):.2f} (n={post['n']})")"""),
md("""## 3. Lo que la CCF **no** puede hacer: separar canales
Todos los conductores (pagos digitales, escasez de efectivo, retiros, dinero disponible, confianza) saltan **el mismo día**. Al medirlos contra las ventas, todos dan su máximo en el mismo rezago, aunque el generador siembra rezagos distintos por canal."""),
code("""tab = ccf.summary_table(f, ft.DRIVERS, dg.SECTORS, "2026-10-15", "2026-12-31", "prewhite", ft.driver_series)
print("Rezagos sembrados en el generador:", dg.TRUE_LAGS)
tab[tab.sector_key == "comercio"][["Conductor", "Rezago del maximo (dias)", "r en el maximo", "Significativo (95%)"]].round(2)"""),
code("""d = pd.DataFrame({ft.DRIVERS[k]: ft.driver_series(f, k, "comercio") for k in ft.DRIVERS}).loc["2026-10-15":].corr().round(2)
d"""),
md("""**Conclusión metodológica.** La CCF detecta bien el **momento** de transmisión (≈ 1 día después del shock) y confirma que la relación aparece solo con el evento. Pero con un único shock común y conductores casi colineales (|r| > 0,9) **no se puede atribuir el efecto a un canal**. Para hacerlo se necesita variación independiente: varios episodios, diferencias entre sectores o regiones, o microdatos. Reportar un "rezago por canal" con estos datos sería engañoso, y verificarlo contra la verdad sembrada es justamente la ventaja de trabajar con un gemelo sintético."""),
md("""## 4. Velocidad de recuperación: liquidez vs. confianza
Un análisis descriptivo, libre de modelo, sí es informativo: ¿cuántos días tarda cada variable en recuperar la mitad de su desvío máximo?"""),
code("""rt = ft.recovery_table(f, "comercio")
ax = rt.set_index("Variable")["Dias a recuperar 50%"].plot.barh(figsize=(9, 3), color="#1f4e79"); ax.invert_yaxis()
ax.set_title("Días hasta recuperar 50% de la desviación máxima (comercio minorista)"); plt.show()
rt.round(2)"""),
md("""**Lectura (en este gemelo).** La liquidez (dinero disponible, escasez de efectivo) se normaliza en pocos días, mientras que los pagos digitales y la confianza tardan semanas. El daño persistente proviene de la **pérdida de confianza**, no de la falta de circulante. Esto es consecuencia del mecanismo sembrado; con datos reales (p. ej. transacciones de redes de pago, retiros en cajeros, ventas de comercios) habría que contrastarlo.

## 5. Escenario: ¿y si la confianza se restablece más rápido?"""),
code("""base = ft.shock_kpis(f).set_index("Sector")["Dias de ventas perdidos (a 31-dic, proy.)"]
fast = ft.shock_kpis(ft.build_features(dg.generate(seed=7, trust_halflife_days=20))).set_index("Sector")["Dias de ventas perdidos (a 31-dic, proy.)"]
pd.DataFrame({"Base (vida media 45 d)": base, "Confianza rápida (20 d)": fast, "Ahorro (%)": ((base - fast) / base * 100)}).round(1)"""),
md("""## 6. Qué se puede y qué no se puede concluir
- ✅ La relación temporal entre el retroceso de los pagos digitales y la caída de ventas **aparece con el evento** y se transmite en ≈ 1 día.
- ✅ La normalización de la liquidez es más rápida que la de los pagos digitales y las ventas.
- ❌ No se puede atribuir la caída a un canal específico con un solo shock.
- ❌ Las cifras son del gemelo sintético: ordenan la discusión, no sustituyen datos reales.

**Siguiente paso con datos reales:** reemplazar `data_generator.py` por la carga de series de transacciones, retiros y ventas, y añadir variación independiente (sectores, regiones, otros episodios) para identificar canales."""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
NotebookClient(nb, timeout=180, resources={"metadata": {"path": "notebooks"}}).execute()
nbf.write(nb, "notebooks/01_explicacion.ipynb")
print("notebook ejecutado y guardado")
