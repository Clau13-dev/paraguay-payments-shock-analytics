# 💳 Paraguay Payments Shock Analytics

**Del pago digital al efectivo:** cómo un shock de liquidez y confianza en el sistema de pagos se transmite a las ventas del comercio, y qué tan bien puede la **correlación cruzada con rezagos** describir esa transmisión.

> **Sobre los datos.** Todo es **sintético por diseño**: un gemelo con un mecanismo causal explícito y *rezagos conocidos*, lo que permite validar el método contra una verdad sembrada. Las magnitudes del shock están ancladas a cifras públicas reportadas por la prensa (nov-2026), usadas solo como orden de magnitud. Los días posteriores al 15-nov-2026 son **proyección sintética** (`is_projection = 1`). Ninguna cifra es oficial.

Proyecto complementario a [`paraguay-fintech-treasury-analytics`](../paraguay-fintech-treasury-analytics): aquel mira la tesorería y el riesgo de una fintech; este mira la **transmisión del shock a la economía real**.

---

## Pregunta de negocio

Cuando se interrumpen transferencias, tarjetas y QR, y parte de los saldos queda inmovilizada, el comercio vuelve al efectivo:

> **¿Cuánto de la caída de ventas se asocia a la falta de efectivo y al retroceso de los pagos digitales, y con qué rezago?**

### Referencias públicas usadas como anclas (reportes de prensa, no verificados por el autor)
- Antes del shock, ≈ 80% de las compras en multitiendas se hacía por medios digitales; después, ≈ 70% de las operaciones pasó a efectivo.
- Caídas de ventas de hasta ≈ 70% en los días más críticos y ≈ 30% a la semana siguiente.
- Reintegro parcial de depósitos: ≈ 41% de los clientes afectados a los pocos días y ≈ 93% hacia mediados de mes.

## Hallazgos del análisis (sobre el gemelo sintético)

| Sector | Pagos digitales pre → mínimo | Caída máx. de ventas | Caída prom. semana 2 | Días de ventas perdidos (a 15-nov) |
|---|---|---|---|---|
| Comercio minorista | 80% → 13% | 52% | 30% | 4,4 |
| Gastronomía | 55% → 15% | 35% | 25% | 3,2 |
| Servicios | 70% → 29% | 28% | 19% | 2,6 |

1. **La relación aparece con el evento.** Antes del shock, la correlación cruzada entre pagos digitales y ventas es débil (|r| ≈ 0,12); después sube a ≈ 0,56, con el conductor **precediendo a las ventas ≈ 1 día**.
2. **La liquidez se normaliza más rápido que la confianza.** Dinero disponible y escasez de efectivo recuperan la mitad del desvío en ≈ 4-5 días; los pagos digitales, en ≈ 28 días, y la confianza, en ≈ 35. El daño persistente viene de la **desconfianza**, no de la falta de circulante *(consecuencia del mecanismo sembrado; con datos reales habría que contrastarlo)*.
3. **La CCF no puede separar canales.** Todos los conductores saltan el mismo día y son casi colineales (|r| > 0,9): todos dan su máximo en el mismo rezago, aunque el generador siembra rezagos distintos (0, 1, 3 y 7 días). **No es un fallo: es la lección.** Con un único shock hay que ser explícito sobre lo que la técnica identifica (el *momento*) y lo que no (el *canal*).
4. **Sensibilidad a la recuperación de la confianza:** reducir la vida media de la desconfianza de 45 a 20 días reduce los días de ventas perdidos a fin de año en ≈ 25%.

## Método

- **Contrafactual:** `brecha = ln(ventas / (nivel base × estacionalidad histórica))`, con el nivel base estimado solo con datos pre-shock. Evita confundir Black Friday, Navidad y aguinaldo con el efecto del shock.
- **Correlación cruzada** (`src/ccf.py`): `r(k) = corr(X[t], Y[t+k])`; k > 0 ⇒ el conductor precede a las ventas. Incluye **primeras diferencias**, **pre-blanqueo Box-Jenkins** (AR(5)), banda ±1,96/√n y **bootstrap de bloques** del rezago del máximo.
- **Verificaciones:** el rezago de una serie de juguete (3 días) se recupera con r = 0,96; comparación pre vs. post-shock; matriz de colinealidad entre conductores.
- **Recuperación:** días hasta recuperar 50% y 90% del desvío máximo por variable (descriptivo, sin modelo).
- **Simulador:** reejecuta el gemelo con otra persistencia de la desconfianza y otra velocidad de reintegro.

> **Decisión de diseño:** la serie es **diaria** (365 días, ene-dic 2026) y no semanal, porque con datos semanales solo habría ≈ 6 observaciones post-shock y los rezagos de pocos días serían inidentificables.

## Estructura

```text
paraguay-payments-shock-analytics/
├── data/raw/payments_daily.csv     # serie sintética diaria
├── data/processed/                 # features_daily.parquet, shock_kpis.csv
├── src/
│   ├── data_generator.py           # gemelo sintético con rezagos conocidos
│   ├── features.py                 # validación, contrafactual, brechas, KPIs, recuperación
│   └── ccf.py                      # correlación cruzada, pre-blanqueo, bootstrap
├── notebooks/01_explicacion.ipynb  # narrativa paso a paso (ejecutado)
├── app.py                          # dashboard Streamlit + Plotly
├── tests/test_payments.py          # 8 tests
├── build_notebook.py               # reconstruye y ejecuta el notebook
├── requirements.txt
└── README.md
```

## Ejecución

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generator      # genera data/raw/payments_daily.csv
python -m src.features            # produce features y KPIs
streamlit run app.py              # dashboard (5 pestañas)
python build_notebook.py          # opcional: regenera el notebook
python -m pytest -q               # tests
```

## Con datos reales

Reemplazar `data_generator.py` por la carga de series diarias con las columnas de `features.REQUIRED`: participación de pagos digitales y ventas por sector, retiros en cajeros, indicador de liquidez disponible y un índice estacional histórico (promedio de los mismos días de años previos). Para **identificar canales**, añadir variación independiente: varios episodios, diferencias entre sectores o regiones, o microdatos de comercios.

## Limitaciones

- **Gemelo sintético:** las cifras ordenan la discusión; no sustituyen datos reales.
- **Asociación, no causalidad.** Un único shock común impide atribuir el efecto a un canal.
- **Magnitudes ancladas a reportes de prensa**, no a estadísticas oficiales; la caída de ≈ 70% se refiere a los días más críticos y a ciertos comercios, no a un promedio nacional.
- **Proyección** desde el 16-nov: escenario hipotético, no pronóstico.
- Proyecto educativo y de portafolio; no refleja la situación de ninguna entidad real.

## Competencias demostradas

Series de tiempo (CCF, pre-blanqueo, bootstrap de bloques, contrafactuales estacionales) · diseño de gemelos sintéticos con verdad conocida para validar métodos · análisis causal con honestidad sobre sus límites · ingeniería de datos y pruebas · visualización ejecutiva · contexto de pagos digitales y sistema financiero de Paraguay.
