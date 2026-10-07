# Auditoría de números — sesión `debugger-2`

2026-10-04. Fue de solo lectura: no se tocó código, base, parquets ni `experimentos/`. No se entrenó
nada. Todo lo CONFIRMADO sale de **predecir con los modelos que ya están en caché** (`lgb.Booster(model_file=…)`),
comprobando antes que `feature_name()` coincide con las predictoras actuales. Los scripts están en el
scratchpad de la sesión (`a_sim.py`, `b_c102.py`, `d_ens.py`, `e_cortes.py`).

**CONFIRMADO** = lo recalculé. **SOSPECHADO** = surge de leer el código o de inferir sin poder medir.

No pude leer la lista de submits del bot (`zulip/respuestas/192436.json`): el clasificador de permisos
la bloqueó. Así que el mapeo submit → archivo lo tomé de los mensajes de aceptación, por orden y hora,
y de la curva que me pasaste.

---

## Resumen: qué se retracta y qué se sostiene

| ítem | afirmación | veredicto |
|---|---|---|
| a | "la curva de corte es una curva" | **SE SOSTIENE**: nested, misma columna, sin empates |
| a | "el óptimo es 9.000" | **MATIZAR**: la forma es mayormente ruido del 25%. Igual empuja hacia un corte corto |
| b | "pesos 0,25 gana: +8,6 M, 8/10, p=0,0078" | **RETRACTAR como resultado general**: vale sólo a 11.000–12.500. **A 9.000 da −0,1 M, 5/10, p=0,94** |
| b′ | "el cambio de target cayó donde decían los folds (+2,34 público)" | **RETRACTAR**: compara 14.500 envíos/3 semillas contra 11.000/20 |
| b″ | "base_pesos25 le gana a base_baja2 por 14,3 M reales" (c103) | **MATIZAR**: es el ensamble de 5 semillas en 202106. Pareado por semilla en 202106 da **+3,3 M** |
| c | "los folds no predicen el público: Spearman 0,00" | **RETRACTAR como evidencia**: con n=4 no hay poder (p=1). No dice ni que sí ni que no |
| d | "el ensamble le gana al 80–85% de las semillas" | **SE SOSTIENE a 11.000.** A 9.000 es 65% / 45%. El código está bien |
| e | "recencia: +10,6 M, 18/20" | **SE SOSTIENE** entre 9.000 y 13.000. A 8.000 se da vuelta |
| e | "λ=0,15: +8,0 M, 10/10" | **MATIZAR fuerte**: a 9.000 da +1,9 M, 5/10, p=0,68, y a 8.000 da **−5,8 M, 2/10**. En el fold C, **−3,6 M, 2/10** |
| f | "202104 le gana a 202103 con los mismos positivos: +34,2 M" | **SE SOSTIENE**: el recorte está bien hecho. El p sólo cubre la semilla |

**El patrón común**: casi todo se midió a **11.000 fijo y en 202106**. 202106 es el mes de mayor
prevalencia (1.098) y su K\* ≈ 14.000. Las variantes que "ganan" ahí corren valor hacia la cola de la
curva, y la entrega final corta en **9.000**, que es donde varias de esas ventajas desaparecen.

---

## (a) La curva de corte en agosto

### a1. Los CSV son nested y salen de la misma columna — CONFIRMADO

Recalculé `ensamble_por_rank` sobre las 20 columnas `s*` de `c107_pesos25/scores_202108.parquet`, y es
**idéntico** a la columna `ensamble` (`np.allclose` = True). Para cada CSV comparé contra el top-k de esa columna:

| archivo | k | = top-k(`ensamble`) |
|---|---:|:-:|
| `corte_8000.csv` | 8.000 | ✓ |
| `c118_corte_8500.csv` | 8.500 | ✓ |
| `corte_9000.csv` | 9.000 | ✓ |
| `corte_9500.csv` | 9.500 | ✓ |
| `c118_corte_10000.csv` | 10.000 | ✓ |
| `corte_10250.csv` | 10.250 | ✓ |
| `c107_pesos25_ensamble.csv` (el punto 11.000 = 92,675) | 11.000 | ✓ |
| `corte_12500.csv` | 12.500 | ✓ |

Cada uno ⊂ el siguiente, sin excepciones. En cada valor de corte hay **un solo** score, o sea que no
hay empates. **La curva es una curva.**

### a2. `c118_base_9000.csv` NO es `corte_9000.csv` — CONFIRMADO

Difieren en **60 ids** (99,33% de solapamiento). `c118_base_9000` es **exactamente** el top 9.000 del
ensamble por rank de **las primeras 10 semillas** (`s[cols[:10]].rank(pct=True).mean(1)` → 9.000/9.000).
`corte_9000` es el de las 20. Hay dos consecuencias:

- Si en c118 se comparó `v101`/`recencia`/`v2corr` contra `c118_base_9000`, el contraste es contra 10
  semillas. Si `c117_v2_ens20_9000` (20 semillas) se comparó contra esa base, además se mezcla el
  tamaño del ensamble con el cambio de modelo. El efecto es chico, pero existe.
- SOSPECHADO, por el orden de los mensajes del día 2: `c118_base_9000` sería el submit **96,1675**.
  Si es así, hay **dos lecturas del "mismo" modelo a 9.000**: 97,24 (20 semillas) y 96,17 (10
  semillas), a 60 ids de distancia. **Esos 1,07 M de diferencia son del mismo orden que el pico de la
  curva** (97,24 contra 95,37 a 9.500). Conviene verificarlo con `list submits`.

### a3. Cuánto de la forma de la curva es el 25% — CONFIRMADO (simulación) + aritmética del bot

**Aritmética.** Las ganancias del bot son múltiplos exactos de 0,0275 (97,24/0,0275 = 3.536). Entre
dos cortes nested, Δ/0,0275 = 40·P − N, donde P = positivos públicos en el tramo y N = ids públicos
en el tramo (≈ 25% del tramo). Con N ≈ 125 por cada 500 envíos:

| tramo | Δ público | P (positivos públicos) |
|---|---:|---:|
| 8.000 → 8.500 | +1,98 | ~5 |
| 8.500 → 9.000 | +2,67 | ~5–6 |
| 9.000 → 9.500 | −1,87 | ~1–2 |
| 9.500 → 10.000 | −2,75 | ~0–1 |
| 9.000 → 11.000 | −4,57 | ~8–9 en ~500 ids |

**El "óptimo en 9.000" se apoya en unos 10 positivos públicos en 8.000–9.000, contra 1 o 2 en
9.000–10.000.** Con una tasa marginal esperada de ~3 por tramo, esa diferencia es compatible con Poisson.

**Simulación.** Usé el mismo pipeline: el ensamble de 20 semillas de c107 en los folds, scoreado en
202105 y en 202106, con 4.000 sorteos de un 25% público y la misma grilla de 8 cortes.

| | 202105-like (870 B+2) | 202106-like (1.098 B+2) |
|---|---:|---:|
| P(argmax público = 9.000) | 23% (8.000: 23%, 11.000: 18%) | 0,7% (11.000: 51%, 12.500: 37%) |
| G(9.000) − G(11.000) público: media ± sd | +1,4 ± 3,1 M | −7,7 ± 4,1 M |
| P(público muestre ≥ +4,57, lo observado) | 16% | **0,08%** |

- Con prevalencia tipo 202105 **el argmax público cae casi al azar entre 8.000, 9.000 y 11.000**.
  Que haya caído en 9.000 no localiza el óptimo con más resolución que ±1.500.
- **El público y el privado del mismo tramo correlacionan −1** (la suma es fija), como ya había medido
  z301. Lo que el público ganó de más en 8.000–9.000, el privado lo tiene de menos. Bajo una verdad
  tipo 202105, después de ver +4,57 en público, el privado espera para 9.000 contra 11.000 apenas
  **≈ +0,4 M por cada 25%** (unos +1,1 M sobre el 75%). La ventaja de 4,6 M **no se va a repetir**.
- Lo que **sí** dice la curva: observar +4,57 es ~200 veces más probable si agosto se parece a
  202105 que si se parece a 202106. **Es evidencia de que 202108 tiene prevalencia baja**, y eso
  justifica un corte corto. O sea, 9.000 es defendible como "corte corto", no como "el óptimo".

Además, 97,24 es el **máximo de 8 lecturas** sobre la misma partición, así que como estimador del
nivel tiene sesgo optimista.

**Veredicto (a)**: no hay que retractar la curva. Hay que retractar la **precisión**: "el óptimo está en
8.000–11.000, más cerca de lo corto, y agosto parece de prevalencia baja", no "el óptimo es 9.000".

---

## (b) "Pesar BAJA+1 en 0,25 gana: +8,6 M, 8/10, p=0,0078"

### b1. El código está bien construido — CONFIRMADO

- `preparar(target='pesos', 0.25)`: y = 1 para BAJA+1 y BAJA+2, w = 0,25 para BAJA+1 y 1 para el resto.
- La ganancia se mide contra `BAJA+2` del mes de validación (`c102:43`) también con target fusionado.
- El pareado está bien: las mismas 5 semillas, los mismos 2 folds y el mismo orden de append (fold → semilla) en las dos configs. Se resta `config − base` fila a fila (`c102:105`).
- Las dos configs tienen el **mismo** corte minimax (11.000), así que esta diferencia no viene de cortes distintos.
- Reproduje con los modelos en caché: **+8,58 M, 8/10, p=0,0078**, exacto.

### b2. El resultado depende del corte, y en el corte de la entrega no existe — CONFIRMADO

Mismo pareado, barriendo K (`b_c102.py`):

| K | Δ medio | gana | p | Δ fold 202105 | Δ fold 202106 |
|---:|---:|:-:|---:|---:|---:|
| 8.000 | +5,5 | 7/10 | 0,078 | +10,6 | +0,4 |
| 8.500 | +4,5 | 6/10 | 0,195 | +10,8 | −1,8 |
| **9.000** | **−0,1** | **5/10** | **0,94** | +9,5 | **−9,7** |
| 9.500 | +0,9 | 5/10 | 0,91 | +12,1 | −10,3 |
| 10.000 | +2,3 | 6/10 | 0,54 | +9,9 | −5,3 |
| 10.500 | +5,5 | 6/10 | 0,105 | +12,1 | −1,1 |
| **11.000** | **+8,6** | **8/10** | **0,008** | +13,9 | +3,3 |
| 11.500 | +8,7 | 9/10 | 0,010 | +13,9 | +3,5 |
| 12.500 | +7,3 | 8/10 | 0,010 | +11,0 | +3,5 |
| 14.000 | −0,8 | 5/10 | 0,89 | +3,3 | −4,8 |

- **202105 favorece a pesos25 en todos los cortes. 202106 no, salvo en la ventana 11.000–13.000.** Es
  lo que el CLAUDE.md ya intuía ("el peso cambia en qué mes anda bien"), pero el "8/10, p=0,0078" lo
  esconde.
- **Pseudorreplicación**: las 5 semillas de un fold comparten train y validación. El p mide el ruido
  de semilla, no la generalización a otro mes, y ahí n = 2 folds.
- Es el mejor de 8 configs (`resumen.sort_values(...).iloc[0]`) sin corrección por comparaciones múltiples.

**Número correcto para la entrega a 9.000: pesos25 − baja2 = −0,1 M (5/10, p=0,94). No hay evidencia
local de que pesar 0,25 ayude en el corte que se va a usar.** No hay evidencia de que perjudique
tampoco: el promedio sobre cortes es positivo y en ningún corte es claramente negativo.

### b3. El "check lindo" del público no es un check — CONFIRMADO por el ledger

CLAUDE.md, Fase 3: *"Local: +8,6 M → ~+2,15 M público. Observado: +2,34 M"*. Los dos submits son:

- `c100_esqueleto`: `baja2`, **3** semillas, **14.500** envíos (ledger: los tres archivos con 14.500) → 89,0175
- `c107_pesos25`: `pesos 0,25`, **20** semillas, **11.000** envíos → 91,355

**Cambian a la vez el target, el corte y la cantidad de semillas.** La curva pública del ensamble de
c107 cae 4,5 M solo entre 11.000 y 12.500, así que pasar de 11.000 a 14.500 probablemente explique más
que el target. **Retractar** "el cambio de target cayó donde decían los folds".

### b4. Los "14,3 M reales" de c103 — CONFIRMADO

`ruido_publico.parquet`: `base_pesos25` contra `base_baja2` da `dif_real_M` = 14,30. Pero eso se
calcula sobre **202106 solo**, con el **promedio de probabilidades de 5 semillas** (`c103:50`).
Pareado por semilla en 202106 a 11.000 da **+3,3 M** (tabla b2). El "el público acierta el orden 90%"
está calculado con la diferencia de 14,3. Con una diferencia real de ~3 M y una sd pública de la
diferencia de ~2,8 M, acertaría ~60%.

---

## (c) "Los folds no predicen el público: Spearman 0,00" — CONFIRMADO

Lo reproduje: sale **exactamente 0,00** con {`c107` base (local 399,7; público 91,355, media de 20
semillas), v1_recencia, v2_v101, v3_combo}, local = 202106 de `c113/resumen.json` contra público a 11.000.

- **n = 4.** La distribución nula de Spearman con n=4 tiene 24 permutaciones: P(|ρ| ≥ 0,8) = 8,3%.
  ρ = 0 tiene p = 1. **Con 4 puntos, ni ρ = ±0,8 sería significativo.** El número no dice nada.
- **Es frágil**: si se cambia la base por `v4_reciente` (que es la misma base, ver auditoría anterior)
  da **+0,6**. Con la local de 202105 da **−0,89**.
- El rango público entre las 4 es de 4,56 M, por debajo del umbral de 6,3 M que el propio c103 calcula
  para distinguir dos submits. **El público no ordena estas 4 configs**, así que no hay contra qué correlacionar.
- Lo que preguntabas: que los locales salgan de modelos entrenados en [202103, 202104] y los públicos
  de modelos de 4 meses **no es el problema**. Así se mide si una *configuración* transfiere. El
  problema real, además de n=4, es otro: v2/v3 tienen features acumuladas que en los folds son casi
  constantes y en 202108 no (está declarado). Para esas dos se sabe *a priori* que el local no mide lo
  mismo que el público.

**Retractar** "los folds no predicen el público". La frase correcta es: "con 4 configs separadas por
menos que el ruido público, no podemos saber si los folds predicen el público".

---

## (d) Ensamble por rank — CONFIRMADO

- `ensamble_por_rank`: `rank(pct=True)` es ascendente, así que más score da más rango. El promedio de
  rangos y `top_k` toman los de mayor valor. **La orientación está bien**: tasa de BAJA+2 en el top
  1.000 del ensamble = 10,3% / 10,7%, contra 0,03% en el fondo.
- Reproducido con los 20+20 modelos de folds de c107:

| | K | ensamble | media de semillas | le gana a | promedio de probabilidades |
|---|---:|---:|---:|---:|---:|
| 202105 | 11.000 | 254,1 | 249,4 | **85%** | 254,1 |
| 202106 | 11.000 | 405,9 | 398,9 | **80%** | 408,1 |
| 202105 | **9.000** | 259,6 | 257,2 | **65%** | 259,6 |
| 202106 | **9.000** | 375,1 | 370,3 | **45%** | 377,3 |

- "80–85%" es correcto **a 11.000**. A 9.000 baja a 65% / 45%: en 202106 la distribución de semillas
  está sesgada y el ensamble queda por debajo de la mediana, aunque por encima de la media.
- Lo que sí se sostiene en los dos cortes: **el ensamble supera a la semilla media por +2,4 a +7 M**.
  Esa es la frase robusta.
- Detalle: el promedio de **probabilidades** empata o le gana al de rangos (+2,2 M en 202106). La
  justificación del docstring ("las escalas no son comparables") no aplica entre semillas del mismo
  target. No cambia ninguna decisión.

---

## (e) Recencia y decaimiento

### e1. Código — CONFIRMADO por lectura

- `edad` (`c110:47`, `c119:79`, `c113:39`) = `(año·12 + mes)` de referencia menos el de la fila: el
  cruce de año está bien. Igual todos los meses son de 2021.
- `w_clase * lam**edad`: `sub` y `preparar` filtran con la misma máscara sobre el mismo `data`, así
  que están alineados (ya estaba verificado).
- Todos los números reportados reproducen exactamente desde `por_semilla.parquet`: c108 +10,62 18/20
  p=0,0002; c110 λ=0,15 +8,03 10/10 p=0,0020, λ=0,05 +7,15 8/10, etc.

### e2. Dependencia del corte — CONFIRMADO (`e_cortes.py`, modelos en caché, val 202106)

| K | solo 202104 − 202103+202104 (20 sem.) | λ=0,15 − λ=1 (10 sem.) |
|---:|---|---|
| 8.000 | **−3,3 M, 4/20, p=0,08** | **−5,8 M, 2/10, p=0,03** |
| **9.000** | **+9,0 M, 16/20, p=0,0004** | **+1,9 M, 5/10, p=0,68** |
| 10.000 | +10,6 M, 18/20 | +6,4 M, 9/10 |
| 11.000 | +10,6 M, 18/20 | +8,0 M, 10/10 |
| 12.000 | +8,8 M, 19/20 | +8,8 M, 9/10 |
| 13.000 | +11,9 M, 19/20 | +10,8 M, 9/10 |

- **Recencia (c108): se sostiene a 9.000.** Matiz: a 8.000 se da vuelta.
- **Decaimiento (c110): a 9.000 desaparece y a 8.000 perjudica.** El decaimiento no mejora el ranking
  de arriba: mejora la cola (10.000+), que es justo la parte que en 202106 tiene valor y que en
  agosto, según (a), parece no tenerlo.
- Selección: λ=0,15 es el mejor de 5 λ probados. El p=0,002 no está corregido por eso.
- Un solo mes de validación (202106) y p de semilla: valen las mismas advertencias que en b2.

### e3. Fold C (`c119`) — CONFIRMADO

- Lo que dice el docstring está bien. BAJA+1 en 202107 sólo necesita 202108, que se tiene. Los
  positivos de train (BAJA+2/BAJA+1 hasta 202105) no están en 202107. No hay leakage.
- **Qué mide de verdad**: los 1.103 BAJA+1 de 202107 son, salvo ~5 altas, **los mismos 1.098 clientes
  que son BAJA+2 en 202106**. El fold C no es independiente del fold B *en personas*. Pregunta por
  las mismas bajas, un mes después y con features más frescas. El sesgo hacia señales tardías está
  declarado.
- El proxy paga por BAJA+1 a 11.000. Sirve para ordenar, no para dar nivel, y eso también está declarado.
- **Resultado**: `decaimiento λ=0,15` da **−3,6 M, 2/10, p=0,086** contra "todos los meses". Los dos
  recientes dan +1,7 M (6/10) y solo el más reciente +0,1 M (5/10). **El fold C no corrobora el
  decaimiento, apunta en contra.** Junto con e2, "λ=0,15: +8,0 M, 10/10" no da para "lo primero a
  corregir" en una entrega a 9.000.

---

## (f) c109: recorte de 202104 a 960 positivos — CONFIRMADO

- El recorte **tira filas BAJA+2** (no las convierte en negativos), con `rng = default_rng(SEMILLAS[1])`.
  Queda 960/163.105 = 0,589%, contra 202103: 960/162.900 = 0,589%. **La tasa base queda igualada, no
  distorsionada.** Los BAJA+1 (peso 0,25) no se tocan: 964 en 202104 contra 1.019 en 202103, una
  diferencia menor que no favorece a 202104.
- Reproduce: recortado − 202103 = **+34,21 M, 10/10, p=0,002**. Completo − recortado = **+4,95 M,
  7/10, p=0,016**.
- Matices:
  - **Un solo sorteo de recorte.** Las 10 "semillas" varían LightGBM, no qué 179 positivos se tiran,
    así que el p no cubre la variabilidad del submuestreo. Para el +34 M no importa: es enorme frente
    a cualquier sorteo plausible. Para el "+5,0 M de los 179 extra" sí importa: es una sola realización.
  - c108 confunde "agregar 202103" con "duplicar filas con `min_data_in_leaf`=174 y 250 rondas fijos".
    Además, los 960 BAJA+2 de 202103 reaparecen como BAJA+1 en 202104: el train de dos meses tiene
    960 casi-duplicados positivos. No es leakage (la validación es otro mes), pero sí un confusor de
    "el mes viejo estorba".
  - "El mes pesa siete veces más que los positivos" (34,2/5,0) compara un efecto bien medido con uno
    de un solo sorteo. Como orden de magnitud está bien; como cociente, no.

---

## Lo que NO encontré

- Ningún otro caso de orden de filas (bug #1). Los 11 CSV de la curva son top-k exactos de su columna,
  y `scores_202108.parquet` de c107 recompone su `ensamble` exacto.
- Ninguna otra instancia de gemelos cruzando un split: c102/c107/c108–c110/c119 validan siempre out-of-time.
- Ningún error de alineación entre peso de clase y peso por recencia.
