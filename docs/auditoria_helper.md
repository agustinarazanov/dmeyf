# Auditoría externa (sesión `helper`, 2026-10-06)

Pedido de `competencia-1`: errores que no tiran error, mejoras con mecanismo, y qué de la lista
de la reunión no vale la pena. Leí `CLAUDE.md`, `competencia.py`, `registro.py`, `reproducir.py`,
`fe_panel.py`, los `c1*.py` del directorio y los `c13x`–`c17x` que quedaron en `/tmp`. Corrí tres
sondas livianas (predicciones sobre modelos cacheados y 15 modelos nuevos de 2.000 rondas).

## 1. Lo que NO encontré

No hay bugs de alineación en el camino de la entrega. `preparar()` saca `X`, `y` y `peso` del
mismo frame filtrado; en todos los scripts `es` sale del mismo `val` sobre el que se predice;
`cargar()` ordena por `(id, mes)`, así que los parquets derivados comparten orden. `c171` alinea
por id explícitamente. `reproducir.py` y `c174` hacen lo mismo y ya están en **10.500**: el
«pendiente» del `CLAUDE.md` sobre `CORTE = 10_000` está resuelto y conviene borrarlo.

El target de `_SQL_BASE` es correcto: 202107 queda con 1.103 `BAJA+1` y 163.245 nulos, 202108
todo nulo, y ningún mes etiquetado tiene nulos. La regla de gap 2 para los folds es correcta.

## 2. Errores

### 2.1 La simulación del 116 usa el mecanismo equivocado (conceptual, afecta una conclusión)

«Si todos tuvieran nuestro modelo, el máximo de 120 por el sorteo del 25% daría 116,4» **resortea
la partición para cada alumno**. La partición pública es **fija y la misma para todos**: 120 copias
de nuestro modelo con semillas distintas darían 120 lecturas con sd ≈ 1,5–2,3 M (el `std dev` del
bot), y el máximo esperado sería ~96–97, no 116. El argumento escrito no vale.

La conclusión («116 es compatible con no tener nada mejor») puede sostenerse por **otra** vía: la
sd de la *diferencia* pública entre dos modelos distintos de igual calidad, que ustedes midieron en
3,1 M (variantes del mismo pipeline) a 16,3 M (datasets distintos). Con 120 alumnos de pipelines
diversos, un máximo de 116 sobre una media de ~90 no es extremo. Pero hay que decirlo así.

Relacionado: «z301 midió que público y privado correlacionan −1 para un modelo fijo» es una
tautología (suman un total fijo), no una medición. No citarlo como evidencia.

### 2.2 Las pendientes del FE cambian de significado con el mes (construcción, afecta `fe228`)

`sql_slope` usa la ventana `historia` sin marco = *unbounded preceding*. Medido en
`competencia_01_fe.parquet`, `ctrx_quarter__slope`:

| mes | NaN | \|slope\| medio |
| --- | ---: | ---: |
| 202103 | 100% | — |
| 202104 | 0,9% | 9,88 |
| 202105 | 0,9% | 7,93 |
| 202106 | 0,9% | 7,01 |
| 202107 | 0,7% | 6,17 |
| **202108** | 0,8% | **5,70** |

En abril es una pendiente de 2 puntos; en agosto, de 6. Los umbrales que el árbol aprende sobre
pendientes de 2–4 puntos se aplican en agosto a pendientes de 6. **Es drift por construcción**, y
los folds lo exponen menos que el público porque mayo y junio están más cerca del train en
profundidad de historia. Es un mecanismo concreto para «el FE gana en los tres folds y pierde en
público». Las 34 columnas `__slope` están afectadas; `__lag1` y `__delta1` no (0,9% nulos fuera
de marzo). Arreglo: marco fijo (`rows between 2 preceding and current row`) o sacar las pendientes.

### 2.3 Menores

- `c172_corte_final.py`: `PREVALENCIAS` incluye **1.000**, que no es un mes observado
  (870/960/1.098/1.139). Inventa un escenario.
- `c165_barrido_poda.py:96`: `abs(hash(nom))` es el `hash` de Python, aleatorizado por proceso
  (`PYTHONHASHSEED`). El nombre del archivo cambia en cada corrida y el caché nunca pega entre
  corridas. No afecta resultados, sólo tiempo.
- `c160_corte_privado.py:108` sigue usando `sd/√20` como error de cada punto público, que es la
  barra que ya retractaron. El script quedó con la lectura vieja.
- `top_k` y `ganancia_acumulada` usan `argsort` no estable: los empates de score se rompen por
  posición de fila. Con el orden canónico es reproducible, pero no es «desempatar por id».

## 3. Mediciones nuevas

### 3.1 Subajuste por rondas: refutado

`z601` eligió `learning_rate = 0,0077` con Optuna y *early stopping* hasta 1.000 rondas, y después
comentó `best_iter` y dejó 250. Sospeché subajuste. Un entrenamiento de 2.000 rondas por
(fold, semilla), evaluado con `num_iteration`, 5 semillas, captura a K = 10.500:

| rondas | 202105 | 202106 | 202107 (`BAJA+1`) |
| ---: | ---: | ---: | ---: |
| **250** | 56,41 | **56,54** | **68,65** |
| 500 | 56,60 | 54,46 | 67,94 |
| 1.000 | 56,44 | 54,12 | 68,32 |
| 2.000 | 56,85 | 55,05 | 68,27 |

Pareado contra 250: mayo +0,0 a +0,4 pts (3/5), **junio −1,5 a −2,5 pts (0/5 en todos)**, julio
−0,3 a −0,7 (0/5). Más rondas no ayuda y en dos folds perjudica. No probé menos de 250; la forma de
la curva de junio sugiere que podría ser aún mejor, pero es un efecto de 1–2 puntos.

### 3.2 Los `BAJA+1` en el top-K: costo medido y techo del oráculo

Con los modelos cacheados de `c107` (pesos 0,25), 5 semillas:

| val | `BAJA+1` en el mes | en el top 10.500 | captura de B+1 | captura de B+2 |
| --- | ---: | ---: | ---: | ---: |
| 202105 | 1.143 | **813** | 71,1% | 56,7% |
| 202106 | 874 | **584** | 66,8% | 56,6% |

El modelo captura **mejor** a los que se van el mes que viene (costo puro) que a los que pagan.
Si un oráculo los sacara todos del ranking y corriera el resto: **+15,0 M (mayo) y +22,4 M
(junio) al mes** a 10.500; a 9.000, +17,2 y +24,4. Es el único lugar que vi con 15–24 M/mes sobre
la mesa, más que todo lo que probaron.

Pero el intento simple no lo captura. Un *demoter* de horizonte 1 (label `BAJA+1`, entrenado hasta
el mes anterior a la validación, limpio), restado por rango con peso α:

| α | mayo @10.500 | junio @10.500 |
| ---: | ---: | ---: |
| 0,1 | −5,5 (0/5) | +4,2 (3/5) |
| 0,3 | −18,5 (0/5) | +7,9 (4/5) |
| 0,5 | −48,0 (0/5) | +3,1 (3/5) |

Firma mayo/junio otra vez, igual que `c140`. No separa «se va en 1» de «se va en 2» desde la misma
foto. **Descartar como está**; queda documentado el techo.

## 4. La lista de la reunión

| propuesta | veredicto | por qué |
| --- | --- | --- |
| Optuna | **no esta semana** | El objetivo correcto sería captura media en los 3 folds limpios a K fijo, nunca CV dentro del mes. Pero el piso de los folds (8–10 M) es mayor que lo que una búsqueda puede distinguir en 4 días, y las rondas ya están contestadas |
| Lado alto del corte | **ya hecho** | 10.500 por minimax es exactamente eso; la asimetría de la curva lo justifica |
| FE histórico «suma mucho» | **sólo con arreglo** | El consejo viene de datasets de 24+ meses. Acá 1 de 4 meses de train no tiene historia y las ventanas de 3 meses sólo están completas en junio. Si se insiste: `lag1`/`delta1` sin pendientes *unbounded* (§2.2), y es lo único con los 3 folds a favor |
| Un modelo por mes | **no** | Cada modelo tiene ¼ de los datos; medido −2 M (z ≈ −0,6) |
| Drift monetario, escalar | **no** | Dos variantes medidas (deflactar, `rank_cero_fijo`): una con firma mayo/junio, otra pierde en ambos. Con 4 meses de train y +15% de inflación el árbol lo absorbe |
| Sacar préstamos | **no** | Circular (el lift 1,02 se midió sobre lo que valida el fold B); 7 de 8 mediciones en contra |
| Tendencias | **igual que FE** | Mismo problema de ventana |
| «Sumar uno si aparece X» | **no esta semana** | Contadores acumulados tienen rango 0–1 en marzo y 0–5 en agosto; con ventana fija queda la rampa inicial. Medido ≈ 0 |
| Papers de churn | **no** | No para el domingo |
| RF para features, stacking | **no** | Base y FE correlacionan 0,99 y promediar no diversificó; un RF sobre las mismas 150 columnas tampoco va a hacerlo |
| Más semillas | **gratis, chico** | sd entre semillas 1,5–2 M por archivo; de 20 a 50 el ensamble gana ~0,2–0,4 M público |

## 5. Sobre bajar el piso de detección

No se puede con el público (~275 positivos). Lo único que agrego a lo que ya hacen: **medir el
solapamiento del top-K antes de gastar un submit**. Si dos configuraciones comparten >95% de los
ids, la diferencia pública va a ser <2 M por construcción (los 500 ids que difieren contienen ~10
positivos en el mes, ~2–3 en el público) y el submit no va a decir nada. Es un chequeo de 1 s que
evita submits ilegibles, y explica por qué `histmin` dio signos opuestos.

## 6. Recomendación

La entrega como está: base 150 columnas, `pesos 0,25`, 202103–202106, 20 semillas, ensamble por
rango, 10.500. Nada de lo que medí ni de lo que leí justifica tocarla. El riesgo mayor de acá al
domingo es cambiarla por ruido.

---

# Segunda ronda (respuesta a las dos preguntas de `competencia-1`)

## A. ¿Se puede separar `BAJA+1` de `BAJA+2` sin la firma mayo/junio? No. Cerrarlo.

Medido directamente: AUC de «es `BAJA+1`» contra «es `BAJA+2`» **entre los que se van** en el mes
de validación, con tres scores distintos (el modelo de la entrega, un modelo de horizonte 1 y un
separador entrenado sólo con los que se van del train, `BAJA+1` contra `BAJA+2`):

| val | n se van | `P_pesos25` | `P_horizonte1` | separador directo | separador dentro del top 10.500 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 202105 | 2.013 | 0,624 | 0,637 | **0,535** | 0,509 |
| 202106 | 1.972 | 0,573 | 0,603 | **0,609** | 0,598 |

Tres conclusiones. (1) La separación es **débil**: AUC ~0,6 es casi nada. Un mes antes de irse y
dos meses antes de irse se ven prácticamente iguales en la foto; el único marcador que distingue
(`*_status` en cierre) va de 11% a 4,6%, el resto es «desenganchado» en los dos casos. (2) La
separación es **distinta por mes**: el separador directo da 0,535 en mayo y 0,609 en junio. Por eso
las dos construcciones (dos etapas y demoter) fallan igual: no es el método, es que la señal de
«cuánto falta» cambia de cohorte en cohorte. (3) **No es un artefacto del K fijo**: evaluado en
captura en el K\* de cada fold, el demoter pierde en mayo también a 8.000 (50,2 → 48,6) y gana en
junio también a 14.000 (66,2 → 66,9). El techo del oráculo (+15–24 M/mes) existe pero no hay
instrumento con AUC 0,6 que lo cobre. **Cerrado.**

## B. ¿Qué familia de enfoque daría 7 puntos de captura?

### B.1 Primero: 7 puntos de modelo es implausible con estos datos

La ley de envejecimiento que midieron vale ~3 puntos por mes de distancia, y el fold C (distancia
1) da 68,7% contra 56,5% a distancia 2. **Siete puntos equivalen a dos tercios de un mes de
información.** Ninguna feature construida sobre la misma foto hace eso: toda la historia (lags,
deltas, pendientes) suma ~1 punto. Un modelo 7 puntos mejor necesitaría información que nosotros no
tenemos, y el dataset termina en 202108 para todos.

### B.2 La explicación que encaja con TODO: el 116 está ajustado al público

Dos mecanismos, uno pasivo y uno activo, y los dos producen exactamente «nada de lo que probamos
mueve más de 3 M, y hay alguien 19 M arriba».

**Pasivo (selección).** La nueva estimación «P(≥116) = 0%» toma la sd de nuestros 36 submits (3,32)
como la dispersión de un alumno. Esos 36 son casi el mismo pipeline, sin cherry-picking. Un alumno
que arma 30–50 modelos **estructuralmente distintos** (otros features, otros meses, otro target) y se
queda con el máximo público tiene una cola derecha mucho más gorda: la sd de la diferencia pública
entre dos modelos distintos es 3–5 M con 80–85% de solapamiento (ustedes midieron hasta 16 M con
datasets distintos), y el máximo de 50 de ésos está a +2,2 sd. Sumado al argmax del corte (+3 M) y a
un modelo algo mejor (+3–5 M), 116 es alcanzable sin que nadie tenga 7 puntos de captura. El «5,6
sigma» es sobre la distribución equivocada.

**Activo (sondeo del público).** El bot devuelve la ganancia pública de **cualquier** archivo de
8.000–15.000 ids, hasta 20 por submit, 13 submits por día. Un acierto público vale +1,1 M y un
no-acierto público −0,0275 M. Entonces: archivo fijo S₀ de 8.000 ids, y 20 variantes que le sacan
un bloque de 400 ids de muy bajo score (≈0 aciertos) y le meten un bloque de 400 candidatos. La
diferencia contra S₀ es `1,1 × aciertos(bloque) ± 0,3` (el ±0,3 es la incertidumbre sobre cuántos
del bloque cayeron en el público). **Un submit mide los aciertos públicos de 20 bloques; un día,
de 260 bloques = 104.000 clientes.** Refinando los bloques con aciertos (400 → 50 → 10) en dos o
tres días se localizan decenas de positivos públicos a ~10 ids cada uno, y cada uno agregado vale
~+0,8 M público neto. La banda 10.500–15.000 tiene ~110 aciertos públicos: sacar 17 de ahí es
trabajo de dos días. Información-teóricamente alcanza y sobra: 2.600 archivos × ~4 bits ≫ los
~3.000 bits que hacen falta para localizar 275 positivos entre 164.647.

No se puede saber cuál de los dos es, ni si es una mezcla, y no hace falta: **los dos inflan el
público y ninguno mueve el privado**, que es el 75% y el que cuenta. La única evidencia que lo
distinguiría es el privado al cierre. **No recomiendo perseguir el 116 por ninguna vía**; lo
describo porque lo pidieron explícitamente y porque es la única explicación con el orden de
magnitud correcto.

### B.3 Si el hueco fuera real, el único indicio estructural que tengo

Más árboles **perjudican** (junio −2,5 pts, 0/5; julio −0,4, 0/5): el modelo gana por ser débil.
Eso no es subajuste, es que el **corrimiento entre meses domina sobre la capacidad**. La familia
coherente con eso no es «más features» sino «features que signifiquen lo mismo en agosto que en
marzo»: cocientes contra la propia historia y rangos dentro del mes **en reemplazo** de los montos,
no sumados, y pesos de importancia hacia filas parecidas a agosto (validación adversaria *después*
de sacar los dos relojes de calendario, que se llevan el 97% del gain y tapan el resto). Ustedes
probaron variantes de las dos primeras y dieron la firma mayo/junio. No creo que valga el cupo.

## C. FE con `lag1`/`delta1`/`percent_rank` y SIN las 21 pendientes (lo que pidieron)

Tres folds limpios, captura, 10 semillas pareadas, base y `fe228` del caché de `c171`:

| K | fold | base | fe228 | sin pendientes | fe228 − base | sin pend. − base | **sin pend. − fe228** |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.000 | 202105 | 52,66 | 52,93 | 52,92 | +0,28 (5/10) | +0,26 (5/10) | −0,01 (6/10) |
| 9.000 | 202106 | 51,63 | 52,76 | 51,98 | +1,13 (7/10) | +0,35 (4/10) | **−0,78 (1/10, p=0,06)** |
| 9.000 | 202107 | 64,05 | 65,52 | 64,66 | +1,47 (9/10) | +0,61 (6/10) | **−0,86 (1/10, p=0,02)** |
| 10.500 | 202105 | 56,29 | 56,10 | 56,03 | −0,18 (3/10) | −0,25 (4/10) | −0,07 (5/10) |
| 10.500 | 202106 | 56,79 | 57,91 | 57,15 | +1,12 (8/10) | +0,36 (7/10) | **−0,76 (2/10, p=0,07)** |
| 10.500 | 202107 | 68,84 | 70,08 | 69,12 | +1,24 (10/10) | +0,28 (4/10) | **−0,96 (1/10, p=0,008)** |

**Resultado contrario a mi expectativa**: las pendientes son la parte del FE que rinde en los
folds. Sin ellas, el FE vuelve casi al base. El mecanismo del §2.2 (la pendiente de 2 puntos en
abril no es la de 6 en agosto) sigue siendo cierto como descripción, pero **no está demostrado que
sea lo que perjudica en agosto**: en mayo/junio/julio, con 3–5 puntos, la pendiente ayuda. Lo que
pasa con 6 puntos en agosto sólo lo puede decir el público, y el público dijo z = −1,0. La variante
que quedaría por probar es la pendiente con marco fijo de 3 filas, consistente desde 202105 en
adelante; no la corrí porque son 30 modelos más y no cambia la decisión de la entrega.

**Decisión que recomiendo: sigue siendo el base**, por la razón que ya tenían (una medición ruidosa
sobre el mes correcto contra tres limpias sobre meses que no son), ahora con un dato más: la
ganancia del FE depende de una construcción que cambia de significado justo en el mes a predecir.

Cómputo de esta ronda: ~8 min de CPU.

---

# Tercera ronda: ¿régimen o capacidad? Ninguna de las dos: saturación

Dos objeciones de `competencia-1` que acepto: (1) «ninguna familia da 7 puntos» era más fuerte que la
evidencia, porque la ley de envejecimiento se midió dentro de la única familia que probamos (GBDT
sobre la foto con historia aplanada en columnas); lo que sostengo es más débil: *dentro de esa
familia* no hay 7 puntos, y fuera de ella no hay medición. (2) «más árboles perjudican ⇒ gana por
débil» admitía la lectura «sobreajuste al régimen del mes de entrenamiento». La grilla que pidieron
decide entre las dos.

Modelos de **un mes solo** y pooleados, scoreados a distancia ≥ 2, captura, 5 semillas, 250 rondas:

| origen | val 05 (`B+2`) | val 06 (`B+2`) | val 07 (`B+1`) |
| --- | ---: | ---: | ---: |
| `[03]` | 56,55 (d2) | 54,21 (d3) | 67,72 (d4) |
| `[04]` | — | 57,36 (d2) | 67,52 (d3) |
| `[05]` | — | — | 68,70 (d2) |

(K = 10.500; val 07 paga por `BAJA+1`, así que los niveles no se comparan entre columnas.)

**Poolear contra el mes más reciente solo**, pareado por semilla:

| comparación | K 9.000 | K 10.500 |
| --- | ---: | ---: |
| val 06: `[03,04]` vs `[04]` | −0,31 (1/5, p=0,31) | −0,95 (0/5, p=0,06) |
| val 07: `[04,05]` vs `[05]` | −0,54 (1/5, p=0,19) | −0,05 (1/5, p=0,56) |
| val 07: `[03,04,05]` vs `[05]` | −0,36 (3/5, p=1,0) | −0,16 (2/5, p=0,63) |
| val 07: `[03,04,05]` vs `[04,05]` | +0,18 (3/5) | −0,11 (3/5) |

**Lectura.** La hipótesis de régimen predice que poolear perjudica y que perjudica más cuantos más
meses se mezclan. No pasa: tres meses ≈ dos ≈ uno, todo dentro de ±1 punto y sin significancia, y
agregar el mes más viejo a `[04,05]` no mueve nada. La hipótesis de capacidad ya estaba refutada
por las rondas. Lo que queda es lo que la tabla dice literalmente: **triplicar los datos no mejora
el modelo**. Con ~1.000 positivos por mes, un GBDT que no gana nada con 3.000 positivos está
saturado: la señal es simple, está en pocas variables fuertes y un mes alcanza para aprenderla. El
techo de ~57% a distancia 2 lo pone la información de la foto, no la muestra ni la capacidad.

La dispersión entre orígenes a la misma validación es de 1–3 puntos (val 06: 3,15 entre d2 y d3;
val 07: 1,18 entre d2, d3 y d4), o sea el envejecimiento tampoco es una constante: a `BAJA+1` casi
no envejece, a `BAJA+2` sí. Es otra razón para no usar «3 pts/mes» como unidad universal, como
objetó `competencia-1`.

**Consecuencia.** Para la entrega da lo mismo entrenar con uno, dos o cuatro meses; cuatro es la
opción conservadora (más semillas efectivas, menos varianza) y se queda. Para la pregunta de fondo,
las dos líneas (régimen y capacidad) quedan cerradas: la familia está saturada y el margen, si
existe, está en información que la foto no tiene.

---

# Cuarta ronda: la pendiente con marco fijo rinde igual que la unbounded

`competencia_01_fe3.parquet` = `fe228` con las 21 `__slope` reemplazadas por `regr_slope` sobre
`rows between 2 preceding and current row`. Construcción verificada: |slope| medio de `ctrx_quarter`
**7,93 / 7,95 / 8,03 / 8,83** de 202105 a 202108 (contra 7,93 → 5,70 con *unbounded*); 202103 NaN,
202104 dos puntos. De 202105 en adelante la feature significa lo mismo, agosto incluido.

Tres folds, captura, 10 semillas pareadas, base y `fe228` del caché de `c171`:

| K | fold | base | fe228 | slope3 | fe228 − base | **slope3 − base** | slope3 − fe228 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.000 | 202105 | 52,66 | 52,93 | 52,84 | +0,28 (5/10) | +0,18 (7/10) | −0,09 (5/10) |
| 9.000 | 202106 | 51,63 | 52,76 | 52,83 | +1,13 (7/10) | **+1,20 (7/10, p=0,10)** | +0,07 (4/10) |
| 9.000 | 202107 | 64,05 | 65,52 | 65,34 | +1,47 (9/10) | **+1,29 (10/10, p=0,002)** | −0,18 (3/10) |
| 10.500 | 202105 | 56,29 | 56,10 | 56,39 | −0,18 (3/10) | +0,10 (5/10) | +0,29 (6/10) |
| 10.500 | 202106 | 56,79 | 57,91 | 58,03 | +1,12 (8/10) | **+1,25 (8/10, p=0,008)** | +0,13 (7/10) |
| 10.500 | 202107 | 68,84 | 70,08 | 69,65 | +1,24 (10/10) | **+0,81 (10/10, p=0,002)** | −0,44 (2/10, p=0,06) |

**Lectura.** La pendiente rinde por la **tendencia**, no por el largo de la historia: con tres puntos
fijos conserva toda la ganancia del FE (slope3 ≈ fe228, diferencias de −0,44 a +0,29, ninguna
significativa) y pierde el defecto de construcción. Esto le saca piso a mi mecanismo del §2.2 como
explicación de «fe228 pierde en público»: la parte de la pendiente que cambiaba de significado no
era la que rendía, así que ese z = −1,0 queda como ruido o como algo que no identificamos.

**Dónde deja la decisión.** `fe_slope3` es ahora el único candidato con mecanismo y evidencia: tres
folds a favor (dos significativos a 10.500), 202 de sus 223 columnas iguales a `fe228`, y una
construcción que significa lo mismo en agosto que en los folds. Vale ~+1 punto de captura ≈ 11
fugados/mes ≈ 12 M/mes ≈ **3 M en público**: real y por debajo del piso de detección (sd 3 M), así
que **no se puede confirmar antes del domingo**. En contra: el único dato público sobre un FE casi
igual dio z = −1,0. A favor del base: no arrastra nada. Es una decisión de criterio, no de medición;
yo me inclino levemente por `fe_slope3` porque es la única configuración donde los tres folds y el
mecanismo apuntan al mismo lado, pero quedarse con el base es defendible y la diferencia esperada es
de ~3 clientes públicos. Si se cambia, cambiar entero (la mezcla base+FE ya se midió y no aporta).

Artefactos: `data/competencia_01_fe3.parquet` (regenerable con `scratchpad/sonda_slope3.py`, 3 s) y
los 30 modelos en `experimentos/helper_slope3/`. Cómputo: 6 min.

---

# Quinta ronda: qué entregaría

**Modelo.** `fe_slope3` **si** entra en `reproducir.py` (construcción del FE desde el crudo) y el CSV
reproducido coincide byte a byte con el del pipeline antes del viernes; si no, base. Valor esperado
+1 pt de captura ≈ +3 M público, indetectable; el costo que pesa es el operativo, con tres bugs de
alineación esta misma semana.

**Corte: 10.000.** Los cinco puntos públicos de 20 archivos, convertidos a tasa de acierto por banda
(un acierto público = +1,1 M, un no-acierto = −0,0275 M, 250 públicos por cada 1.000 de corte):

| banda | Δ público | aciertos públicos | tasa |
| --- | ---: | ---: | ---: |
| 8.000 → 9.000 | +4,0 M | ~10 en 250 | **4,0%** |
| 9.000 → 11.000 | −2,9 M | ~10 en 500 | 2,0% |
| 11.000 → 13.000 | −4,2 M | ~9 en 500 | 1,7% |
| 13.000 → 15.000 | −4,0 M | ~9 en 500 | 1,8% |

Equilibrio 2,5%. La cola 9.000–15.000 entera son ~28 aciertos en 1.500 públicos: **1,85% ± 0,35%**,
una banda de 6.000 clientes medida ~2 sd bajo el equilibrio, no «4 clientes». Es la forma de mayo
(1,8%), no la de junio (2,9%). El **nivel** público (142 aciertos en el top 9.000 → ~570 en el mes)
en cambio implica prevalencia tipo junio si la captura fuera la de los folds. Nivel de junio con
cola de mayo sólo cierra si la curva de agosto es **más empinada arriba** que las de los folds, lo
que coincide con el «agosto viene mejor que la validación» anotado el 3-oct. Conclusión: las curvas
de captura de mayo/junio no describen agosto, y el arrepentimiento de `c172` está calculado sobre
curvas que agosto ya mostró no seguir.

Entre 9.000 y 10.000: si la cola es 1,85%, 10.000 pierde ~7 M/mes; si fuera tipo junio, 9.000 pierde
~16 M/mes. Minimax: **10.000**. 10.500 es aceptable (−3,6 más en el escenario favorecido por el
público, +8 en el otro). 9.000 no, 11.000+ no (la banda 10–11k está medida bajo el equilibrio).

**El sesgo «la entrega entrena con 4 meses y captura más, así que su K\* es más alto» no es real.**
La grilla de la tercera ronda muestra que el modelo de 4 meses no captura más que el de 1 o 2 a la
misma distancia (−0,16 / −0,95 pts). El sesgo real va al revés: las curvas de los folds están
calibradas en mayo y junio y agosto no las sigue.

**Submit con valor de decisión, pre-registrado.** `fe_slope3`, 20 archivos de una semilla, corte
9.000, mismas 20 semillas de `c107`, pareado contra `c152_corte9000` (94,22) y `c153_fe228`
(91,11). No confirma (esperado +3, sd 3) pero **puede vetar**: ≤ 91 → base; ≥ 94 → `fe_slope3`;
entre medio decide la condición operativa. Opcional, el mismo modelo a 11.000 × 20 contra `c107`
(91,36). Ningún submit de corte entre 9k y 11k puede resolver 2,0% contra 2,5% con ~10 aciertos
por banda.

---

# Cierre (2026-10-07): base al corte 10.000

Los cuatro submits del día, 20 archivos cada uno, mismas semillas de `c107`:

| corte | base | `fe_slope3` | fe3 − base |
| ---: | ---: | ---: | ---: |
| 9.000 | 94,22 | 90,93 | −3,29 |
| 10.000 | **93,24** | 93,49 | +0,25 |
| 11.000 | 91,36 | 92,39 | +1,03 |

Media −0,67 con ±3 y las tres lecturas comparten casi toda la selección y la misma partición:
**no hay diferencia**. La regla pre-registrada (fe3@9.000 ≤ 91 → base) se disparó por 0,07 y se
honra, con la autocrítica de `competencia-1` anotada: el umbral tenía que ser un intervalo. Que
`fe228` (91,11) y `fe3` (90,93) coincidan a 9.000 es consistencia interna (202 de 223 columnas
iguales, mismas semillas, misma partición), no dos evidencias. Entre dos modelos indistinguibles,
el que ya reproduce byte a byte gana por defecto.

Curva pública completa del base: 8k 90,23 · 9k 94,22 · **10k 93,24** · 11k 91,36 · 13k 87,15 ·
15k 83,18. Tasa marginal 8–9k 4,0% · 9–10k 2,14% · 10–11k 1,82% (±0,9% por banda, equilibrio
2,5%). 10.000 sigue siendo el minimax. Quedaron 9 submits sin usar a propósito.

**Entrega: base 150 columnas · `pesos 0,25` · 202103–202106 · 20 semillas · ensamble por rango ·
corte 10.000.**
