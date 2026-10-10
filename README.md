# DMEyF 2026 · Primera Competencia

Agustina Razanov · Data Mining en Economía y Finanzas (UBA), comisión de los lunes.

Objetivo: elegir a qué clientes de la foto `202108` mandar el estímulo de retención, maximizando
`+1.072.500` por cada `BAJA+2` acertado y `−27.500` por cada estimulado que no lo es. El bot de Zulip
`@competencia-uno` devuelve la ganancia pública (~25% de agosto) de cada archivo enviado.

> Estado al 2026-10-10 por la tarde. **La entrega es `c241_rondas2000_ens20_14000`** (sección 1), reproducida
> bit a bit con `reproducir.py`. El diario completo, con retractaciones, está en `docs/diario_competencia.md`;
> los gráficos, en `notebooks/`.

## 1. La entrega, en una tabla

| Pieza | Elección | Dónde |
| --- | --- | --- |
| Datos | `competencia_01_crudo.csv` → `clase_ternaria` con horizonte 2 (`fe_panel.construir_base`) | `scripts/c100` |
| Variables | las 150 crudas + **lag 1, delta 1, lag 2 y delta 2 de todas** (750 columnas); `ccajas_depositos` → NA en 202105 | `scripts/c200_lags_todo.py` |
| Fuera del modelo | `numero_de_cliente`, `foto_mes`, las columnas de baja, `Visa_Finiciomora`, `Master_Finiciomora` (ceros inyectados en 202105 y 202108) | `src/competencia.py` |
| Clase | positivo = `BAJA+1` ∪ `BAJA+2`, con `BAJA+1` pesando 0,25 (`target='pesos'`); pesos 0,5 y 1 perdieron (−2,0 / −1,4), BAJA+2 solo perdió −5,3 | `scripts/c242`, `c243`, `c213` |
| Hiperparámetros | receta de Denicolay (Zulip `J-Clase 08 > En Limpio`): `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5, `feature_fraction_bynode` 0,2, `learning_rate` 0,005, 83 hojas, `min_sum_hessian_in_leaf` = 12,79 × filas / 326.184, **2.000 rondas** | `scripts/c201_receta_lags.py`, `c241` |
| Entrenamiento | 202103–202106, 654.066 filas, 8.067 positivos, sin undersampling | `scripts/c210_variante.py` |
| Semillas | 261431, 269281, 429899, 560771, 749401 + 15 sorteadas con `np.random.seed(261431)` | `src/competencia.py` |
| Ensamble | promedio de **rangos** (no de probabilidades) de las 20 semillas | `competencia.ensamble_por_rank` |
| Corte | **14.000**: pico de la curva pública con 20 archivos por punto, y minimax del regret local a horizonte 2 en 13.500 (sección 3) | `scripts/c216_captura_foldB.py` |
| Archivo entregado | `experimentos/c241_rondas2000/envios_14000/c241_rondas2000_14000_ens20.csv` (en el ledger, submit `c241_rondas2000_ens20_14000`), un solo archivo, SHA-256 `657d747addc61efad64145ec9cef550c29c50c5ce283745f176d243270bb9d2e`, 103,79 público; reenviado como último submit el 10-oct 19:05 | `reproducir.py` |

## 2. Cómo reproducir

```bash
pyenv local facultad          # Python 3.14, duckdb 1.5.5, lightgbm 4.7.0, pandas 2.3.3, numpy 2.5.3
python reproducir.py --rondas 2000 --salida entrega_final.csv      # ~4 h en 8 cores
```

`reproducir.py` parte del crudo de la cátedra en `data/` (symlink a `../dmeyf2026/monday/data`),
reconstruye la clase, construye la historia (lag 1, delta 1, lag 2, delta 2 de las 150 variables),
ordena las filas por `(numero_de_cliente, foto_mes)` —DuckDB no garantiza el orden y LightGBM depende
de él—, entrena las 20 semillas con la receta, ensambla por rango y escribe los 14.000 ids ordenados.
Imprime el SHA-256 y lo compara con el del archivo entregado: **el 10-oct dio `657d747a…`, idéntico**.

Dos detalles que decidieron el bit a bit: `deterministic=True`, `force_col_wise=True` y 8 hilos; y **un
`lgb.Dataset` nuevo por semilla**. LightGBM muestrea 200.000 filas con la semilla para construir los bins, así
que un `Dataset` compartido entre semillas entrena otros modelos (la primera corrida difirió en 157 de
14.000 ids por eso). `scripts/c280_verificar_semilla.py` entrena una semilla y la compara con la registrada.

Para medir sin reentrenar: los modelos quedan en `experimentos/<exp>/modelo_<semilla>.txt` y los
scores de 202108 en `scores_202108_s<semilla>.parquet`; `scripts/c201_cortar.py` re-corta en segundos.

## 3. Cómo se midió

El público es ~25% de agosto. Entre semillas de un mismo modelo el desvío por archivo es **1,8–2,5**, así que
la media de 5 archivos tiene sd ≈ 0,9 y la de 20, ≈ 0,45. Reglas que valen para todo lo que sigue:

- **La referencia honesta es la media de 20 archivos (103,85)**, no el mejor submit. Las primeras 5 semillas
  salieron con suerte en los dos modelos de referencia (+1,2 y +1,7 sobre su media de 20): la "maldición del
  ganador" medida en casa.
- Una variante medida con 5 archivos se compara contra la franja 103,85–105,51 (media de 20 / mismas 5 semillas
  de la receta). Lo que cae adentro no se distingue de la receta. **Se adopta solo con 20 contra 20.**
- Cada submit se registra con su hipótesis y contra qué se compara antes de enviarse (`ledger/v_submits.csv`).

### Lo que movió el público

| cambio | público | contra | lectura |
| --- | ---: | --- | --- |
| línea de base: 150 col, params de `z701`, `BAJA+2` solo, 11.000 | 89,02 | — | el esqueleto funciona |
| `BAJA+1` pesando 0,25, 20 semillas, 11.000 | 91,36 | 89,02 | dentro del ruido; la evidencia es de los folds |
| corte 9.000 de ese ranking | 94,22 | 91,36 | la base pica en 9.000 y cae 11 M hasta 15.000 |
| FE propio (`z402`, 228 col) / FE histórico de 80 col + sacar préstamos + deflactar | 91,11 / 87,62 | 94,22 / 92,62 | pierden; el profesor lo confirmó: no sacar variables para 202108 |
| **lags/deltas 1 y 2 de las 150 + receta de Denicolay**, 20 semillas, 10.000 | **100,03** | 93,24 | **+6,8 pareado (mismas semillas, corte y partición)** |
| desacople: lags con params viejos / base con la receta, 10.000 | 96,43 / 97,06 | 93,24 | +3,2 y +3,8, aditivos |
| el mismo ranking a 14.000, 20 semillas | 103,00 | 100,03 | el óptimo se corrió a la derecha |
| **2.000 rondas en vez de 1.000**, 20 semillas, 14.000 | **103,85** | 103,00 | +0,85, y +0,85 a +1,56 en los cinco cortes medidos (tabla de abajo) |
| ensamble por rango de las 20 semillas, un archivo | 103,79 | 103,85 | el ensamble reduce varianza, no "gana": **la entrega** |

### La curva de corte, 20 archivos por punto (sd ≈ 0,45)

| corte | 10.000 | 12.000 | 13.000 | 13.500 | **14.000** | 14.500 | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 rondas (`c201`) | 100,03 | — | 101,74 | 102,69 | **103,00** | 102,50 | 101,32 |
| 2.000 rondas (`c241`) | — | 101,12 | 103,30 | 103,56 | **103,85** | 103,37 | 102,81 |

El pico es 14.000 para los dos. Lo que el público no puede decir: entre 12.000 y 14.000 el salto son ~15 bajas en
500 clientes públicos (3,1% de precisión contra 2,5% de equilibrio), con desvío ±4 que ninguna semilla achica.
Fuera del público, la captura de junio a horizonte 2 (`c216`, fold B) da óptimos entre 13.500 y 14.000 según
la prevalencia de agosto, con el minimax del regret en 13.500 (`notebooks/n02_experimentos.ipynb`, sección 8).

### Las variantes sobre la receta, 5 archivos cada una, a 14.000, contra la franja 103,85–105,51

| familia | variante | público | lectura |
| --- | --- | ---: | --- |
| hiperparámetros | 3.000 rondas · 127 hojas · ff 0,4 · ff 0,3 | 103,80 · 104,12 · 104,26 · 103,89 | dentro de la franja |
| hiperparámetros | hessiano × 0,5 · × 2 | 102,23 · 101,49 | pierden |
| target y meses | pseudo-etiquetas de julio · sin marzo · BAJA+2 solo | 103,21 · 100,95 · 98,82 | neutro · pierde · pierde |
| datos y drift | comisión deflactada · aguinaldo · calendario · lags de ranks | 104,56 · 104,52 · 103,49 · 103,16 | dentro de la franja |
| columnas nuevas | 11 proporciones (815 col) · sumas, conteos y rachas (848 col) | 104,62 · 103,27 | dentro de la franja aunque se llevan 20–29% del gain |
| menos columnas | sin las 52 nunca usadas (698) · solo las 392 del 99% del gain | 102,71 · 102,05 | pierden en las dos intensidades |
| segundo modelo | horizonte 1 (BAJA+1, hasta julio) mezclado por rango, pesos 0,15 / 0,3 / 0,5 / 0,7 / 1 | 105,70 / 105,75 / 105,67 / 103,65 / 99,92 | plano hasta 0,5, se hunde después |
| sacar las importantes | sin la familia `ctrx_quarter` (745) · sin las 10 familias top (700) · sin una de cada par con \|r\| ≥ 0,95 (708) | 102,71 · 87,21 · 104,12 | pierde · se derrumba · dentro de la franja |

Las tres últimas (`c290`, `c291`, `c292`, 10-oct) son la prueba de la clase de ensambles llevada a LightGBM: en el
random forest sacar `ctrx_quarter` subía la ganancia; acá con muestreo de columnas la descorrelación ya está y lo
que queda es pérdida de información. Sin las 10 familias que concentran el 45% del gain el modelo vuelve al
nivel de la primera semana: la señal no se recupera desde las otras 140. Sacar las 42 casi duplicadas no cambia nada.

### Validación interna

- **Fold A (`c222`)**: la búsqueda bayesiana (AUC en junio y julio) eligió hiperparámetros que en mayo, mes que no
  vio, quedan debajo de la receta. Con ~1.100 positivos por mes el error muestral del AUC es del tamaño de la mejora.
- **Fold B (`c216`)**: captura a horizonte 2 → curva de ganancia esperada por prevalencia y regret por corte.
- **Local vs público (primera semana)**: seis experimentos con las dos medidas, Spearman 0,26. La validación en un
  mes no distinguía variantes; los lags no se podían ver localmente porque marzo no tiene mes anterior.

## 4. Lo que no funcionó, y por qué queda documentado

- **Búsqueda bayesiana con CV dentro del mes**: ganaba en CV y perdía fuera de tiempo. Causa: los
  `BAJA+2` de marzo son `BAJA+1` de abril, y el split por filas los ponía a ambos lados (leakage de gemelos).
- **Cadena de validación a horizonte 1** (`c144`–`c147`, `c166`): 99,5% de los positivos de validación
  ya estaban en el entrenamiento. Inválida. Los folds limpios son A `[03]→05`, B `[03,04]→06`, C `[03,04,05]→07`.
- **Sacar marzo, decaimiento por antigüedad, entrenar sólo con el mes reciente**: atacaban "los meses
  viejos son malos"; lo que pesa es la distancia al mes a predecir, y 202106 ya está a dos meses de agosto.
- **Reducción de dimensionalidad**: poda con canaritos en cinco niveles (nunca ayudó), sacar las 52 columnas que
  ningún árbol usa (−1,1) y quedarse con las 392 que juntan el 99% del gain (−1,8). Las columnas "muertas" sirven
  de ruido útil en el muestreo de columnas de cada árbol, o la importancia en train no es la que vale en agosto.
- **Validación adversaria**: AUC 0,9998 por dos relojes de calendario (`*_fultimo_cierre`); sacarlos no
  cambió nada. Separable no implica perjudicial.
- **El instrumento**: durante una semana se mandó un archivo por submit cuando se podían mandar veinte; después
  se usó el desvío entre semillas como si fuera el ruido de la partición; y las primeras 5 semillas se tomaron
  como referencia cuando estaban +1,7 sobre la media de 20. Las tres cosas dejaron comparaciones debajo del
  piso de ruido y se retractaron.
- **Los folds no ven los lags**: marzo no tiene mes anterior y lag 2 está vacío en todo el
  entrenamiento de los folds A y B. Por eso "el FE pierde" se sostuvo tanto tiempo: lo medimos donde
  no podía ganar.

## 5. Trazabilidad

Cada experimento, submit, archivo y respuesta del bot está en el ledger de Postgres (`src/registro.py`) y
exportado a `ledger/*.csv`. Un submit se registra con hipótesis y delta **antes** de enviarse
(`scripts/enviar.py`), y la respuesta cruda del bot se guarda al volver. `ledger/v_submits.csv` es el
leaderboard propio completo; `enviado_en` es la hora del mensaje al bot en Zulip. Los CSV enviados están en
`experimentos/`, con su SHA-256 en el ledger.

`notebooks/n01_eda.ipynb` es el dataset mirado antes de modelar; `notebooks/n02_experimentos.ipynb` es todo lo
de arriba en gráficos, generado desde el ledger y los scores cacheados.

## 6. Fuentes

- Zulip de la materia (archivado en `../zulip/`): `J-Clase 08 > En Limpio` (receta), `J-Clase 08 >
  venciendo Primero Competencia` (préstamos), `Cartelera > Entrega a Primera Competencia` (reglas de entrega),
  `general > Private vs Public` (tabla público/privado del 8-oct). Resumen con ids en `docs/pistas_profesor.md`.
- Repos públicos de compañeros con las mismas reglas: `ezequielabregu/DMEyF_competencia_1` (110,25) y
  `farpp8/dmeyf2026-competencia01` (103,18). De ahí salió la pista de los lags de todo.
- Libro de la Asignatura, §1.5.1: reproducibilidad exacta desde el dataset original, semillas incluidas.
