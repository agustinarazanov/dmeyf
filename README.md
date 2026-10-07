# DMEyF 2026 · Primera Competencia

Agustina Razanov · Data Mining en Economía y Finanzas (UBA), comisión de los lunes.

Objetivo: elegir a qué clientes de la foto `202108` mandar el estímulo de retención, maximizando
`+1.072.500` por cada `BAJA+2` acertado y `−27.500` por cada estimulado que no lo es. El bot de Zulip
`@competencia-uno` devuelve la ganancia pública (~25% de agosto) de cada archivo enviado.

> Estado al 2026-10-07 por la noche. La entrega final todavía no está fijada; este README se actualiza
> con cada decisión. El diario completo, con retractaciones, está en `docs/diario_competencia.md`.

## 1. El modelo vigente, en una tabla

| Pieza | Elección | Dónde se decidió |
| --- | --- | --- |
| Datos | `competencia_01_crudo.csv` → `clase_ternaria` con horizonte 2 (`fe_panel.construir_base`) | `scripts/c100` |
| Variables | las 150 crudas + **lag 1, delta 1, lag 2 y delta 2 de todas** (750 columnas); `ccajas_depositos` → NA en 202105 | `scripts/c200_lags_todo.py` |
| Fuera del modelo | `numero_de_cliente`, `foto_mes`, las columnas de baja, `Visa_Finiciomora`, `Master_Finiciomora` (ceros inyectados en 202108) | diario, "Leakage" |
| Clase | positivo = `BAJA+1` ∪ `BAJA+2`, con `BAJA+1` pesando 0,25 (`target='pesos'`) | `scripts/c102`, retractado en parte: vale al corte 11.000 |
| Hiperparámetros | receta de Denicolay (Zulip `J-Clase 08 > En Limpio`): `max_bin` 31, `min_data_in_leaf` 0, `feature_fraction` 0,5, `feature_fraction_bynode` 0,2, `learning_rate` 0,005, 83 hojas, 1.000 rondas, `min_sum_hessian_in_leaf` = 12,79 × filas / 326.184 | `scripts/c201_receta_lags.py` |
| Entrenamiento | 202103–202106, 654.066 filas, 8.067 positivos, sin undersampling | `scripts/c201` |
| Semillas | 261431, 269281, 429899, 560771, 749401 + 15 sorteadas con `np.random.seed(261431)` | `competencia.SEMILLAS` |
| Ensamble | promedio de **rangos** (no de probabilidades) de las semillas | `competencia.ensamble_por_rank` |
| Corte | **banda 13.500–14.500**, medido en público el 7-oct (ver §3) | `scripts/c201_cortar.py` |

## 2. Cómo reproducir

```bash
pyenv local facultad          # Python 3.14, duckdb 1.5.5, lightgbm 4.7.0, pandas 2.3.3
python reproducir.py --salida entrega_final.csv
```

`reproducir.py` parte del crudo de la cátedra en `data/` (symlink a `../dmeyf2026/monday/data`),
reconstruye la clase, ordena las filas por `(numero_de_cliente, foto_mes)` —DuckDB no garantiza el
orden y LightGBM depende de él—, entrena las 20 semillas y escribe el CSV ordenado por id. Imprime
el SHA-256 para compararlo con el entregado. **Pendiente**: pasarlo al dataset de lags y a la receta
(hoy todavía reproduce la entrega anterior, la base de 150 columnas a 10.000).

Para medir sin reentrenar: los modelos quedan en `experimentos/<exp>/modelo_<semilla>.txt` y los
scores de 202108 en `scores_202108_s<semilla>.parquet`; `scripts/c201_cortar.py` re-corta en segundos.

## 3. Lo que movió el público

Toda comparación se hace con 20 archivos de una semilla cada uno (el bot devuelve la media), mismas
semillas y mismo corte que la referencia. Un cliente público vale ~1,1 M; el ruido de partición entre
dos modelos es ±3 M y entre cortes anidados del mismo ranking ±4 M.

| cambio | público | contra qué | lectura |
| --- | ---: | --- | --- |
| línea de base: 150 col, params de `z701`, `BAJA+2` solo, 3 semillas, 11.000 | 89,02 | — | el esqueleto funciona |
| `BAJA+1` pesando 0,25, 20 semillas, 11.000 | 91,36 | 89,02 | dentro del ruido; la evidencia es de los folds (8/10, p=0,008) y sólo vale a 11.000 |
| corte 9.000 de ese ranking | 94,22 | 91,36 | la base pica en 9.000 y cae 11 M hasta 15.000 |
| FE de `z402` (historia de 21 variables, 228 col) | 91,11 | 94,22 | z = −1, no concluyente |
| sacar préstamos, deflactar, FE histórico de 80 col (los tres juntos) | 87,62 | 92,62 | pierde; el profesor lo confirmó: no sacar variables para 202108 |
| **lags/deltas 1 y 2 de las 150 + receta de Denicolay**, 5 semillas, 10.000 | **100,25** | 93,24 | **+7 M al mismo corte, z ≈ 2,3** |
| el mismo ranking a 14.000 | **104,16** | 87 (base a 13.000) | el óptimo se corrió a la derecha |

La curva del modelo vigente, 5 archivos por punto:

| corte | 9.000 | 10.000 | 11.500 | 12.500 | 13.000 | 13.500 | **14.000** | 14.500 | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| público | 98,6 | 100,2 | 100,4 | 102,1 | 102,0 | 103,2 | **104,2** | 103,8 | 102,7 |

## 4. Lo que no funcionó, y por qué queda documentado

- **Búsqueda bayesiana con CV dentro del mes**: ganaba en CV y perdía fuera de tiempo. Causa: los
  `BAJA+2` de marzo son `BAJA+1` de abril, y el split por filas los ponía a ambos lados (leakage de gemelos).
- **Cadena de validación a horizonte 1** (`c144`–`c147`, `c166`): 99,5% de los positivos de validación
  ya estaban en el entrenamiento. Inválida. Los folds limpios son A `[03]→05`, B `[03,04]→06`, C `[03,04,05]→07`.
- **Sacar marzo, decaimiento por antigüedad, entrenar sólo con el mes reciente**: atacaban "los meses
  viejos son malos"; lo que pesa es la distancia al mes a predecir, y 202106 ya está a dos meses de agosto.
- **Poda con canaritos** en cinco niveles: nunca ayudó; 46 de 73 columnas nuevas le ganan a todo el ruido.
- **Validación adversaria**: AUC 0,9998 por dos relojes de calendario (`*_fultimo_cierre`); sacarlos no
  cambió nada. Separable no implica perjudicial.
- **El instrumento**: durante una semana se mandó un archivo por submit cuando se podían mandar
  veinte; después se usó el desvío entre semillas como si fuera el ruido de la partición. Las dos cosas
  dejaron las comparaciones debajo del piso de ruido y se retractaron.
- **Los folds no ven los lags**: marzo no tiene mes anterior y lag 2 está vacío en todo el
  entrenamiento de los folds A y B. Por eso "el FE pierde" se sostuvo tanto tiempo: lo medimos donde
  no podía ganar.

## 5. Trazabilidad

Cada experimento, submit, archivo y respuesta del bot está en el ledger de Postgres (`registro.py`) y
exportado a `ledger/*.csv`. Un submit se registra con hipótesis y delta **antes** de enviarse
(`enviar.py`), y la respuesta cruda del bot se guarda al volver. `ledger/v_submits.csv` es el
leaderboard propio completo.

## 6. Fuentes

- Zulip de la materia (archivado en `../zulip/`): `J-Clase 08 > En Limpio` (receta), `J-Clase 08 >
  venciendo Primero Competencia` (préstamos), `J-Clase 06 > Entrenar en CUATRO meses` (pseudo-etiquetado).
- Repos públicos de compañeros con las mismas reglas: `ezequielabregu/DMEyF_competencia_1` (110,25) y
  `farpp8/dmeyf2026-competencia01` (103,18). De ahí salió la pista de los lags de todo.
- Libro de la Asignatura, p. 11: criterios de evaluación, Línea de Muerte, originalidad y reproducibilidad.
