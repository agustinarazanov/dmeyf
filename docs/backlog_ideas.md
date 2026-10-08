# Backlog de ideas para mejorar la predicción

Cada ítem lleva fundamento, costo y estado. Las que se prueban pasan al diario con su medición; las que
se descartan quedan acá con el motivo, para no repetirlas. Regla: un submit mide una hipótesis con
fundamento; lo que no tiene fundamento no se manda.

## A. Las seis propuestas del 7-oct (de los repos de compañeros y la receta de la cátedra)

| # | Idea | Fundamento | Estado |
| --- | --- | --- | --- |
| 1 | Lag 1 / delta 1 / lag 2 / delta 2 de **todas** las variables | Ramírez 86,7 → 98,6 público; Denicolay "superador" | **Hecho sobre valores crudos** (`c200`, `c201`): 100–104 público. **Pendiente la variante sobre montos rankeados por mes**, para que el lag no arrastre inflación (Abregu la usa) |
| 2 | Hiperparámetros para 750 columnas: receta de Denicolay (`min_data_in_leaf` 0, `lr` 0,005, `ff` 0,5, `ff_bynode` 0,2, 83 hojas, 1.000 rondas) y después Optuna sobre `num_iterations`, `num_leaves`, `min_sum_hessian` | Ramírez +4,6 encima de los lags; Abregu +6,8 del semillerío con `ff` 0,40 | **Receta hecha** (`c201`). **Pendiente** Optuna (`c180` sobre el dataset nuevo, arreglando antes el bug del slice de folds) con early stopping por AUC en el fold C |
| 3 | Re-barrer el corte para el modelo nuevo | un ranking mejor captura más profundo y corre el óptimo a la derecha | **Hecho**: meseta 13.500–14.500, pico 14.000 (5 semillas). Repetir con 20 archivos y con el ensamble |
| 4 | Stacking en vez de promedio: meta-modelo chico sobre los scores de varios preprocesamientos (base, lags crudos, lags rankeados, deflactado) | experimento colaborativo 2025: stackear tres preprocesamientos +14,2 M, promediarlos 0; nosotros vimos base+FE con correlación 0,99 y promedio sin efecto | Pendiente. Entrenar el meta-modelo con los scores del fold B (202106) y aplicarlo a los de 202108 |
| 5 | Reparaciones de datos: `ccajas_depositos + ccajas_otras` (reclasificación abr–jun); sacar `*_mconsumototal` (duplica `mconsumospesos`); NA en centinelas de `*_Fvencimiento` (< −1.000.000); NA en `mfinanciacion_limite` > 10× `mlimitecompra`; `ccajas_depositos` NA en 202105 | Ramírez D01, Denicolay (`En Limpio` 1.1) | Solo `ccajas_depositos` NA hecho. El resto pendiente, barato, un dataset `c202` |
| 6 | 202107 con corrección positive-unlabeled: BAJA+1 conocidos como positivos, y **sacar** los negativos que el base puntúa alto (los ~870 BAJA+2 escondidos) | `c168` perdió 33,7 M al etiquetarlos negativos; la ley de envejecimiento dice que un mes de distancia vale puntos de captura; Aramendía en Zulip: "los BAJA+1 de 202107 sí sirven" | Pendiente. Validación limpia: train [03,04] + [05 con BAJA+1 y negativos podados] → val 202106 BAJA+2 |

## B0. Desacoplar FE de hiperparámetros — HECHO 7-oct 23:18: lags +3,2, receta +3,8, aditivos
`lags12 + params z701` y `base + receta Denicolay`, 5 semillas cada uno, corte 10.000 y 14.000, contra
`c201` (100,25 / 104,16) y `c191` (93,24). Sin esto el +7 no está atribuido (mismo error que `c133_todo`).

## B. Ideas nuevas, por orden de expectativa × costo

### B1. Target: tres variantes sobre el ranking nuevo — MEDIDO 8-oct: BAJA+2 puro −5,3 a 14k. Refutado
`BAJA+2` solo (K1), `BAJA+1 ∪ BAJA+2` (K2) y `pesos` 0,25 (lo actual). Ramírez midió en público K1 91,8
contra K2 86,7 **con el mismo modelo**, y nosotros adoptamos `pesos` con evidencia que solo valía a
11.000. Con el modelo nuevo nunca se midió. Costo: dos entrenamientos de 5 semillas (~1 h), dos submits.

### B2. Meses de entrenamiento con lags — MEDIDO 8-oct: sin marzo −3,2 a 14k. Refutado
Con lags, marzo tiene **todas** las columnas de historia nulas y abril no tiene lag 2: el patrón de
nulos vuelve a codificar el mes (ya se midió que explicaba ~6 de 47 M en el FE viejo). "Sacar marzo"
perdió para el base por la ley de distancia, pero el trade-off cambia cuando marzo aporta 1/4 de las
filas y 0 de la historia. Costo: un entrenamiento de 5 semillas, un submit.

### B3. Ensamble heterogéneo — EN CURSO (`c221`, top 5 × 4 semillas). OJO: `c222` mostró que el top 5 sobreajusta a jun/jul (5/5 debajo de la receta en mayo)
Denicolay pide "10+ modelos con hiperparámetros distintos" promediados. El semillerío de 20 iguales
cancela ruido de orden; modelos distintos (otro `num_leaves`, otro `ff`, otro dataset) cancelan sesgos.
Salida natural del Optuna de A2: promediar los rangos del top-5 de trials. Abregu: 40 semillas no
mejoran sobre 20, o sea el ruido de semilla ya está agotado.

### B4. Estimar la prevalencia de agosto con un modelo calibrado
El corte depende de cuántos BAJA+2 hay en agosto (870–1.139 en los meses vistos) y lo venimos
infiriendo del barrido público. Un modelo con target `BAJA+2` puro y `objective binary` da
probabilidades; la suma de `p` sobre 202108 estima los positivos de agosto. Validar: la misma suma
sobre 202106 (fold B) contra los 1.098 reales. Si calibra razonablemente, es un segundo instrumento
para el corte, independiente del público. Costo: casi cero con un modelo K1 de B1.

### B5. Features de transición y de recencia — EN DATASET (`c230`, 800 col: 36 transiciones, 4 recencias, reparaciones de Ramírez); se entrena después de `c221`
Indicadores "pasó a cero este mes" y "pasó a negativo" para las variables de monto, y "meses desde
el último sueldo / desde la última transacción / desde que entró en rojo". Clara Rodríguez midió en
Zulip transiciones a cero en `mcaja_ahorro` (32% → 41% en la cohorte BAJA+2) y `mpasivos_margen`.
Son binarias, así que no sufren inflación ni drift de escala, y el árbol no las reconstruye bien desde
lags continuos con `max_bin` 31. Costo: SQL en `fe_panel`, un dataset, un submit.

### B6. Lags sobre rangos por mes — MEDIDO 8-oct: −1,0 a 14k, neutro. No se adopta
`percent_rank` dentro de `foto_mes` de los 73 montos **antes** de calcular lag y delta. Un delta de
pesos entre junio (aguinaldo) y julio mide calendario, no conducta; un delta de percentil mide conducta.
Abregu lo hace (`l1_rk0_`, `d1_rk0_`). Costo: `c202`, un entrenamiento, un submit.

### B7. Bloque de nulos simultáneos — NO EXISTE en nuestro parquet (8-oct): los nulos vienen en 4 bloques estructurales (99.644 / 47.186 / 579.470 / 127.760 filas); el flag quedó constante en 0 y es inerte
35.301 filas (3,5%) tienen todo el bloque de variables de negocio en nulo a la vez (Zulip, Alexander
Arias). Para esas filas los lags y deltas salen nulos o espurios. Un flag `bloque_nulo` y su lag
separan "el dato falta" de "el valor cambió". Costo: trivial; va junto con A5.

### B8. Regularizadores que mejoran la transferencia entre meses
`extra_trees=True`, `path_smooth`, `monotone_constraints` en las variables de dirección conocida
(`ctrx_quarter` baja el riesgo, `mcuentas_saldo` baja el riesgo) y `max_bin_by_feature` más grueso en
las columnas que `c128` marcó con AUC móvil. Fundamento: el modelo pierde ~50 M por drift entre meses
(`diagnostico_fe`), y estos cuatro reducen la dependencia de umbrales finos aprendidos en un mes. Costo:
entran como dimensiones del Optuna de A2, no hacen falta corridas aparte.

### B9. DART
German Reintgen (Zulip) lo reportó como su mejor ganancia y menor sobreajuste, "lentísimo". Con 6
minutos por modelo GBDT, DART de 1.000 rondas es de madrugada. Solo si sobra una noche de máquina.

### B10. Pseudo-etiquetado de 202107 y 202108 (versión del profesor)
`J-Clase 06 > Entrenar en CUATRO meses`: entrenar en los meses con clase, pintar una clase artificial en
los meses sin clase con un corte de probabilidad, reentrenar con todos. Más agresivo que A6 (A6 solo
poda negativos dudosos; esto inventa positivos). Riesgo de circularidad; el propio profesor lo plantea
como "bomba lógica" para probar. Costo: dos entrenamientos. Después de A6.

### B11. Submit de 16.000 filas
El `help` del bot no menciona tope; los 8.000–15.000 son del mensaje del profesor. Si el bot acepta,
un punto más de la curva (que ya baja en 15.000, así que la expectativa es confirmar el límite, no
ganar). Hay que saltear `MAX_ENVIOS` y `MAX_FILAS` a propósito; lo corre la usuaria.

## C. Descartado con medición (no volver a probar sin un motivo nuevo)

- Sacar variables con drift para 202108 (préstamos incluidos): el profesor, `c133` en público, Abregu, Ramírez.
- Deflactar por mediana mensual, `rank_cero_fijo`, FE histórico de 80 columnas sobre 21 variables.
- Decaimiento por antigüedad, entrenar solo con el mes reciente, sacar marzo **para el base**.
- Poda con canaritos en cinco niveles; también a una alumna de los jueves le bajó la ganancia.
- Arreglar `Visa_delinquency` en 202105/202108: 0 efecto; sacar a los inyectados cuesta −8,7 M.
- Sacar los relojes de calendario `*_fultimo_cierre`: separables pero inertes.
- Demoter de horizonte 1 ingenuo, dos etapas, modelo por mes, `scale_pos_weight`: firma mayo/junio o nada.
- Más rondas con árboles complejos (45 hojas, `min_data_in_leaf` 174): sobreajusta.
- Lo que el profesor lista como inútil: PCA, imputar, outliers, AutoML, desbalanceo, features de clustering.

## D. Deuda técnica abierta (del code review del 7-oct noche, sesión competencia-1-05)
- `reproducir.py` reproduce la entrega vieja (base 150 col, 10.000) y no fija `deterministic`/`force_col_wise`/
  `num_threads` como `c201`. Se reescribe entero cuando se fije la entrega final.
- `data/competencia_01_lags12.parquet` del 7-oct tiene los lags de `ccajas_depositos` sobre los ceros crudos de
  202105 (una columna de 750). `c200` ya está corregido y se niega a pisar el archivo sin `--forzar`: el
  desacople (B0) se corre sobre el archivo viejo para que el dataset sea idéntico al de `c201`; recién después
  se regenera y se reentrena lo que quede vigente.
