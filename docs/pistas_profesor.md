# Pistas del profesor (Gustavo Denicolay) para la Primera Competencia

Recopiladas de Zulip (archivo en `../zulip/`, ids entre corchetes). Son las fuentes de autoridad del
curso; cuando una medición nuestra las contradice, se mide de nuevo antes de descartarlas.

## La receta oficial — `J-Clase 08 > En Limpio` [193837–193846], 6-oct
- Data quality: `ccajas_depositos` → NA en 202105 (todos sus valores son cero).
- FE histórico: "es superador agregar lags y delta lags de orden 1 y 2".
- Training: sin undersampling en la búsqueda ("son muy pocos meses"); métrica AUC.
- Fijos: `max_bin` 31, `min_data_in_leaf` 0 ("muy importante"), `feature_fraction` 0,50,
  `feature_fraction_bynode` 0,20, `learning_rate` 0,005 ("muy importante").
- Bayesianos: `num_iterations`, `num_leaves`, `min_sum_hessian_in_leaf`.
- Entrega: varios modelos (10+) con hiperparámetros distintos, entrenados en 202103–202106, sin
  undersampling, promediando probabilidades por cliente. Probar cortes 10.000 / 10.500 / 11.000.
- FE hojas de Random Forest y canaritos asesinos: "tbd" (nunca los completó para esta competencia).

## Drift y variables — `J-Clase 08 > venciendo Primero Competencia` [192777, 192801, 192811], 5-oct
- "¿Vos me has visto escribir que para predecir 202108 hay que eliminar alguna variable?" — **no**.
- Sobre sacar los préstamos (la "salsa mágica" que ganaba en junio): "hice los experimentos y
  funciona peor". Siempre que cambia el mes a predecir hay que rehacer los experimentos.
- El `help` del bot no fija tope de filas; los 8.000–15.000 son del mensaje [191492].

## La línea de muerte — Tutorias [192556] y Libro p. 11
- "Para entender en dónde se encuentran parados contra el Public Leaderboard de 202108, si se caen
  del precipicio de los 100k o no."
- Libro: superar la Línea de Muerte da un 5; lo de arriba se mapea a [5, 10]. El valor se publica
  antes del cierre, el script nunca.

## Public vs Private — `general > Private vs Public` [197324], 8-oct 18:56
Sobre todos los submits del curso con 10 o más archivos (≈7.300):

| público | private < 300 | private > 300 | % > 300 |
| --- | ---: | ---: | ---: |
| < 95 | 2.659 | 25 | 1% |
| 95–100 | 1.706 | 115 | 6% |
| 100–105 | 1.829 | 327 | 15% |
| > 105 | 349 | 257 | 42% |

`mean(Public/Private) = 2,98`. "Bastante bien si se considera que los submits sucesivos van
overfiteando el Public Leaderboard, ya que la gente solo vuelve a subir cosas parecidas a las que
dieron bien en lo que puede ver." → el umbral que mira en private es **300 M** (≈ 100 público × 3).

## Canaritos y foco — `J-Clase 06 > Canaritos y selección de variables` [196876], 8-oct 14:53
- "El uso de canaritos para selección de variables es útil recién para la Segunda Competencia."
- "La selección de variables NO la hacemos para mejorar el poder predictivo del modelo" (es para
  hacer lugar en memoria). "Ahora que estás entrenando en 4 meses para la predicción final, no te
  hace falta hacer espacio."
- "Conclusión 1: no te compliques la vida al pedo en esta Primera Competencia. Conclusión 2:
  focalizate en cosas que SÍ incrementen la ganancia, y en no estar teniendo en algún lado un
  'moquito' que haga que estés por abajo de 100M en el Public Leaderboard."

## Pseudo-etiquetado — `J-Clase 06 > Entrenar en CUATRO meses` [190539–190562], 25-sep
Entrenar en los meses con clase, predecir una clase artificial `{BAJA+1, BAJA+2}` en los meses sin
clase con un corte de probabilidad, reentrenar con todos. Presentado como "bomba lógica" para que
los agentes "muerdan el polvo": él mismo lo pone a prueba, no lo afirma.

## Lo que lista como inútil — `off-topic > Instructional Scaffolding` [184534], 21-ago
Deep learning, PCA, imputación de nulos, tratamiento de outliers, AutoML, técnicas para el
desbalanceo, variables derivadas de clustering. "Los mejores modelos se hacen sin reducir la
dimensionalidad y sin imputar nulos" [183214]. `scale_pos_weight`: "ni en pedo" [187068].

## Semillas y overfitting — Cartelera [191495], 2-oct
"Generan varios modelos, cada uno solamente cambiando el hiperparámetro seed... esos múltiples
archivos (hasta 20) son los que entregan en un submit, y el bot les devuelve la media y la varianza,
con lo cual reducen enormemente el overfitting (técnicamente NO lo eliminan por completo)."

## Cierre
- Zulip [189817]: domingo 11-oct 23:59 ART. `help` del bot (6-oct): `2026-10-11T03:01:01` (UTC → sábado
  10 23:59 ART). Una tutora el jueves 8: "a 3 días del cierre" (→ domingo). **Sin zanjar; la entrega
  va adentro el sábado.**
- Cupo del bot: 13/día hasta el 8-oct 16:11; desde entonces reporta `x/17`, sin anuncio en Zulip.
