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

## De dónde sale la receta — `J-Clase 06 > un detalle insoslayable` [189058–189080], 18-sep
- `z495_GustavoPasquini.ipynb` (adjunto en [187193]) optimiza solo `num_iterations`, `num_leaves` y
  `min_sum_hessian_in_leaf`, y hace **4 × 3 = 12 iteraciones no inteligentes y 0 inteligentes**:
  "simplemente se está quedando con la mejor de las 12". La receta de `En Limpio` no es un óptimo fino.
- Rangos de su bayesiana ("MUY generosos, no hay trampa"): `num_iterations` 64–4096, `num_leaves`
  16–2048, `min_sum_hessian_in_leaf` **1e-06–0,1 absoluto**, sobre ~13k filas (undersampling 0,1 en la
  búsqueda, métrica average_precision). Por fila es hasta ~7e-6: la zona donde `c220` encontró su
  óptimo y que después no transfirió (`c221` público, `c222` mayo).
- El `12,791817 × filas / 326.184` **no es de Denicolay**: 326.184 son las filas de 202103+202104, es la
  calibración de Ramírez. Nuestro barrido ×0,5/×2 (12,8 y 51 absolutos) pisa una región que ni la
  cátedra ni `c220` recorrieron.

## Entrega — confirmado por la tutora Silvana Contreras, `Tutorias > J_08` [197392], 8-oct 19:41
- "Para la competencia final **vale el último que hayas enviado** y debe ser una submission de **un
  solo archivo**." El bot lo repite en cada `list`: "The private gain is only known for your LAST
  pre-deadline submission". → el último submit antes del cierre tiene que ser la entrega, de un archivo.
- `z-Entrega Final > Primera Competencia` [191074], 29-sep: ahí se postea el link a GitHub con
  scripts/notebooks e instrucciones "de forma que la cátedra sea capaz de replicar en forma exacta su
  predicción final". **Al 8-oct 22:40 nadie posteó nada en ese tópico, nosotras tampoco.**
- `general` [191563], 2-oct: "ir cambiando de a un registro para subir en el Public" es "actividad
  ilícita". Nuestros submits de curva de corte van de a 500 y se registran como medición con hipótesis.
- `J-Clase 06 > Canaritos` [197252], 8-oct 18:18: **Public ≈ 25% / Private ≈ 75%** de 202108. "El único
  que cuenta para la aprobación y nota" es Private > 300. Al 8-oct 18:46 no hay leaderboard global
  publicado; el podio eran "tres alumnas, una del lunes y dos del jueves" y "va bajando día a día".

## La guía de la tutoría previa al cierre — Tutorias [193986], PDF en Drive
Es el cuestionario de la cátedra y sirve de índice para el README: variables rotas; drift por
inflación / dólar / UVA / ranking / ranking con cero fijo; imputación; FE intra e histórico; canaritos;
particiones y meses; undersampling sí/no; target {BAJA+1, BAJA+2}; métrica; cuántas iteraciones de BO,
qué entra y con qué bordes; feature importance; meses del final; ensembles; cómo deciden la cantidad
de envíos; plan para los días que quedan; "¿identificaron cosas que tienen que hacer sí o sí?".
Cada ítem tiene un experimento en el diario (c215 rank, c167 cero fijo, c127 deflactar, canaritos,
c213/c242/c243 target, c220/c222 BO, c245 ensambles, curva de corte).

## Cerrado con medición (sesión monday-3d, 8-oct): `*_fultimo_cierre` y `Visa_mpagado`
Laura Pasquini [192773] reportó ceros raros; Denicolay no respondió. Medido sobre `competencia_01.parquet`:
`*_fultimo_cierre` es reloj puro (moda por mes 1/2/5/0/3/6; el 0 de junio es el día modal, 113.299
ceros, BAJA+2 0,52% con cero vs 0,47% sin); `Visa_mpagado` en 05 y 08 pasa de ~141k ceros a ~114k, o sea
~27k clientes con pago real por ciclo. Nada que reparar.

## Cierre
- Zulip [189817]: domingo 11-oct 23:59 ART. `help` del bot (6-oct): `2026-10-11T03:01:01` (UTC → sábado
  10 23:59 ART). Una tutora el jueves 8: "a 3 días del cierre" (→ domingo). **Sin zanjar; la entrega
  va adentro el sábado.**
- Cupo del bot: 13/día hasta el 8-oct 16:11; desde entonces reporta `x/17`, sin anuncio en Zulip.

## Barrido del 10-oct a la mañana (mensajes desde el 9-oct 12:00): nada que mueva el modelo

- **Silvana Contreras, `general > Private vs Public` [198194, 198197]:** "ayer se recomendó intensamente que no saquen
  la conclusión de que les está yendo bien por la medida del leaderboard público"; la recomendación para los días
  que quedan es construir el baseline de `J-Clase 08 > En Limpio` (lo que ya es nuestra entrega) y optimizar
  **sin undersampling**. Propone comparar modelos "semilleados" con **Wilcoxon sobre vectores de ganancia** [198178]
  (es lo que hacemos con 20 archivos contra 20, mismas semillas).
- **Karina Formoso [198148]** (tabla): dos modelos con 50 semillas, corte 13.000: público 105,49 / 106,54 promediados,
  106,25 / 104,33 como 10 archivos de 5 semillas. Mismo orden de magnitud que el nuestro; su meseta local en junio
  va de 11.400 a 14.600 y la tutora le objeta que una meseta de 3.000 envíos "es un mesetón".
- **Florencia Rodriguez [198165]:** sus modelos > 100 son exactamente la receta (03–06, ambas bajas positivas,
  `ccajas_depositos` NA en mayo, lags 1 y 2 + deltas, 762 variables, préstamos adentro).
- **Emmanuel Valdez [198166, 198182]:** sugiere **rankear las variables entre −1 y 1** (rank con signo: positivos en
  (0, 1], negativos en [−1, 0), 0 → 0) e imputar nulos con el mes anterior o el promedio. Sin medición publicada.
  Nuestra medición previa de rank por mes (`c150_pormes`, dataset viejo) perdió −2; sobre lags12 no se midió.
- **German Reintgen [199199]:** curva de ganancia en junio (train marzo–abril), 20 semillas + ensamble por promedio
  de probabilidades: pico ~12.000 en test local, igual que el sesgo chico de nuestro fold C.
- Alejandro Bolaños arregla `list submit` del bot (sale truncado) [199714]. Nadie posteó todavía en
  `z-Entrega Final > Primera Competencia`.
