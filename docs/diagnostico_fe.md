# Diagnóstico: por qué tres rondas de FE no movieron la ganancia

Fecha: 2026-10-04. Sin features nuevas: sólo se midieron los modelos ya cacheados de `c116_v2`
(base / v101_viejo / v2_corregido × 10 semillas × 2 folds) y unos pocos modelos chicos de sonda.
Scripts en el scratchpad de la sesión (`d1`…`d6`), no en el repo.

## La conclusión en una línea

**El FE no puede mover la aguja porque en este espacio de features no queda nada que agregar.** El
techo medido sin *drift* (CV dentro de 202106) da **452 M con las 150 crudas y 456 M con v2: +3,8
M**. Mientras tanto, el *drift* entre meses cuesta **~50 M** (452 → 403). Además el instrumento no
resuelve efectos por debajo de ~8–10 M. Quedan en pie **H1 + H2 + H3**, y la palanca es **H4: el
mes**.

---

## 1. La escala real del efecto: ±5 clientes

Cada positivo que entra o sale del top-11.000 vale **1,1 M** (+1.072.500 que se gana o se pierde,
más los 27.500 del negativo que ocupa su lugar). Entre `base` y `v2`, en el ensamble por rank:

| fold | clientes que cambian de lado | positivos que salen | positivos que entran | neto |
| --- | ---: | ---: | ---: | ---: |
| 202105 | 535 | 13 | 6 | **−7** |
| 202106 | 537 | 11 | 14 | **+3** |

**Tres rondas de FE movieron entre 3 y 7 clientes en meses de 870–1.098 positivos.** Los "−1,5
a −5,6 M" son eso.

## 2. H3: el instrumento no tiene resolución para lo que se busca medir — **EN PIE**

### 2a. El Spearman de 0,99 no es una pista: es el piso del ruido

| comparación (202106) | Spearman | comparten del top-11k |
| --- | ---: | ---: |
| base vs v2 | 0,989 | 95,1% |
| base vs v101 | 0,991 | 94,8% |
| **base semilla 1 vs base semilla 2** | **0,994** | **94,2%** |

Una configuración con 17 columnas nuevas se aleja del base **lo mismo** que el base se aleja de sí
mismo con otra semilla. El 0,988–0,995 dice que **el cambio de features es del tamaño del ruido
de semilla**, no que todos los modelos "vean lo mismo".

### 2b. Bootstrap de clientes: el intervalo es más ancho que cualquier efecto medido

Remuestreando clientes del mes de validación (1.000 réplicas, ensamble por rank):

| fold | Δ v101 − base | IC 95% | Δ v2 − base | IC 95% |
| --- | ---: | --- | ---: | --- |
| 202105 | −4,8 M | [−16,5, +5,5] | −8,0 M | [−18,7, +1,1] |
| 202106 | +4,4 M | [−6,6, +16,5] | +2,9 M | [−8,8, +14,3] |

**sd del delta ≈ 5–6 M por fold.** Con dos folds, el efecto mínimo detectable al 95% es de
**~8–10 M al mes ≈ 8–9 clientes**. Ninguna de las tres rondas llegó a ese tamaño, ni a favor ni en
contra. El "−1,5 a −5,6 M" **no es una pérdida medida, es un cero con ruido**. (El pareado por
semilla de `c112`/`c116` sólo controla la semilla, no qué clientes se fueron ese mes, y por eso da
p chicos que no se sostienen.)

### 2c. El AUC sí ve la mejora; la ganancia @K no

| | ΔAUC pareado por semilla | gana en | p |
| --- | ---: | :-: | ---: |
| v101, 202105 | +0,0008 | 9/10 | 0,014 |
| v101, 202106 | +0,0009 | 9/10 | 0,004 |
| v2, 202105 | +0,0004 | 8/10 | 0,084 |
| v2, 202106 | +0,0011 | 9/10 | 0,020 |

La ganancia @K es una métrica **de pertenencia al conjunto**: reordenar dentro del top-11k vale
cero, y reordenar dentro de la cola también. Sólo cuenta cruzar el corte. Las features mejoran el
orden **lejos del corte**, entre los que ya se mandaban y entre los que no se iban a mandar nunca.

### 2d. Las features de ventana significan otra cosa en los meses de train

Las columnas de los meses de **entrenamiento** de los dos folds tienen la ventana truncada:

| feature (media) | 202103 | 202104 | 202105 | 202106 |
| --- | ---: | ---: | ---: | ---: |
| `w_rojo_3m` | 0,204 | 0,404 | 0,600 | 0,567 |
| `w_canales_cero_3m` | 0,037 | 0,073 | 0,107 | 0,106 |
| `v_meses_en_rojo` | 0,204 | 0,404 | 0,600 | 0,759 |
| `w_ctrx_min_3m` | 118,2 (= `ctrx_quarter`) | 114,0 | 110,3 | 110,9 |

La corrección de `c115` a "ventanas fijas" estabiliza 202105 → 202108, pero **no** los meses de
train. En 202103 la ventana tiene **una** fila: `w_rojo_3m ≡ w_rojo` y `w_ctrx_min_3m ≡
ctrx_quarter`. El fold A entrena sólo en 202103, así que **aprende cortes sobre {0,1} y valida sobre
{0,1,2,3}**. Y `w_ctrx_min_3m` es la **segunda feature en importancia** del modelo v2 (10% del
gain). **Ninguna feature histórica fue medida en condiciones comparables a 202108.**

### 2e. Contaminación del fold A

En 202105, 225 de las 3.222 filas con `Visa_Finiciomora = 0` inyectado entran en el top-11k del
base, con tasa de BAJA+2 de 0,81% (~−6 M de peso muerto). Afecta a todas las configuraciones por
igual, así que no sesga el Δ. Pero sí ensucia el nivel del fold que menos se parece a 202108.

## 3. H4: el margen cambia de mes a mes — **EN PIE: es donde está la plata**

**Prueba directa.** Se tomaron los clientes con la *flag* que el base dejaba en los puestos
11k–30k y se los pasó al frente, sin reentrenar nada. Es lo máximo que esa *flag* podía aportar:

| flag | 202105 | 202106 |
| --- | ---: | ---: |
| `v_rojo_sin_acuerdo` | −4,4 M | +16,5 M |
| `v_meses_sin_trx >= 2` | −3,3 M | +15,4 M |
| `w_aq1_ctrx0` | −6,6 M | +14,3 M |
| `w_ctrx_cae_fuerte` | −7,7 M | +13,2 M |
| `w_canales_cero_3m >= 3` | −11,0 M | +6,6 M |
| **todas juntas** | **−24,2 M** | **+38,5 M** |

**Las mismas *flags*, con signo opuesto según el mes.** La tasa de BAJA+2 entre los que tienen la
*flag* **dentro de la franja del margen** (rank 6k–20k) lo explica:

| flag | 202105 con / sin flag | 202106 con / sin flag |
| --- | --- | --- |
| `v_rojo_sin_acuerdo` | 2,9% / 1,7% | **11,3% / 2,6%** |
| `v_meses_sin_trx >= 2` | 1,6% / 1,7% | **11,3% / 2,6%** |
| `w_aq1_ctrx0` | 0,3% / 1,7% | **15,9% / 2,7%** |

En 202106 las *flags* son oro en el margen (×4–6). En 202105 no valen nada, y algunas son
negativas. Los lifts univariados de 9–11× son reales, pero pertenecen **a la población**. En el
margen, que es donde se gana la plata, el lift depende del mes. Es el mismo fenómeno que `z701`
encontró con `mprestamos_personales` y que `c108`/`c110` midieron como «manda la recencia»: la
relación X→y en el margen deriva mes a mes. Promediada sobre los dos folds, da ≈ 0.

## 4. H1: LightGBM ya lo tenía — **EN PIE**

Los positivos de cada *flag* que el base **ya mandaba** sin ella:

| flag | % de sus positivos ya en el top-11k (202105 / 202106) | positivos con flag que quedaban afuera |
| --- | --- | ---: |
| `v_tc_cerrando` / `w_tc_cierre_nuevo` | 97% / 83–84% | 1 / 5 |
| `v_meses_sin_trx >= 2` | 94% / 85% | 4 / 20 |
| `v_rojo_sin_acuerdo` | 88% / 80% | 7 / 25 |
| `w_ctrx_cae_a_cero` | 94% / 93% | 1 / 2 |

Importancia (gain, promedio de 10 semillas, fold 202106): las *flags* de lift alto caen en
`w_rojo_sin_acuerdo` **#65**, `w_tc_cerrando` #63, `w_ctrx_cae_a_cero` #108 y `w_aq1_ctrx0` #123
de 167. El árbol llega solo con dos cortes. Las que el modelo **sí** usa son las continuas e
históricas: `w_ctrx_min_3m` #2 (10%), `w_fee_por_trx_rk` #3 (6,9%) y `w_anclas` #7. Entre todas
las nuevas suman 25–30% del gain. **El modelo las usa, y mucho**: no es que las ignore. Les saca
importancia a las crudas que ya cubrían lo mismo (`ctrx_quarter` baja de primera holgada a 11%).

### Redundancia: ¿se puede reconstruir la feature desde las 150 crudas de la misma fila?

LightGBM de 300 árboles, entrenado en 202104 y evaluado en 202106:

| feature | derivable | |
| --- | --- | --- |
| `w_rojo`, `w_rojo_sin_acuerdo`, `w_aq1_ctrx0`, `w_tc_cerrando` | AUC **1,000** | idénticas a dos cortes del árbol |
| `w_mora_vieja` | R² 0,9995 | |
| `w_cobra_sueldo`, `w_usa_cajero` / `w_anclas`, `w_fee_por_trx_rk` | AUC 1,000 / R² 1,000, 0,998 | las que más importancia toman: **son crudas re-expresadas** |
| `w_ctrx_cae_a_cero` | AUC 0,976 | |
| `w_ctrx_min_3m` | R² 0,966 | la segunda en importancia, y casi `ctrx_quarter` |
| `w_ctrx_cae_fuerte` | AUC 0,918 | |
| `w_canales_mes` / `w_tc_saldo` | R² 0,91 / 0,89 | |
| `w_canales_cero_3m` / `w_rojo_3m` | R² 0,76 / 0,69 | lo único con historia genuina… |
| `v_meses_sin_trx` / `v_meses_en_rojo` | R² 0,70 / 0,53 | historia genuina (y truncada en train, §2d) |
| `w_tc_cierre_nuevo` | AUC **0,58** | …y esta: la única de verdad nueva, con 0,4% de cobertura |

### Canaritos (30 columnas de ruido, fold B, v2, 3 semillas)

El mejor canarito queda en el puesto **#57** de 197. **111 de las 167 columnas quedan por debajo**
—reproduce el 120/154 de `z401`—, y entre ellas **siete de las 17 nuevas**: `w_rojo_sin_acuerdo`,
`w_aq1_ctrx0`, `w_ctrx_cae_a_cero`, `w_ctrx_cae_fuerte`, `w_canales_cero_3m`, `w_mora_vieja` y
`w_usa_cajero`. **Las de lift 10–14× puntúan como ruido**: son redundantes, no inútiles. Las que
superan al canarito son las continuas que reemplazan a una cruda (`w_ctrx_min_3m` #2, `w_anclas`
#4, `w_fee_por_trx_rk` #6).


## 5. H2: saturación del espacio de features — **EN PIE** (corrige mi lectura inicial)

| escenario, validando en 202106 | ganancia @11k |
| --- | ---: |
| oráculo (mandar exactamente a los 1.098 BAJA+2) | 1.178 M |
| **sin *drift*: CV 5-fold dentro de 202106, v2** | **455,9 ± 1,6 M** (AUC 0,9065) |
| **sin *drift*: CV 5-fold dentro de 202106, base** | **452,1 ± 2,2 M** (AUC 0,9053) |
| *out of time*, ensamble base (train 202103+04) | 405,9 M |
| *out of time*, base, train 202104 completo | 402,6 ± 3,6 M |
| ídem con 50% / 25% / 12,5% de las filas | 395,3 / 363,0 / 389,8 M |

Tres lecturas:

- **Ni en el mejor caso el FE vale más de ~4 M.** Con el modelo entrenando sobre el mismo mes que
  se evalúa, sin *drift* y sin truncamiento, v2 le gana al base por **+3,8 M**: unos 3,5 clientes.
  Eso es el techo de lo que estas features pueden aportar, y está por debajo del umbral de
  detección.
- **El oráculo está lejos (1.178 M), pero no es alcanzable con estas columnas.** El 34–36% de los
  BAJA+2 **ningún** modelo de los 30 de cada fold lo manda nunca (316 / 371 clientes). Todos los
  modelos fallan con los **mismos** clientes: es el núcleo «Te ghostea» de `v101`, sin rastro en
  los datos de cuenta.
- **Más filas tampoco.** Con 1/8 de 202104 el modelo ya está a ~13 M del mes completo (la curva es
  ruidosa con 3 semillas, pero plana). Y `c108` ya mostró que sumar un mes viejo **resta**.

La distancia que sí existe es **452 → 403 = ~50 M, y es *drift***: lo que el modelo pierde por
entrenar en un mes y predecir en otro. Es 12× el techo del FE.

### Corrección a §3

El empuje de +38 M en 202106 se eligió **mirando 202106** (los lifts del tamiz incluyen ese mes).
El techo *in-month* de v2 dice que el modelo, viendo el mes, sólo saca +3,8 M de esas mismas
columnas. O sea: **las crudas ya alcanzan para encontrar el margen de 202106 si se entrena en
202106**. Lo que cambia entre meses no es qué columnas hay sino **qué corte sobre ellas separa el
margen**.


---

## Qué implica para el esfuerzo que queda

1. **Dejen de buscar features.** El número: **+3,8 M de techo** —v2 contra base, entrenando y
   validando en el mismo mes, sin *drift*—, contra un efecto mínimo detectable de **~8–10 M** por
   fold. Aunque una cuarta ronda fuera perfecta, no se podría ver. Las *flags* de lift 10× son
   reconstruibles al 100% desde la fila (AUC 1,000) y puntúan por debajo de un canarito.

2. **El esfuerzo va al *drift* temporal: son ~50 M en juego**, la brecha entre 452 *in-month* y
   403 *out of time*. Las palancas ya medidas que superan el umbral son todas temporales: `c108`
   (train sólo con el mes más reciente, +10,6 M, 18/20) y `c110` (decaimiento λ = 0,15, +8,0 M,
   10/10). Lo siguiente es **entrenar 202108 con 202106 dominante** (decaimiento), y el **corte**
   para una prevalencia desconocida (`corte_minimax`).

3. **Cambiar el estadístico de comparación.** El Wilcoxon pareado por semilla sobre la ganancia
   @11k no remuestrea clientes, así que da p chicos para efectos de ±5 clientes. Para decidir
   entre configuraciones: **bootstrap de clientes** del mes de validación (sd ≈ 5–6 M por fold) y,
   como desempate, AUC o ganancia integrada en 8k–15k.

4. **Las features de ventana nunca se midieron bien** (§2d): en 202103 `w_*_3m` tiene una fila. No
   vale la pena arreglarlo **para medirlas**, porque el techo *in-month* ya acota su valor en
   ~4 M. Pero si v2 entra en la entrega, el decaimiento temporal resuelve el truncamiento de
   rebote, porque los meses truncados pesan poco.

5. **Usar v101/v2 en la entrega es neutro**: Δ indistinguible de 0, AUC +0,001 (9/10). No hay
   motivo para sacarlas ni para pelear por ellas.

## Correlación de errores, para el video de Michelina

Cada positivo de los meses de validación, contado sobre los 30 modelos de su fold (3
configuraciones × 10 semillas):

| | 202105 | 202106 |
| --- | ---: | ---: |
| lo mandan **todos** | 445 (51%) | 551 (50%) |
| lo mandan algunos (disputados) | 109 (13%) | 176 (16%) |
| **no lo manda ninguno** | **316 (36%)** | **371 (34%)** |

**Un tercio de las bajas es invisible para cualquier modelo de esta familia**: es el «Te ghostea»
de `v101`. Todo el FE pelea por la franja de disputados, que son 109–176 clientes. Y aun ahí el
techo sin *drift* da ~4 M.
