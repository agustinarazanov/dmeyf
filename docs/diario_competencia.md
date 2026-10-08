# Primera Competencia DMEyF 2026 — contexto del directorio

Cierra el **domingo 11 de octubre 2026, 23:59:59 ART**. Vale el **ÚLTIMO** submit recibido antes de
esa hora. Este archivo es el mapa de `monday/competencia/`; el del repo entero es `../CLAUDE.md`.

## Qué se predice

**No** es "los clientes que se van en agosto". Es: de los **164.647 clientes de la foto 202108**,
cuáles reciben **estímulo**. Paga `+1.072.500` si el estimulado resulta `BAJA+2` (está en 202108,
está en 202109, **no** está en 202110) y cuesta `−27.500` si no. El CSV es una **lista de acciones**,
no de predicciones; coinciden sólo porque el conjunto óptimo de acciones es el top-K por
`P(BAJA+2)`.

## Las reglas del bot (Zulip topic 61, msgs 191490–191496)

| | |
| --- | --- |
| Destino | DM a `@competencia-uno`, user_id **1692** |
| Cuerpo | `submit    <NombreUnico>` — minúsculas, sin espacios, **distinto cada vez** |
| Adjuntos | 1–20 `.csv`, nombres distintos |
| Formato | **sin encabezado**, una columna de `numero_de_cliente`, **8.000–15.000** filas, enteros |
| Cupo | **13 submits/día**, renueva **21:00 ART** |
| Respuesta | `Public Gain mean` y `Public gain std dev` sobre los archivos, **sin escalar, en millones de ARS**, calculado sobre el **~25% público** |
| Otros comandos | `list submits`, `help` (no consumen submit) |

## Dos trampas verificadas en los datos

1. **`numero_de_cliente` es `DOUBLE`** en `competencia_01.parquet`, y los dos ids que el instructor
   marcó —`48000000` y `31100000`— **están los dos en 202108**. Sin castear a `int64` se escribe
   `4.8e+07` y el submit falla. `escribir_envios()` castea, **relee el archivo** y falla si encuentra
   `e+` o un punto.
2. **`Visa_Finiciomora` tiene ceros inyectados sólo en 202105 (3.222) y 202108 (3.249)**, cero en
   todos los demás meses. Cae justo en el mes de la competencia. Las dos `F*iniciomora` están en
   `ROTAS_EN_202108` y se sacan; la mora se lee de `Visa_delinquency` / `Master_delinquency`.

## El hallazgo que más condiciona todo: el corte lo manda el MES, no el modelo

El **mismo modelo** (train 202103), scoreado en dos meses distintos:

| validación | BAJA+2 | K* óptimo |
| --- | ---: | ---: |
| 202105 | 870 | **8.092** |
| 202106 | 1.098 | **14.051** |

Y con la validación fija en 202106, cambiar el modelo mueve K* apenas (13.055 → 14.788). O sea: **la
prevalencia del mes domina**, y la de 202108 **no se conoce** (los meses etiquetados van de 870 a
1.139).

Por eso el corte **no** se elige con el óptimo de un mes. Se elige por **minimax regret** sobre los
dos folds (`competencia.regret` / `corte_minimax`):

| K | regret 202105 | regret 202106 | peor caso |
| ---: | ---: | ---: | ---: |
| 8.000 | −0,7% | −14,4% | −14,4% |
| **11.000** | −4,5% | −4,5% | **−4,5%** ← minimax |
| 14.500 | −15,9% | −0,2% | −15,9% |

Elegir 14.500 —el óptimo de 202106— es la misma trampa que perseguir el pico de **una** semilla, un
nivel más arriba: se persigue el pico de **un mes**. El mínimo de regret esperado cae en 10.000 y el
minimax en 11.000, los dos lejísimos de 14.500.

## Lo que contestó el bot, y qué mide su `std dev`

Primer submit (`c100_esqueleto`, 3 semillas, 11.000 envíos), 2026-10-03:

```
📦 Submission received: 3 candidate models
📊 Public gain -> mean: 89.0175, std dev: 0.6496
🆔 Submission ID: 1572-1791073635598180
📅 Submissions today: 1/13
```

> **El `std dev` del bot NO es el ruido de la partición.** Es el desvío entre los **archivos**
> —o sea entre **semillas**— sobre una partición pública **fija**. Da 0,73% de la media. La
> simulación local del sorteo 25/75 da 11,8%, pero esa partición **no se vuelve a sortear**: es la
> misma para todos y para siempre. Entonces comparar dos submits está **pareado** sobre ella, y el
> umbral útil para decidir **no es 11,4 M**. Hay que medir el ruido *pareado*, que es mucho menor.

Otras lecturas de esa respuesta:

- `89,0175 × 4 ≈ 356 M` extrapolado al mes entero, contra los **317 M** que predijeron los dos
  folds. 202108 viene **mejor** que la validación — probablemente porque el modelo final entrena
  con cuatro meses y los folds sólo con uno o dos.
- El `help` confirma el formato: *"1 column with the IDs you predict as positive (no header)"*.
- *"each submission spends 1, no matter how many files it has"*: **los archivos son gratis, lo que
  se gasta son submits**. Conviene mandar siempre 20 semillas, no 3.
- *"This is now your final submission — your LAST pre-deadline submit always is, with no separate
  step needed"*: el bot lo hace solo, no hay que reenviar como pedía el mensaje del profesor.

> **El deadline del bot está mal y NO hay que preocuparse.** El `help` dice
> `Deadline: 2026-10-05T03:01:01`, que es exactamente el cierre viejo (`dom 04-oct 23:59:59 ART`,
> en UTC) de **antes** del cambio de cronograma del 21-sep. El cierre real es **dom 11-oct 23:59**,
> como dicen los mensajes del profesor. **Decisión tomada: no se gasta ni un submit en protegerse
> de ese reloj.** No volver a levantar el tema.

## Estrategia de submits

El cupo (117 submits hasta el cierre) **no es el recurso escaso**; la suerte en el privado sí.
`z301` midió que **para un modelo fijo público y privado correlacionan −1**, y la simulación local
del sorteo 25/75 sobre 202106 da **sd ≈ 11,4 M sobre una media de 96 M (11,8%)**. Diferencias
menores que eso entre dos submits **son el sorteo, no el modelo**.

1. **El bot no es un conjunto de validación.** Se elige localmente: dos folds × semillas × Wilcoxon.
2. **~8 submits en total**, cada uno pre-registrado con su hipótesis y su delta.
3. **Un submit = una configuración × hasta 20 semillas**, para que media/sd aíslen la semilla.
4. **Reenvío final el viernes 10**, no la noche del 11.

> **Regla: los submits se gastan en MEDIR, no en que la entrega vigente se vea bien.** Vale el
> último submit antes del cierre, así que acomodarlo durante la semana **desperdicia cupo**. Cada
> submit tiene que contestar algo que los datos locales no pueden: los folds validan **junio**, el
> público mide **agosto**, y en este problema el mes manda. El submit final se arregla **cerca del
> domingo**, de una.

> **Regla dura: nunca se envía nada sin preguntar antes**, mostrando nombre, config, delta contra el
> submit anterior, archivos con su cantidad de envíos y la ganancia esperada. La estructura lo
> impone: un `submit` nace `estado='preparado'` con `enviado_en NULL`.

## Qué hay acá

```
competencia.py      datos, ganancia, corte, regret, público/privado, escritura de envíos, caché
registro.py         el ledger en Postgres (base `competencia`)
c100_esqueleto.py   fase 0: params de z701, sin búsqueda, de punta a punta en ~70 s
experimentos/<nombre>/
    params.json  modelo_*.txt  scores_202108.parquet  regret.parquet  envios/*.csv
```

El código de Zulip vive en el **otro repo**, `../../zulip/competencia.py`:
`validar`, `comando`, `submit`, `escuchar`, `ultimas`.

## Después de un reboot

Nada se pierde, pero **nada arranca solo**. En orden:

```sh
open -a Docker                       # Docker Desktop no arranca solo en macOS
docker start zulip-postgres          # el contenedor tiene RestartPolicy=no
docker exec zulip-postgres pg_isready
```

Los datos viven en el volumen `zulip`, que sobrevive reboots y recreaciones del contenedor: el
*study* de Optuna, el ledger y el archivo de Zulip quedan intactos. Los modelos están en disco bajo
`experimentos/`, así que **retomar no reentrena nada**.

La búsqueda se retoma sola con el mismo comando: `load_if_exists=True` arranca donde quedó y el TPE
lee los *trials* previos, así que **no pierde lo aprendido**. `--trials N` son N *trials* **más**.

> Si se corta la búsqueda a mitad, el *trial* en vuelo queda `RUNNING` para siempre y Optuna lo
> cuenta. Hay que marcarlo `FAIL` a mano:
> ```python
> from optuna.trial import TrialState
> for t in s.trials:
>     if t.state == TrialState.RUNNING:
>         s._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
> ```

## Persistencia y caché

| Qué | Dónde |
| --- | --- |
| Study de Optuna | **Postgres** (`registro.dsn_url()`) — resumible y permite varios procesos a la vez |
| Boosters | `experimentos/<exp>/modelo_<etiqueta>.txt`, patrón `entrenar_o_cargar` de `z701` |
| **Scores** | `experimentos/<exp>/scores_<mes>.parquet` — **el caché que más rinde**: permite re-cortar en otro K, re-ensamblar y re-simular público/privado **sin reentrenar** |
| Ledger y resultados | **Postgres** |

La llave de caché hashea `params + meses + target + peso + dataset + semilla`: volver a correr con
los mismos parámetros **carga**, no reentrena. `registro.purgar()` borra los bytes de los submits
descartados y **conserva la fila**, para no repetir un experimento ya hecho.

## La identidad con la que se envía — se descubrió a los golpes

> **El submit tiene que salir de la cuenta personal de la alumna, no del bot que archiva.**

`~/zuliprc` autentica como **`Claude` (user_id 1687, `is_bot=true`)**, que es la cuenta con la que
corre el ingestor. Mandarle `help` a `@competencia-uno` desde ahí devuelve:

> :prohibited: You are not authorized to use this bot. If you think you should be, please tell a teacher.

Las tres cuentas en juego:

| id | nombre | bot | |
| ---: | --- | :-: | --- |
| **1572** | Agustina Razanov | no | la cuenta de alumna, **la que tiene que enviar** |
| 1687 | Claude | sí | la del `~/zuliprc`, sólo para archivar |
| 1692 | competencia-uno | sí | el bot de la competencia |

La solución es un zuliprc aparte apuntado por **`COMPETENCIA_ZULIPRC`** (por defecto
`~/zuliprc-competencia`): el ingestor sigue usando el bot y sólo los submits usan la identidad
personal. La API key se saca en Zulip: rueda dentada → *Personal settings* → *Account & privacy* →
*API key*.

`exigir_cuenta_de_alumna()` **rechaza el submit** si la cuenta configurada es un bot, antes de subir
un solo archivo. `competencia.py quien` dice con qué identidad se está hablando.

## Cómo recibimos la respuesta del bot

Zulip tiene **API de eventos en tiempo real** (`register` + `get_events`, long-polling): el servidor
mantiene la conexión abierta hasta que llega el evento. No hace falta *polling* tonto ni un webhook
—los *outgoing webhooks* son para bots que **reciben** mensajes, y acá somos el usuario que recibe un
DM—. `escuchar()` registra una cola con `narrow=[["is","dm"]]` y filtra por `sender_id == 1692` en el
callback, porque el narrow no filtra por remitente.

## Lo que NO hay que heredar de los notebooks de clase

1. **`numero_de_cliente` y `foto_mes` fuera de `X`.** `z601` deja las dos adentro. El id memoriza; y
   **202108 es un `foto_mes` que el modelo nunca vio**, así que LightGBM lo manda a un lado
   arbitrario.
2. **`bagging_fraction` sin `bagging_freq` no hace nada** (`z601` lo olvida, `z701` lo arregla).
3. **`best_iter` se calcula y no se usa**: `z601` entrena con `num_boost_round=250` hardcodeado.
4. **Una sola convención de ganancia**: `+1.072.500` / `−27.500`, la consistente con el umbral.
5. **`ntile()` sobre columnas con muchos ceros no es determinístico**: desempatar con el id.
6. `fe_panel.COLUMNAS_DE_BAJA` fuera de `X` siempre: **`es_baja` *es* la respuesta**.

## Fase 1: qué dijeron los folds

Los cuatro esquemas de target × dos datasets, cada uno **en su propio corte**, pareado por
(fold, semilla) contra `base_baja2`, cerrado con Wilcoxon:

| config | corte | 202105 | 202106 | Δ vs base | gana en | p |
| --- | ---: | ---: | ---: | ---: | :-: | ---: |
| **`base_pesos25`** | 11.000 | 250,8 | **399,7** | **+8,6 M** | **8/10** | **0,0078** |
| `base_pesos5` | 10.500 | 253,8 | 386,6 | +3,5 M | 5/10 | 0,37 |
| `base_baja2` | 11.000 | 236,9 | 396,4 | — | — | — |
| `base_baja12` | 11.000 | 251,0 | 379,3 | −1,5 M | 5/10 | 0,75 |
| `fe_baja2` | 11.500 | 237,0 | 349,5 | **−23,4 M** | 2/10 | 0,037 |

**Pesar `BAJA+1` en 0,25 gana**, y es lo único significativo de la grilla. Es la respuesta a la
pregunta de Zulip (*¿qué métrica optimiza con las dos bajas juntas?*): **ninguno de los dos
esquemas puros; la interpolación**.

### El FE pierde, y no por lo que yo creía

Primera hipótesis, **equivocada**: en 202103 las 102 columnas de historia son **100% nulas** (no
hay mes anterior) contra ~1,4% en el resto, así que el patrón de nulos identifica a marzo y vuelve
a meter el `foto_mes` que sacamos a propósito. Es un defecto real —y **introducido por nosotros**,
`z402` nunca lo vio porque entrenaba un mes solo— pero entrenando **sólo en 202104** el FE sigue
perdiendo **40,9 M** (antes 46,9 M). Explica ~6 de 47 M.

| train [202104] → 202106 | @11.000 | Δ vs base | gana en |
| --- | ---: | ---: | :-: |
| `base_p25` | 409,0 M | +6,4 M | 4/5 |
| `base` | 402,6 M | — | — |
| `ferank_p25` (sin lag/delta/slope) | 374,2 M | −28,4 M | 0/5 |
| `fe` | 361,7 M | −40,9 M | 0/5 |

Explicación que queda en pie: los hiperparámetros son los que `z701` afinó para **150** columnas y
se están aplicando a **297**. `feature_fraction`, `num_leaves` y `min_data_in_leaf` dependen de la
dimensionalidad, así que **esto nunca fue una prueba justa del FE**. Lo decide Optuna, por dataset.

> Con 5 semillas el p de Wilcoxon más chico posible es 0,0625: un `0/5` es la evidencia más fuerte
> que ese tamaño de muestra puede dar.

## El umbral REAL para leer el leaderboard

Simulando 2.000 particiones del 25% sobre 202106, el sd **de la diferencia** entre dos submits:

| comparando | sd de la diferencia pública |
| --- | ---: |
| variantes del mismo pipeline | **3,15 M** |
| datasets distintos | 16,3 M |

**Para declarar ganador con ~95% hace falta una brecha pública ≥ 6,3 M**, que son ~25 M de
diferencia real en el mes entero. Verificación: `base_pesos25` le gana a `base_baja2` por 14,3 M
reales → se ve como 3,59 ± 2,81 M en público → el público acierta el orden **90%** de las veces.

> **No usar el `std dev` del bot como umbral.** Dio 0,6496, que es la **semilla**; el ruido de una
> comparación es **5× más grande**. Y mi estimación previa de ±11,4 M era igual de inútil pero al
> revés: sobreestima, porque la partición no se vuelve a sortear.

## Fase 2: la búsqueda no le ganó al baseline

31 *trials* de Optuna sobre `base`, con `esquema` y `peso_baja1` **dentro** del espacio, más
`lambda_l1/l2`, `min_gain_to_split` y `max_bin`. Revalidando el top 5 *out of time*:

| | corte | 202105 | 202106 | media |
| --- | ---: | ---: | ---: | ---: |
| **`base_pesos25`** (params de `z701`, w=0,25) | 11.000 | 250,8 | **399,7** | **325,3** |
| optuna t15 (w=0,91) | 10.000 | 262,5 | 373,3 | 317,9 |
| optuna t17 (w=0,99) | 10.000 | 262,7 | 372,0 | 317,4 |
| `base_baja2` (w=0) | 11.000 | 236,9 | 396,4 | 316,7 |

Los candidatos de Optuna quedan **~7,5 M abajo** del baseline (5/10, o sea empate). **La búsqueda no
aportó nada.** Spearman entre el ranking CV y el *out of time*: **+0,10**.

**Por qué falló**: el objetivo era ganancia en CV **dentro** de 202103-202104, y encima con el corte
elegido *post-hoc* sobre el mismo mes. Las dos cosas favorecen fusionar las clases —más positivos, y
la fusión no paga el cambio de escala—, así que TPE se fue a `peso ≈ 0,9`, que es justo lo que los
folds rechazan. **Optimizó bien hacia el blanco equivocado.** Además descubrió "bajar el *learning
rate* y entrenar más rondas", que sube el CV del mes y tocó el techo de 1.500 rondas.

Y hay algo estructural: subir el peso no mejora el modelo, **cambia en qué mes anda bien**
(202105 mejor, 202106 peor). El peso, como el corte, es **una apuesta sobre la prevalencia del mes**.

> Para el video de Michelina esto es material directo de *"qué no funcionó"*: una búsqueda
> bayesiana de 31 *trials* perdiendo contra unos hiperparámetros heredados, por tener mal el
> objetivo y no el algoritmo.

## Fase 3: la entrega vigente

`c107_pesos25`: params de `z701` + target fusionado con `BAJA+1` pesando **0,25**, entrenado en los
cuatro meses etiquetados, **corte 11.000** (minimax con 20 semillas, peor caso −3,47%).

**El ensamble por rank se midió, no se supuso**: 254,1 M vs 249,4 M de la semilla media en 202105 y
405,9 M vs 398,9 M en 202106 — **le gana al 80-85% de las semillas sueltas**, aunque no a la mejor
(que no se puede saber de antemano). Por eso es el candidato de un solo archivo.

| submit | archivos | public mean | std |
| --- | ---: | ---: | ---: |
| `c100_esqueleto` (w=0) | 3 | 89,0175 | 0,6496 |
| `c107_pesos25` (w=0,25) | 20 | 91,3550 | 1,4873 |
| **`c107_pesos25_ens`** (ensamble) | 1 | **92,6750** | **0,0000** |

Dos cosas que salieron como estaban previstas:

- **El cambio de target cayó donde decían los folds.** Local: +8,6 M al mes → ~+2,15 M en el 25%
  público. Observado: **+2,34 M**. Un *check* lindo de toda la maquinaria de validación.
- **El `std` de un solo archivo es 0,0000.** No es precisión, es ausencia de medición.

> Pero **+3,66 M en total sigue por debajo del umbral de 6,3 M**: el leaderboard **no confirmó**
> nada. Se movió en la dirección que predijeron los folds, que es alentador, no evidencia
> independiente. La razón para creerle a `pesos25` sigue siendo el pareado (8/10, p=0,0078).

## El drift: no es «marzo es malo», es DISTANCIA

> [!warning] Esto corrige una interpretación que estuvo vigente casi todo el trabajo

**Lo que se midió primero.** Entrenar con un mes solo (202104) le gana a entrenar con dos
(202103+202104): **+10,6 M, 18/20, p=0,0002**. Se duplicaron los positivos y el modelo empeoró.
Y no es la prevalencia: recortando 202104 a los mismos **960** positivos que 202103, igual le saca
**+34,2 M (10/10, p=0,002)**.

**La interpretación equivocada** fue *«marzo es un mes de entrenamiento malo»*. De ahí salieron
tres remedios y **los tres fallaron**: decaimiento exponencial por antigüedad, sacar marzo, y
entrenar sólo con el mes más reciente.

### El test que lo resuelve

Marzo siempre estuvo a 3 meses de la validación y abril a 2: mes y distancia venían confundidos.
Separándolos con la **captura** (qué fracción de los que se van cae en el top K — no depende de la
prevalencia del mes):

| caso | 5k | 9k | 11k | 15k |
| --- | ---: | ---: | ---: | ---: |
| marzo → mayo (**distancia 2**) | 39,1% | 52,6% | 57,8% | 64,7% |
| marzo → junio (**distancia 3**) | 35,6% | 49,6% | 55,8% | 67,0% |
| abril → junio (**distancia 2**) | 35,0% | 51,9% | 58,9% | 68,3% |

- efecto de la **distancia** (mismo marzo, a 2 vs a 3): **+1,6%**
- efecto del **mes** (misma distancia, abril vs marzo): **−0,0%**

**Marzo y abril son igual de buenos. Lo que pesa es cuán lejos está el dato del mes a predecir.**

### Y la aritmética cierra

A K=11.000 sobre 202106 (1.098 bajas): marzo captura 55,8% → 613 aciertos → 371,5 M; abril captura
58,9% → 647 → 408,9 M. Diferencia **+37,4 M** contra los **+34,2 M** medidos. **La brecha entera son
3,1 puntos de captura: 34 fugados más, ni uno menos.**

> [!note] El modelo envejece ~3 puntos de captura por mes de distancia
> Y eso explica de una vez las tres cosas que tratábamos por separado:
> - **Sacar marzo no sirvió** (−4,07 M en agosto): no sacábamos un mes malo, sacábamos un cuarto
>   de los datos. El perjuicio ya estaba diluido.
> - **El decaimiento y «sólo el mes reciente» fallaron**: atacaban «los meses viejos son malos»,
>   que no es el problema. El mes más nuevo que tenemos —202106— **ya está a 2 meses de agosto**,
>   y no hay forma de acercarlo.
> - **Los ~50 M de brecha in-month vs out-of-time no son un defecto nuestro**: son el costo
>   estructural de predecir con dos meses de atraso, que es lo que el enunciado exige.

### Lo que NO explica a marzo (descartado midiendo)

Nulos (6,88% contra 6,6–6,9% de los demás), composición de clientes, antigüedad, actividad,
productos: **marzo es idéntico a los otros meses**. Y la relación variable→fuga también: el lift
del decil 1 de `ctrx_quarter` contra la tasa base de cada mes da 5,97 / 6,09 / 5,30 / 5,87 — los
cuatro separan igual.

> [!warning] Un cociente con denominador chico me engañó cuatro veces hoy
> Llegué a «medir» que `ctrx_quarter` separaba 33,7× en marzo contra ~50× en los otros meses. Era
> un artefacto: el denominador es el decil 10, que tiene **9 a 17 bajas en total**. Normalizando
> contra la tasa base de cada mes, los cuatro meses separan igual. Es el mismo error que el pico
> de la curva de corte (4 clientes), el bloque 9k→15k, y el `K*` de junio.

## Las features de v101, y el límite de los folds

14 columnas sacadas del EDA de Miranda, con hipótesis de negocio cada una. Los *lifts* crudos contra
la tasa base son altísimos: `v_meses_sin_trx ≥ 2` **11,6×**, `v_rojo_sin_acuerdo` **10,5×**,
`v_tc_cerrando` **9,4×**.

| | Δ vs base | gana en | p |
| --- | ---: | :-: | ---: |
| las 14 | +4,0 M | 13/20 | 0,057 |
| sólo las de la misma fila | +3,6 M | 13/20 | 0,076 |
| sólo las acumuladas | +0,4 M | 11/20 | 0,67 |

> **Predicción mía, equivocada.** Dije que las acumuladas serían las valiosas —el árbol no puede
> inventarlas— y aportaron nada. Pero el motivo es un **defecto del test, no de las features**: en el
> fold A se entrena en 202103, que tiene **cero** historia, y en el fold B con un mes a lo sumo.
> `meses_en_rojo` casi no varía ahí; en 202108 tendría seis meses de recorrido.
>
> **Los folds no pueden medir features históricas.** Es el mismo defecto que hundió al FE de `z402`.
> Que den +0,4 M no significa que no sirvan: significa que **no las medimos**.

### De paso, dos cosas de v101 que corrigen trabajo anterior

- **`ctrx_quarter` es una ventana móvil de 90 días**, así que meses consecutivos comparten ⅔ de
  ventana y su pendiente está autocorrelacionada **por construcción**. `z402` construyó pendientes
  sobre ése y otros 33 campos: es una explicación mejor que la mía («hiperparámetros mal ajustados»)
  de por qué el FE grande perdió.
- **`catm_trx` no cuenta los cajeros ajenos**: 13.634 clientes que usan sólo ajenos figuran como
  «nunca usa cajero». Cualquier corte por esa variable está mal informado.

## Bugs propios encontrados en la auditoría (sesión `debugger`)

Ninguno de estos es de la cátedra: son todos míos, de esta semana.

| # | Qué | Consecuencia |
| --- | --- | --- |
| **Gemelos en `c105`** | Los **960 de 960** `BAJA+2` de 202103 son `BAJA+1` en 202104. `StratifiedShuffleSplit` parte **filas, no clientes**, así que el 47% de los positivos del target `pesos` tiene a su gemelo del otro lado del split | **Invalida la búsqueda de Optuna.** Explica el CV de 1.008 M contra 318 M *out of time*, y explica por qué TPE se fue a `peso ≈ 0,9`: **más peso en `BAJA+1` = más pares gemelos = más leakage**. La búsqueda optimizaba la fuga de información |
| **Orden de filas en `c118`** | Scores de una tabla indexados contra ids de otra (DuckDB reordena al escribir) | Submit con ganancia **−49,86**. Arreglado con `verificar_seleccion()` |
| **`w_ctrx_cae_a_cero` en `c115`** | `coalesce(lag(...), 1)` hace que en 202103 **todo** `ctrx=0` dispare el flag (no hay mes anterior) | La feature significa algo distinto en 202103 — **el mismo defecto de inconsistencia entre meses que yo estaba tratando de arreglar** |
| **`percent_rank` y los NULL** | `ratio_seguro(comision, ctrx)` da NULL cuando `ctrx=0`, y `percent_rank` manda los NULL a **1,0** | El lift de `w_fee_por_trx_rk >= 0,9` está **inflado**: son clientes inactivos disfrazados |
| **El guard de `validos`** | Se le pasaba el mismo array del que salían los ids elegidos | Era **tautológico**. Ahora `escribir_envios(..., referencia=df)` llama a `verificar_seleccion` |
| **`resultado` sin `unique`** | `alta_resultado` dos veces duplicaba la fila y `v_submits` multiplicaba `n_archivos` | Arreglado con `unique (submit_id)` |
| **El parquet de FE cambió** | Tiene **228** columnas; los experimentos de `c102`/`c104` corrieron con **302** | El "el FE pierde 46,9 M" salió de un archivo que ya no existe. `clave()` hashea params y nombre de dataset, **no el contenido ni la lista de columnas** |

> **La moraleja**: de siete bugs, cinco sólo aparecen **midiendo**, no leyendo. Y el de los gemelos
> cambió la interpretación de un resultado que ya habíamos reportado.

## RETRACTACIONES (auditoría de números, sesión `debugger-2`)

Tres cosas que reporté hoy **no se sostienen**. Van acá arriba porque varias están citadas más
abajo en este mismo archivo.

### 1. «`pesos25` gana +8,6 M, 8/10, p=0,0078» — vale SÓLO al corte 11.000

| corte | delta | gana en | p | fold 202105 | fold 202106 |
| ---: | ---: | :-: | ---: | ---: | ---: |
| 8.000 | +5,5 M | 7/10 | 0,078 | +10,6 | +0,4 |
| **9.000** | **−0,1 M** | **5/10** | **0,939** | +9,5 | **−9,7** |
| 10.000 | +2,3 M | 6/10 | 0,539 | +9,9 | −5,3 |
| 11.000 | +8,6 M | 8/10 | 0,008 | +13,9 | +3,3 |

**Al corte 9.000 —el que mide mejor en agosto— el efecto es cero.** Y es de **un solo fold**: 202105
lo favorece en todos los cortes, 202106 lo castiga justo en el rango que usamos. Además el p mide
ruido de semilla con un n efectivo de **2 folds**, y era el mejor de 8 configuraciones probadas.

### 2. «El cambio de target cayó donde decían los folds (+2,34 público)» — no era un check

`c100` era target `baja2`, **14.500** envíos, **3** semillas; `c107` era `pesos25`, **11.000**, **20**.
Cambian target, corte y semillas a la vez. Coincidencia, no verificación.

### 3. «Spearman 0,00 entre folds y público» — no hay poder para afirmarlo

Con **n = 4** el p es 1,00, y cambiando la config de referencia da +0,6. **No es evidencia de que la
validación local no prediga el público**; es una muestra demasiado chica para saberlo.

### La calibración que faltaba: un cliente = 1,1 M público

`c118_base_9000` y `corte_9000` son el mismo modelo con 10 y 20 semillas: difieren en **60 ids de
9.000** (99,33% iguales) y en **1,07 M** de ganancia pública. Esos 60 ids son ~15 clientes públicos:
**1,07 M / 1,1 M = exactamente 1 acierto.**

> **Un cliente mueve el score público ~1,1 M.** Entonces: los vaivenes de la curva de corte son
> **1–2 clientes**; el pico de 9.000 sobre 11.000 son **~4 clientes**; cualquier comparación entre
> configs de 1–5 M son **1–5 clientes**. Antes de creerle a una diferencia pública, dividirla por
> 1,1 y preguntarse si esa cantidad de clientes es señal.

### Lo que SÍ se sostiene

- **La curva de corte es una curva**: los 8 CSV son top-k exactos y anidados del mismo ranking. La
  **forma** (sube hasta ~9.000, baja después) es mucho más probable si agosto tiene prevalencia baja
  tipo 202105. Pero hay que decir **«corte corto, 8–11k»**, no «óptimo 9.000».
- **El ensamble por rank**: el código está bien y le gana a la semilla media por +2,4 a +7 M.
  El «80–85% de las semillas» vale a 11.000; a 9.000 baja a 45–65%.
- **Recencia (`c108`)**: +34,2 M de 202104 sobre 202103 con la tasa base igualada (0,589% en los dos).
  Se sostiene también a 9.000 (+9,0 M, 16/20).
- **El leakage de gemelos en `c105`** invalida la búsqueda vieja de Optuna.

## Estado

**Hecho**
- Base `competencia` + esquema + `registro.py`. `psycopg[binary]` instalado en `facultad`.
- `../../zulip/competencia.py`: subir archivos, mandar DM, escuchar por cola de eventos, parsear.
- `c100_esqueleto.py` corre de punta a punta en ~70 s y deja 3 CSV validados.
- Corte por minimax regret (11.000) en vez del pico de un mes (14.500).
- Chequeo de leakage: KS 0,63–0,67 **no** es leakage — las top features son `ctrx_quarter`,
  `mpayroll`, `mprestamos_personales`, y ninguna columna de baja está en el modelo. Se mide *out of
  time* sobre un mes que el modelo nunca vio.

- **Primer submit enviado y aceptado**: `c100_esqueleto`, **89,0175 M** público (sd 0,6496).
  `~/zuliprc` ahora es la cuenta 1572 (antes era el bot `Claude`).
- Parser arreglado **contra la respuesta real**: las regex originales buscaban `Public Gain mean` y
  formato español, y habrían convertido `89.0175` en `890175`. Ahora saca media, sd, `submission_id`,
  cupo y un flag `aceptado`, y está probado contra la aceptación **y** contra el rechazo.
- `competencia_01_fe.parquet` regenerado: **155 → 302 columnas**, 437 MB, 73 s. Reproduce los
  números del vault: `corr(ctrx_quarter, lag1) = 0,986` y la pendiente sólo 0,364 — información
  nueva de verdad.

**Pendiente**
- `c102_comparacion.py`: base vs FE × cuatro esquemas de target, pareado por (fold, semilla).
- Medir el ruido **pareado** del leaderboard público (el umbral real para comparar submits).
- Fase 2: Optuna. Fase 3: modelo final 20 semillas + ensamble por rank.
- Fase 1: regenerar `competencia_01_fe.parquet` (**no existe**; el `.csv` homónimo es copia byte a
  byte del base), base vs FE, los tres esquemas de target, Wilcoxon.
- Fase 2: Optuna. Fase 3: modelo final 20 semillas + ensamble por rank.

> `*.md` está ignorado en el `.gitignore` de la raíz y **`CLAUDE.md` no tiene negación** (el
> `../CLAUDE.md` dice que sí la tiene, pero no es cierto). Estos docs viven en disco y no se
> versionan. Se arregla agregando `!CLAUDE.md`.

---

## La mora inyectada de agosto: real en el dato, inerte en el modelo

`Visa_Finiciomora` tiene ceros inyectados en **202105 (3.222) y 202108 (3.249)** y en ningún otro mes
— ya estaba documentado. Lo que faltaba: **el mismo proceso roto también prende `Visa_delinquency`**.

| mes | `delinquency=1` | de esos, con `Finiciomora=0` |
| --- | ---: | ---: |
| 202103 / 202104 / 202106 / 202107 | 1.479 / 1.427 / 928 / 981 | **0** |
| **202105** | **4.344** | **3.222** |
| **202108** | **4.163** | **3.249** |

Que el flag es espurio está fuera de duda: de los 3.222 marcados en mora en mayo, en **abril sólo 48
y en junio sólo 57** estaban en mora. El 98,5% nunca lo estuvo. Y el barrido de las 21 columnas
`Visa_*` deja a `delinquency` sola como anómala (99,0% cambia de julio a agosto contra 59,5% del
control); las columnas de consumo difieren 12–18pp por **composición** —los inyectados son tarjetas
más activas—, no por corrupción: `Visa_status` cambia *menos*.

**Y sin embargo no hay que tocarlo.** Tres mediciones, en este orden:

1. **Arreglar el feature** (`delinquency = 0` donde `Finiciomora = 0`) da **−0,2M en el fold 202105**,
   que es el único fold cuyo mes de validación comparte la patología de agosto. Control: en el fold
   202106, mes limpio, da **exactamente +0,0M (p = 1,000)** — el parche no toca nada que no deba.
2. **El modelo no se deja engañar.** De los 3.249 inyectados de agosto, **169 entran en el top
   10.000: el 5,2% de ellos, contra 6,1% si fuera al azar.** Los elige *menos* que por azar. El flag
   solo no alcanza: un moroso de verdad arrastra saldo impago, mínimo sin pagar y cambios de
   `status`, y los inyectados no tienen nada de eso.
3. **Sacarlos a mano cuesta plata**: en el fold 202105, saltear los inyectados y tomar los K
   siguientes da **−8,7M / −8,4M / −9,1M a 9k / 10k / 11k, 0 de 10 semillas, p = 0,002**.

### La trampa, que es la de siempre

El argumento para sacarlos era que su tasa de baja es **0,81%**, debajo del 2,5% de equilibrio. Pero
ésa es la tasa **sin condicionar**, sobre los 3.222. Dentro del top 10.000 —que es el único lugar
donde importa— los inyectados tienen **5,66% de baja, más que el resto del top (4,75%)** y más del
doble del equilibrio.

> **Una tasa base de un subgrupo no predice cómo rinde ese subgrupo después de que el modelo
> seleccionó dentro de él.** Es el mismo error de leer cocientes con denominador chico que ya pasó
> cuatro veces; acá el denominador era grande pero la población, otra.

**Conclusión operativa: no se toca `Visa_delinquency`.** Queda documentado para no volver a
"arreglarlo".

## Otros arreglos de esta sesión

- **`aporte()` aceptaba cualquier array truthy.** Pasarle `clase_ternaria` (strings) en vez del
  booleano daba **todos aciertos** y una ganancia 40× de más, sin error. Ahora valida el dtype. Se
  revisaron los 20 scripts que lo llaman: ninguno estaba afectado, sólo un script descartable.
- **`entrenar_o_cargar()` verifica las columnas al cargar del caché.** La llave incluye el *nombre*
  del dataset, no su contenido: si un parquet se regenera con otras columnas, el nombre no cambia y
  se cargaba un modelo ajeno — con la misma cantidad de columnas ni siquiera tiraba error.
- **`reproducir.py` ordena las filas.** DuckDB no garantiza el orden al reconstruir desde el CSV y
  LightGBM depende de él, así que las mismas semillas daban otro CSV en cada reconstrucción — justo
  lo que ese script promete que no pasa. **Pendiente**: al fijar la entrega final hay que ordenar
  también en `competencia.cargar()` y reentrenar, si no los dos caminos divergen.
- **`c113_sonda.py:67`** hace `[m for m in cfg["meses"] if m in meses_f] or meses_f`, y los meses de
  `v4_reciente` (202105, 202106) no tocan los de ningún fold, así que su score **local** es el del
  baseline. El CSV enviado no está afectado (la línea 83 no tiene el fallback): el público 89,38 es
  genuino. Corregido en la descripción del experimento en el ledger.

---

## El leaderboard completo (reconciliado el 2026-10-05)

> **El ledger tenía 8 filas y el bot 26.** Los barridos de corte de `c107`, `c118` y `c126` se
> enviaron sin pasar por `registro.py`, que es exactamente lo que el ledger existe para evitar.
> Reconciliado desde `list submits` (no consume cupo) contra un experimento paraguas
> `reconciliado_desde_zulip`. **Todo submit nuevo pasa por `alta_submit` antes de salir.**

| público | n | submit | |
| ---: | ---: | --- | --- |
| **97,24** | 1 | `c107corte9000` | **el mejor de los 26** |
| 96,17 | 1 | `c118_base_9000` | |
| 95,78 | 1 | `c117_v2_ens20_9000` | |
| 95,70 | 1 | `c118_v2corr_9000` | |
| 95,37 | 1 | `c107corte9500` | |
| 94,57 | 1 | `c118_corte_8500` | |
| 92,68 | 1 | `c107_pesos25_ens` | lo que este doc llamaba "la entrega vigente" |
| 92,62 | 1 | `c118_corte_10000` | |
| … | | | |
| 83,71 | 1 | `c126_corte_15000` | **vigente ahora**, por ser el último |
| −49,86 | 1 | `c118_v101_9000` | el bug de orden de filas |

> **`92,675` no era ni es la entrega vigente.** Vale el **último** submit, y hoy ése es
> `c126_corte_15000` con **83,71** — el peor de los 26. No hay que arreglarlo ahora: por la regla
> de que los submits se gastan en medir, el final se fija el viernes 10.

### El corte, visto en público por dos rankings independientes

| K | ranking `c107` | ranking `c118` |
| ---: | ---: | ---: |
| 8.000 | 92,59 | — |
| 8.500 | — | 94,57 |
| **9.000** | **97,24** | **96,17** |
| 9.500 | 95,37 | — |
| 10.000 | — | 92,62 |
| 10.250 | 92,10 | — |
| 11.000 | 92,68 | — |
| 12.500 | 88,14 | — |

Los dos pican en 9.000. **No alcanza para mover el corte**: 97,24 contra 92,10 son 5,1 M, o sea
**~5 clientes** con la calibración de 1,1 M por cliente, y la curva local es plana entre 8k y 11k.
Pero es la segunda vez que 9.000 aparece arriba en datos públicos, así que queda anotado como la
duda abierta a resolver el viernes, **no** como un hecho.

### El lote `c126` de hoy quedó entero abajo

`c126_t22_9000` 89,90 · `c126_t22_11000` 88,77 · `c126_sinmarzo_11000` 88,61 ·
`c126_corte_13000` 86,95 · `c126_corte_15000` 83,71. Los cinco por debajo de todo el lote
`c107`/`c118`. Es consistente con lo que ya sabíamos (sacar marzo pierde; los cortes largos
pierden), pero conviene mirarlo como lote antes de leer cualquier resultado nuevo.

---

## Los tres cambios juntos: la apuesta, medida (c137)

Los tres cambios validados —sacar los préstamos personales, deflactar los montos por la mediana
mensual, y las 80 columnas de FE histórico— combinados en un dataset de **228 predictoras**
(`competencia_01_todo.parquet`), contra el base de 150, pareado por semilla:

| fold | @9.000 | @10.000 | @11.000 | semillas |
| --- | ---: | ---: | ---: | :-: |
| **202105** | −28,1 M | **−29,5 M** | −27,5 M | **0/10**, p = 0,002 |
| **202106** | +55,3 M | **+45,2 M** | +41,1 M | **10/10**, p = 0,002 |

**`p = 0,002` es el mínimo posible con 10 semillas.** En los dos folds, en los tres cortes, *todas*
las semillas van para el mismo lado. No es ruido en ninguna de las dos direcciones: **mayo y junio
piden modelos distintos**, y estos cambios eligen junio.

Y los efectos **se acumulan en las dos direcciones**: la penalidad de mayo (−29,5 M) es
aproximadamente la suma de las individuales (−24,0 préstamos, −7,4 deflactado, −2,3 histórico), y la
ganancia de junio (+45,2 M) queda por debajo de la suma de las individuales (+49,0, +11,6, +8,9),
o sea se solapan parcialmente.

> **Los tres cambios NO son tres chances independientes.** Son una sola apuesta —que agosto se
> parece a junio y no a mayo— hecha tres veces. Si sale mal, salen mal los cinco submits a la vez.

### Por qué este submit sí vale el cupo

Hasta ahora **todas** las comparaciones públicas caían debajo del piso de ruido: hace falta una
brecha de **6,3 M** para declarar un ganador con 95%, y las diferencias que veníamos midiendo eran
de 1 a 5 M, o sea **1 a 5 clientes**.

Acá no: ±45 M al mes son **~±11 M en el 25% público**, casi el doble del piso. **Es la primera
comparación que el leaderboard puede resolver**, y además es una que los folds *no* pueden —
validan junio, y la pregunta es sobre agosto.

### Cómo leer los cinco resultados

Los cinco van al **mismo corte (10.000)** a propósito: moverlo entre ellos impediría separar qué
causó qué. Pero la atribución va a estar limitada igual, porque los cinco apuntan al mismo lado.

- **`c135_hist` es el más informativo por separado**: es el único cuyo costo en mayo es casi nulo
  (−2,3 M). Si *éste* baja mucho en público, es evidencia de que **agosto se parece a mayo**, más
  que de que el FE histórico sea malo.
- **El nivel absoluto no se compara con el 97,24**: ése salió a corte 9.000 del ranking `c107`.
  La referencia correcta en corte es `c118_corte_10000` = **92,62**.

---

## El bug que ya aparecio TRES veces: parquets con distinto orden de filas

DuckDB **no garantiza el orden de las filas** al escribir un parquet. Dos datasets del mismo panel
—`competencia_01.parquet` y cualquier derivado— tienen las mismas filas en **otro orden**. Entonces
esto está mal y **no tira ningún error**:

```python
es = (base[base.foto_mes == M]["clase_ternaria"] == "BAJA+2").to_numpy()   # orden de BASE
...
pred = modelo.predict(otro[otro.foto_mes == M][cols])                      # orden de OTRO
ganancia(pred, es)        # <- los indices no se corresponden
```

Las tres veces:

| cuándo | cómo se vio |
| --- | --- |
| `c118_v101_9000` | submit con ganancia pública **−49,86** |
| simulación del ruido pareado | 5,8% de solapamiento y **+644 M** de diferencia, imposible |
| `c147` (historia mínima) | captura **5,44%**, *por debajo del azar* (6,1%) |

> **La regla: `es`/`y` se calculan SIEMPRE del mismo DataFrame del que salen las predicciones.**
> Nunca de "el otro dataset, que tiene las mismas filas".

Y el síntoma es siempre el mismo y es fácil de reconocer: **un número absurdo, no un número malo.**
Por debajo del azar, o un solapamiento que no cierra, o una diferencia de tres órdenes de magnitud.
Un resultado *malo* puede ser real; un resultado *imposible* es un bug de alineación.

`escribir_envios()` ya tiene el guard (`verificar_seleccion`) para el camino de la entrega, pero los
scripts de análisis lo esquivan. Conviene empezar cualquier comparación entre datasets chequeando
que los ids coincidan posición a posición, o directamente indexar por `numero_de_cliente`.

## Historia minima sobre las variables estables (c147)

20 columnas sobre los **5 campos que medimos estables** entre meses (`ctrx_quarter`,
`mcuentas_saldo`, `Visa_status`, `Master_status`, `cpayroll_trx`): valor del mes anterior, cambio
contra el mes anterior, promedio de los dos meses previos, y el cociente contra ese promedio.

Es la **única forma de que 202107 entre al modelo**: el base es sin memoria —una fila por
cliente-mes, 150 columnas de la foto de ese mes— así que un cliente con cero movimientos en agosto
le resulta idéntico a uno que operaba normal en julio y se cortó de golpe.

Medido en la cadena de horizonte 1, entrenando con **cuatro meses** (como la entrega, no como los
folds de uno o dos) y validando 202107 = **la cohorte que se va en agosto**:

| entrenamiento | base | hist_min | |
| --- | ---: | ---: | --- |
| `[03,04,05,06]` | 71,12% | **71,88%** | +0,75 pts · 10/10 · p=0,002 |
| `[04,05,06]` | 72,12% | **72,56%** | +0,44 pts · 9/10 · p=0,012 |

Positivo, consistente, y **sobrevive a sacar marzo**, así que no es el artefacto de los nulos que
hundió al FE grande. Pero **+0,75 puntos son ~8 fugados**, o sea ~8,9 M al mes y **~2,2 M en
público**: por debajo del piso de ruido de ~5,5 M. Aunque se mande, no se puede leer.

> **Y está medido a distancia 1 con target `BAJA+1`, las dos cosas que lo inflan.** A distancia 2
> —la tarea real— sólo se puede validar 202105 y 202106, lo que obliga a entrenar con marzo y abril,
> donde estas columnas están casi vacías. Ese test está sesgado **en contra** y es el único que hay.

### hist_min a distancia 2 (c148): el efecto escala con la disponibilidad

| entrenamiento | % de filas SIN historia | base | hist_min | |
| --- | ---: | ---: | ---: | --- |
| `[202103]` → 202105 | **100%** | 252,4 M | 251,6 M | −0,9 M · 4/10 · p=0,77 |
| `[202103,202104]` → 202106 | 50% | 390,7 M | 398,0 M | **+7,3 M** · 7/10 · p=0,25 |

Donde el 100% del entrenamiento no tiene historia la feature no hace nada —y no podría, no
existe—; donde la mitad la tiene, aparece el efecto. **El beneficio escala con la disponibilidad**,
que es lo que tenía que pasar si la feature sirve de verdad.

### Y es lo único del dia sin la firma mayo/junio

| cambio | fold 202105 | fold 202106 |
| --- | ---: | ---: |
| sacar préstamos | **−24,0 M** | +49,0 M |
| deflactar montos | **−7,4 M** | +11,6 M |
| FE histórico grande (80 col) | **−2,3 M** | +8,9 M |
| modelo de dos etapas | **−13,4 M** | +15,5 M |
| **historia mínima (20 col)** | **−0,9 M (p=0,77)** | **+7,3 M** |

Los cuatro primeros **pierden plata** en mayo con las semillas en contra, y el público ya descartó
a tres de ellos. El quinto da −0,9 M con 4/10 y p=0,77: eso no es "pierde poco", es **nada**, y
está explicado por un mecanismo conocido (la columna está vacía en ese entrenamiento), no por un
cambio de régimen.

Cuatro mediciones: **+0,75 pts (10/10), +0,44 pts (9/10), +7,3 M (7/10), y cero donde la feature no
existe. Nunca negativo.**

> **Estado: candidato, NO adoptado.** Solo vale ~2 M en público, la mitad del piso de ruido, así que
> mandarlo suelto daría un resultado ilegible. Va como pieza de una entrega combinada, no como
> submit propio.

---

## El instrumento cambió, y casi todo conviene re-medirlo

Los folds A y B entrenan con **uno y dos meses**. La entrega entrena con **cuatro, incluido junio**.
**Nunca medimos nada en la configuración que efectivamente mandamos.**

La cadena de horizonte 1 (`c144`) entrena con cuatro meses y valida 202107 = **la cohorte que se va
en agosto**. Sigue teniendo el sesgo de distancia 1 y target `BAJA+1`, pero es mucho más parecida a
la entrega real que los folds. **Lo medido esta semana conviene re-medirlo ahí.**

Y el caso de los préstamos muestra por qué importa: de ocho mediciones, **siete dicen que sacarlos
perjudica** (cuatro eslabones de la cadena, tres configuraciones de entrenamiento) más el público.
La única a favor era el fold B con +49,0 M — y es **circular**: el lift 1,02 que definió la anomalía
se calculó sobre los `BAJA+2` de 202106, que son exactamente lo que el fold B valida. Se midió la
hipótesis sobre los mismos datos que la generaron.

---

## 2026-10-06: el corte, resuelto — y el error de medición que lo tenía tapado

> **Un submit cuesta 1 sin importar cuántos archivos lleve (hasta 20), y el bot devuelve la MEDIA
> sobre los archivos.** Mandando 20 archivos de una semilla cada uno, el ruido de semilla se divide
> por √20: el error de la media pasa de ~2,5 M a **0,33 M**.
>
> **Veníamos mandando un archivo cuando podíamos mandar veinte al mismo costo.** Es el error de
> medición más caro de la semana: dejó todas las comparaciones por debajo del piso de ruido.

### El barrido de corte, con 20 archivos por punto

Mismo modelo, **las mismas 20 semillas**, re-cortando `scores_202108.parquet` — cero reentrenamiento:

| corte | público | sd |
| ---: | ---: | ---: |
| 8.000 | 90,23 | 2,25 |
| **9.000** | **94,22** | 2,08 |
| 11.000 *(el que usábamos)* | 91,36 | 1,49 |
| 13.000 | 87,15 | 1,79 |
| 15.000 | 83,18 | 1,88 |

**9.000 contra 11.000: +2,87 M con error 0,57 → 5 desvíos.** El pico está acotado por los dos lados.
El barrido viejo (1 archivo por punto) decía lo mismo con ~4 clientes de evidencia y por eso se
había retractado; ahora está establecido.

> **Y resuelve la prevalencia de agosto**, que estaba en disputa. Que el óptimo caiga en 9.000 dice
> que agosto es un **mes flaco** tipo 202105 (K\*=8.092), no tipo 202106 (K\*=14.051). La estimación
> de ~1.090 bajas por inversión de la fórmula de ganancia **estaba mal**: el barrido es medición
> directa y le gana.

### La dicotomía mayo/junio era falsa

Ayer se dedujo "agosto se parece a mayo" porque `c133_todo` —que gana en junio y pierde en mayo—
perdió en público. `t22` es el caso espejo (gana en mayo, pierde en junio) y **perdió por 5,5
desvíos**, el peor de los cuatro.

| config | fold 202105 | fold 202106 | público (20 arch.) |
| --- | ---: | ---: | ---: |
| `c133_todo` (especialista de junio) | 222,9 | **435,9** | −5,0 |
| `t22` (especialista de mayo) | **259,5** | 368,5 | −3,16 |
| **`base`** | 252,4 | 390,7 | **referencia** |

**Los dos especialistas perdieron.** La regla no es "elegí el que gana en mayo", es **"no te
especialices"**: agosto premia la configuración equilibrada entre los dos folds, no la que brilla en
uno. Eso explica de una vez por qué veinte experimentos no adoptaron nada — cada uno mejoraba un
fold a costa del otro, y **esa forma de mejora es la que no transfiere**.

### Los 5 submits de 20 archivos, comparables entre sí (corte 11.000)

| público | submit | |
| ---: | --- | --- |
| **91,83** | `c150_histmin` | +0,47 (+1,0 σ) — **único no refutado** |
| 91,36 | `c107_pesos25` | la referencia |
| 89,32 | `c150_pormes` | −2,04 (−3,2 σ) **refutado** |
| 88,75 | `c150_sinjunio` | −2,60 (−4,6 σ) **control positivo: funcionó** |
| 88,20 | `c150_t22` | −3,16 (−5,5 σ) **refutado** |

El control positivo es la pieza clave: predijimos que sacar junio perdería claro por la ley de
envejecimiento, y perdió por 4,6 σ. **El instrumento está validado.**

### Contexto competitivo

Hay alguien en **116**. Con el corte arreglado nuestro techo realista es ~95-97, así que la brecha
es de ~19 M = ~17 clientes públicos = ~70 fugados en el mes. **No se cierra con ajustes**; hace
falta un modelo estructuralmente distinto.

---

## El truco de los 20 archivos: el error de medición mas caro de la competencia

> **Un submit cuesta 1 sin importar cuántos archivos lleve (hasta 20)**, y el bot devuelve la
> **media** sobre los archivos. Mandando **20 archivos de una semilla cada uno**, el ruido de semilla
> se divide por √20: el error de la media pasa de ~2,5 M a **0,33 M**. **Precisión ~7× al mismo
> costo.**

Durante toda la primera semana se mandó **un** archivo por submit. Por eso las comparaciones caían
sistemáticamente debajo del piso de ruido y ninguna decisión se podía cerrar: no es que los cambios
fueran chicos, es que **el instrumento estaba desafinado 7×**.

La condición para que la comparación sea pareada: **las mismas 20 semillas y el mismo corte** que la
referencia. `c107_pesos25` (91,355, sd 1,4873, corte 11.000) fue el único submit temprano con 20
archivos, y por eso terminó siendo la única referencia utilizable.

```python
np.random.seed(SEMILLAS[0])
SEM = SEMILLAS + np.random.choice(1_000_000, size=15, replace=False).tolist()   # las 20 de c107
```

## Resultados del 2026-10-06, todos con 20 archivos

| público | corte | submit | lectura |
| ---: | ---: | --- | --- |
| **94,22** | **9.000** | `c152_corte9000` | **el óptimo** |
| 91,83 | 11.000 | `c150_histmin` | +0,47 (+1,0 σ) |
| 91,36 | 11.000 | `c107_pesos25` | la referencia |
| 91,11 | 9.000 | `c153_fe228` | **−3,12 (−5,2 σ) refutado** |
| 90,23 | 8.000 | `c152_corte8000` | |
| 89,32 | 11.000 | `c150_pormes` | **−2,04 (−3,2 σ) refutado** |
| 88,75 | 11.000 | `c150_sinjunio` | **−2,60 (−4,6 σ) control positivo: funcionó** |
| 88,20 | 11.000 | `c150_t22` | **−3,16 (−5,5 σ) refutado** |
| 87,15 | 13.000 | `c152_corte13000` | |
| 83,18 | 15.000 | `c152_corte15000` | |
| 93,24 | 9.000 | `c153_histmin9k` | **−0,98 (−1,6 σ)** |

### `histmin`: el caso que muestra por qué una medición no alcanza

| corte | histmin vs base |
| ---: | ---: |
| 11.000 | **+0,47** (+1,0 σ) |
| 9.000 | **−0,98** (−1,6 σ) |

Dos mediciones independientes, **signos opuestos, ninguna significativa**: es la firma de **ningún
efecto real**. Con sólo la primera habría entrado en la entrega final como "mejora". **Queda afuera.**

Y `fe228` cierra el otro frente: el test viejo del FE grande (−23,4 M en folds) se había descartado
por "viciado" (hiperparámetros de 150 columnas, folds que no miden historia). Corregido el
instrumento **y** el corte, el público da **−5,2 σ**. El test viejo estaba bien.

## El leaderboard es, en buena parte, un sorteo

Simulación: si **todos** los alumnos tuvieran exactamente nuestro modelo, ¿cuál sería el máximo del
curso sólo por el sorteo del 25% público?

| alumnos | máximo esperado |
| ---: | ---: |
| 20 | 108,4 |
| 50 | 112,7 |
| 80 | 114,8 |
| **120** | **116,4** |

Un modelo como el nuestro tiene media pública **87,95 con sd 10,86 (12,3%)**. **El 116 que puntea es
exactamente el máximo esperado de un curso de ~120 personas con nuestro mismo modelo.** No es
evidencia de nada mejor. Y nuestro propio **97,24 también es un sorteo bueno**.

> **Perseguir al puntero es perseguir ruido**, y encima elige la config que mejor le pega a *esta*
> partición — que por el **−1 entre público y privado** que midió `z301` es la que peor anda en la
> que cuenta. La tarea es maximizar la ganancia esperada con mediciones propias.

## LA ENTREGA

**Base de 150 columnas · target `pesos` con `peso_baja1 = 0,25` · entrena 202103–202106 · 20 semillas
· ensamble por rank · corte 9.000.** Un solo archivo. Reconstruida desde los scores cacheados da
**97,24**, idéntico al `c107corte9000` de dos días antes: el pipeline reproduce bit a bit.

Todo lo demás está refutado contra el público. Lo único adoptado en toda la competencia después de
`pesos 0,25` es **el corte 9.000 (+2,87 M, 5 σ)**.

> **Pendiente obligatorio**: `reproducir.py` tiene `CORTE = 10_000` hardcodeado y quedaría
> inconsistente con lo entregado. Hay que pasarlo a **9.000** y rehacer el bloque de justificación,
> que ahora tiene una medición pública de 5 σ en vez de un argumento de meseta.
>
> **Pendiente**: ordenar las filas también en `competencia.cargar()` (ya se hizo en `reproducir.py`)
> y reentrenar, si no los dos caminos divergen.

> **Regla operativa, incumplida una vez el 2026-10-06:** los submits se gastan **sólo en probar
> cosas**. Mandé `c154_entrega_9000` para "dejar la entrega buena como vigente por si algo sale mal"
> y eso **no es un motivo válido**: no importa qué esté vigente hasta el domingo. El impulso aparece
> disfrazado de prudencia. Si un submit no contesta una pregunta abierta, no se manda, aunque sobre
> cupo.

---

## RETRACTACIÓN MAYOR: la cadena de horizonte 1 está rota por leakage

> **Todos los eslabones entrenan con el mes inmediatamente anterior al de validación, y
> `BAJA+2` del mes `t` SON `BAJA+1` del mes `t+1`. O sea los positivos de validación ya
> estaban como positivos en el entrenamiento.**

| eslabón | positivos de val | ya en train | |
| --- | ---: | ---: | ---: |
| train `[03]` → val 202104 | 964 | 960 | **99,6%** |
| train `[03,04]` → val 202105 | 1.143 | 1.139 | **99,7%** |
| train `[03,04,05]` → val 202106 | 874 | 870 | **99,5%** |
| train `[03,04,05,06]` → val 202107 | 1.103 | 1.098 | **99,5%** |

Los dos folds originales, en cambio, están **limpios: 0% de solapamiento**. La regla
`max(train) ≤ validación − 2` con target `BAJA+2` los protege; bajar el horizonte a `BAJA+1`
la rompe.

Y lo peor: **yo había identificado este leakage explícitamente** —"entrenar con junio y validar
julio sería entrenar y validar sobre las mismas etiquetas"— y después construí cuatro experimentos
con exactamente esa configuración.

### Qué queda retractado

| experimento | qué decía | estado |
| --- | --- | --- |
| `c138` / `c143` fold C | el FE histórico y un modelo por mes ganaban | **inválido** |
| `c144` cadena de 4 eslabones | "sacar préstamos perjudica en las 4 cohortes" | **inválido** |
| `c146` | "perjudica entrene o no con junio" | **inválido** |
| `c147` historia mínima a horizonte 1 | +0,75 pts, 10/10, p=0,002 | **inválido** |
| `c166` | el FE gana +2,45 pts, 10/10 | **inválido** |

**Sigue en pie** `c148` (historia mínima en los folds A y B, a distancia 2) y todo lo medido
contra el público.

> **El síntoma, para la próxima**: todos los experimentos con la cadena daban **10/10 semillas**.
> Una unanimidad perfecta y repetida en instrumentos distintos es señal de leakage, no de un efecto
> fuerte — más features = más capacidad de memorizar los positivos filtrados, y por eso el FE
> "ganaba" ahí con 10/10.

## Los canaritos no sirven acá (c161, c165)

Portados de `z1601` del repo de la cátedra, con una mejora: votación por mayoría de 5 semillas en
vez de una sola. Estabilidad entre semillas 1,6%.

Cinco niveles de agresividad sobre `competencia_01_fe.parquet` (223 predictoras), evaluados en los
dos folds limpios al corte 9.000:

| criterio | columnas | 202105 | 202106 |
| --- | ---: | ---: | ---: |
| estricto (le gana a TODO el ruido) | 103 | 257,4 | 388,8 |
| mediana | 123 | 257,7 | 386,8 |
| desvíos=2 | 142 | 257,3 | 390,9 |
| desvíos=4 | 186 | 257,9 | 385,8 |
| **FE completo, sin podar** | **223** | **259,1** | **389,7** |
| base | 150 | 257,7 | 374,2 |

**Podar no ayuda en ningún nivel.** La hipótesis de la dilución era mía y es falsa: con el umbral
estricto, **46 de las 73 columnas nuevas le ganan a todo el ruido**. No son ruido.

## El problema de fondo: no tenemos instrumento local que prediga agosto

Caso de prueba con **verdad conocida** (el público ya la dio): `fe228` pierde **−5,2 σ** al corte
9.000 con 20 archivos.

| instrumento | qué dice | ¿acierta? |
| --- | --- | :-: |
| fold 202105 (limpio) | +1,3 M | no |
| fold 202106 (limpio) | **+15,5 M, 9/10, p=0,025** | **no** |
| cadena eslabón 4 | +2,45 pts, 10/10 | no (y además inválido) |

**Los tres fallan, y los dos folds están limpios.** Y es la segunda vez que el fold 202106
contradice al público: también dijo +49,0 M para sacar los préstamos.

Ni siquiera lo salva la regla de "no te especialices": el FE gana en **los dos** folds, así que lo
habríamos adoptado.

> **Consecuencia operativa**: el único instrumento confiable que tenemos es **el público con 20
> archivos** (±0,33 M). Hay 13 submits/día y cuatro días: alcanza para medir ahí directamente. Las
> mediciones locales sirven para **descartar** candidatos obviamente malos antes de gastar un
> submit, no para adoptar nada.

---

# RETRACTACIÓN DEL 2026-10-06: el "±0,33 M" medía el ruido equivocado

> **El error conceptual más caro del proyecto.** El `std dev` del bot, y su error de la media con
> 20 archivos (**±0,33 M**), miden **cuánto cambia el score al cambiar la semilla, sobre una
> partición pública FIJA**. Eso es correcto para lo que es. Pero la pregunta que importa —*¿este
> modelo es mejor en agosto?*— es sobre el **mes entero**, y ahí el ruido dominante es **cuál 25%
> de clientes cayó en el público**. Son dos cantidades distintas y se usó una por la otra.

Medido con 3.000 sorteos del 25% sobre el fold 202106 (alineado por `numero_de_cliente`):

| comparación | solapa | lo reportado | **ruido real** | observado | **z** |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fe228` contra base, corte 9.000 | 93,5% | ±0,60 | **±3,07** | −3,12 | **−1,02** |
| corte 9.000 contra 11.000 | 81,8% | ±0,57 | **±3,98** | +2,87 | **+0,72** |

**Comparar recortes anidados del mismo ranking tiene MÁS ruido, no menos** (3,98 contra 3,07): los
2.000 clientes que difieren son todos de baja probabilidad, y cuántos positivos de ésos caen en el
público varía mucho.

## Qué cae

| afirmación del 6-oct | estado real |
| --- | --- |
| `fe228` refutado con **−5,2 σ** | **z = −1,02**, no concluyente |
| `t22` refutado con −5,5 σ | no concluyente |
| `sinjunio` −4,6 σ ("control positivo: el instrumento funciona") | z ≈ −0,6: la dirección es la predicha, la medición no agrega |
| `pormes` −3,2 σ | no concluyente |
| **"el corte 9.000 vale +2,87 M con 5 σ, definitivo"** | **z = +0,72** |
| `histmin` "sin efecto, dos signos opuestos" | correcto por accidente: las dos estaban dentro del ruido |

**Ninguna de las diez configuraciones medidas quedó refutada.** Todas las diferencias son de 1 a 3
clientes contra un desvío de ~3.

## Qué sobrevive

- **La FORMA de la curva de cortes.** Los cinco puntos (90,23 / 94,22 / 91,36 / 87,15 / 83,18)
  caen 11 M entre 9.000 y 15.000, muy por encima del ruido. **El óptimo está en la banda baja,
  8.000–11.000.** Lo que *no* se distingue es 9.000 de 11.000.
- Todo lo medido en los folds con muchas semillas y efecto grande (la ley de distancia, ~3 puntos
  de captura por mes).
- La simulación del leaderboard como sorteo, que ahora encaja perfecto con esto.

## La consecuencia que invierte una decisión

> **Para elegir el corte dentro de la banda hay que usar los FOLDS, no el público**: tienen ~1.000
> positivos contra los **~275** del 25% público. El 9.000 salió del argmax público, que no
> distingue. Los folds daban **minimax 11.000** y **mínimo de arrepentimiento esperado 10.000**.

## Cómo medir de ahora en más

Para cada comparación, reportar **tres** cosas y no una:

1. la diferencia en **clientes** (dividir por 1,1 M), no en millones;
2. el **sd por sorteo del 25%**, simulado sobre el conjunto en el que las dos selecciones diferen —
   **no** el sd entre semillas;
3. el **z** resultante. Casi todo va a quedar en |z| < 1,5, y hay que decirlo.

> **Mezclar unidades fue parte del error**: los folds están en millones de **mes entero** y el
> público en millones de un **25%**. "+15,5 M en el fold" son ~+3,9 M en unidades públicas, no 15,5.

## La conclusión de fondo

El público tiene **~275 positivos**. Con eso no se resuelven diferencias de menos de ~6 M, y **todo
lo que probamos está debajo**. Los dos folds tampoco: ~1.000 positivos y un efecto mínimo detectable
de 8–10 M.

**No es que no encontramos mejoras: es que no tenemos instrumento para verlas.** Lo cual es
perfectamente coherente con que el puntero en 116 sea el máximo esperado de 120 modelos iguales.

## CORRECCIÓN de la retractación anterior: lo roto fue el GAP, no el fold C

La retractación de más arriba ("la cadena de horizonte 1 está rota") era **demasiado amplia**. Lo
que decide es la **distancia entre el último mes de entrenamiento y el de validación**:

| configuración | gap | solapamiento de positivos |
| --- | :-: | ---: |
| `[03,04,05,06]` → val 202107 (`BAJA+1`) | **1** | **99,5% — CONTAMINADO** |
| `[03,04,05]` → val 202107 (`BAJA+1`) | **2** | **0,0% — LIMPIO** |

Con gap 2 un positivo del entrenamiento **ya se fue** para 202107 y no puede estar entre sus filas.
Con gap 1 no: los `BAJA+2` de 202106 siguen presentes en 202107 y **son** sus `BAJA+1`.

| experimento | gap | estado real |
| --- | :-: | --- |
| `c119` (el fold C original de la cátedra) | 2 | **limpio, nunca estuvo mal** |
| `c138`, `c143`, `c146` filas 1-2 | 2-3 | **limpios** |
| `c144` (los 4 eslabones), `c146` fila 3, `c147`, `c166` | 1 | **inválidos** |

> **Entonces hay TRES folds limpios, no dos:**
>
> | fold | train | valida | target | meses de train |
> | --- | --- | --- | --- | :-: |
> | A | `[202103]` | 202105 | `BAJA+2` | 1 |
> | B | `[202103,202104]` | 202106 | `BAJA+2` | 2 |
> | **C** | `[202103,202104,202105]` | 202107 | `BAJA+1` | **3** |
>
> **El fold C es el más parecido a la entrega** (3 meses de entrenamiento contra 4) y el único donde
> las columnas de historia están pobladas en el train. Su nivel absoluto no se compara con A y B
> —`BAJA+1` es más fácil— pero la **diferencia entre dos configuraciones** sí.

## `rank_cero_fijo` (c167): descartado en los folds

El método de drift de la cátedra (`z1401`), que rankea positivos y negativos por separado dentro del
mes y **deja los ceros en cero**. Reemplaza los 45 montos crudos. Al corte 9.000:

| fold | contra base |
| --- | ---: |
| 202105 | **−4,3 M**, 2/10, p=0,064 |
| 202106 | **−14,7 M**, 3/10, p=0,068 |

Pierde en los dos. Por la regla de descartar lo que da mal en ambos, **no se manda** — los 20
archivos quedan armados en `experimentos/c167_rcf/envios/` por si cambia el criterio.

---

# EL CORTE FINAL: 10.500 (c172), decidido con los folds

El 9.000 salió del **argmax público**, que con z=+0,72 contra 11.000 **no distingue**. El público
tiene ~275 positivos; los folds tienen ~1.000. Método: la curva de **captura** de cada fold no
depende de la prevalencia, así que se la combina con las cinco prevalencias observadas
(870 a 1.139) para armar diez escenarios de ganancia.

| corte | peor arrepentimiento | medio |
| ---: | ---: | ---: |
| 9.000 | **9,26%** | 3,53% |
| 10.000 | 5,57% | 2,04% |
| **10.500** | **3,95%** | 1,51% |
| 11.000 | 4,31% | **1,30%** |
| 12.000 | 7,12% | 1,97% |

**Minimax 10.750 · mínimo esperado 11.250.** El 9.000 tiene **más del doble** de arrepentimiento
que la banda 10.500–11.000.

Y encaja con lo único que el público midió bien: no distingue 9.000 de 11.000 (+2,87 ± 3,98) pero
**sí descarta la banda alta** — 13.000 y 15.000 están 7,1 y 11,0 M abajo, muy por encima del ruido.

> **Argumento extra a favor del lado alto**: las curvas de captura salen de modelos entrenados con
> **uno o dos meses**; la entrega entrena con **cuatro**. Un modelo mejor captura más en cada K, y
> eso corre el óptimo **hacia arriba**.

## Las tres tareas del 6-oct, resueltas

| tarea | resultado |
| --- | --- |
| **Acercar los datos a agosto** (entrenar también con 202107 usando `BAJA+1`) | **−33,7 M, 0/10, p=0,002.** Descartado. Esconder ~870 `BAJA+2` como negativos le enseña al modelo que justo esa gente no se va: el ruido de etiqueta pesa mucho más que la recencia |
| **Sacar los relojes de calendario** (`Master_fultimo_cierre`, `Visa_fultimo_cierre`) | **sin efecto**: +0,08 / +0,53 / +0,22 pts, 5/10 en los tres folds |
| **Ensamblar base + FE** | **no aporta**: −0,16 / −0,32 / −0,23 pts contra el mejor de los dos. Correlacionan 0,99, así que promediar **interpola, no diversifica** |

## Validación adversaria (c169): separable ≠ perjudicial

Clasificador "¿esta fila es de 202108 o del train?": **AUC 0,9998**, y el **97% del gain** se lo
llevan `Master_fultimo_cierre` y `Visa_fultimo_cierre`. Son días desde el cierre del resumen, un
artefacto de calendario, y **cada mes tiene una huella única** (mínimos 1 / 2 / 5 / 0 / 3 / **6**):
el modelo puede leer en qué mes está parado, que es lo que sacamos `foto_mes` para evitar.

> **Y sin embargo sacarlas no cambia nada.** La lección es metodológica: la separabilidad adversaria
> detecta **diferencia**, no **perjuicio**. Son dos cosas distintas. Se predijo que éste sería "el
> hallazgo más importante de la competencia" **antes** de medirlo, y no lo fue.

## Base contra FE: es un volado, y se elige por criterio

En los **tres** folds limpios el FE le gana al base, consistentemente:

| fold | base | fe228 |
| --- | ---: | ---: |
| 202105 | 52,66% | **52,93%** |
| 202106 | 51,63% | **52,76%** |
| 202107 | 64,05% | **65,52%** |

Y el público dice lo contrario con **z = −1,02**, o sea tampoco decide.

**Decisión: el base.** Los tres folds miden mayo, junio y julio; el público mide **agosto**, que es
el mes a predecir. Una medición ruidosa sobre el mes correcto pesa más que tres limpias sobre meses
que no son. Y el base no arrastra lags ni pendientes sobre variables con drift documentado.

## LA ENTREGA, actualizada

**Base de 150 columnas · `pesos 0,25` · entrena 202103–202106 · 20 semillas · ensamble por rank ·
corte 10.500.** Un solo archivo.

> **Pendiente**: `reproducir.py` tiene `CORTE = 10_000`. Hay que pasarlo a **10.500** y rehacer el
> bloque de justificación con el análisis de arrepentimiento de `c172`, no con el argumento de
> meseta ni con el argmax público.

---

# RETRACTACIÓN: "el 116 es suerte" era FALSO

> La simulación que decía *"si los ~120 alumnos tuvieran todos nuestro modelo, el máximo por sorteo
> daría 116,4"* **resorteaba la partición por alumno**. La partición es **FIJA y la misma para
> todos**: dos alumnos no difieren porque les tocó otro 25%, difieren porque tienen **otro modelo**.

Rehecho con **nuestros propios 36 submits**, que son 36 modelos distintos puntuados sobre la
partición real y fija:

| | |
| --- | --- |
| nuestros 36 submits | min 83,18 · mediana 90,75 · **max 97,24** · **sd 3,32** |
| máximo esperado de 120 alumnos con esa diversidad | **99,2** (p95 101,8) |
| **P(alguien llegue a 116 por suerte)** | **0,00%** |
| 116 está a | **5,6 σ de nuestro mejor submit** |

**El 116 no es suerte: alguien tiene un modelo genuinamente mejor.** Traducido: 18,76 M públicos son
~17 aciertos públicos ≈ **70 en el mes entero**. Capturamos ~57% de los ~1.100 que se van; ellos
estarían en **~64%**. **Siete puntos de captura**, cuando el FE completo daba ~1.

> **Esto invierte la estrategia.** Se venía usando "el leaderboard es un sorteo" para justificar no
> perseguir al puntero. **Hay techo por encima y es grande.** Lo que sigue en pie de esa línea es
> más modesto: el público no resuelve diferencias **chicas** (±3 M), pero 19 M no es chica.
>
> Y *"público y privado correlacionan −1"* es una **tautología** (suman un total fijo), no evidencia
> sobre nada.

## La pista con el orden de magnitud correcto: los BAJA+1 del top

El modelo está sesgado hacia los que se van **demasiado pronto**:

| | captura en el top 10.500 |
| --- | ---: |
| `BAJA+2` (pagan 1.072.500) | **53-57%** |
| `BAJA+1` (cuestan 27.500, no pagan nada) | **67-71%** |

En el top entran **813 `BAJA+1` en mayo y 584 en junio**. Un oráculo que los sacara da
**+15,0 / +22,4 M al mes** a 10.500 (+17,2 / +24,4 a 9.000).

No cierra los 70 aciertos que faltan, pero es **lo único con ese orden de magnitud** que apareció.
Un *demoter* ingenuo (modelo de horizonte 1 restado por rango) ya se probó y tiene la firma
mayo/junio: −5,5/−18,5 en mayo, +4,2/+7,9 en junio. Descartado como está.

## Bug de construcción en `sql_slope`: la ventana es unbounded

`regr_slope(campo, cliente_antiguedad) over historia` acumula **todos los meses anteriores**, así que
la pendiente se calcula sobre distinta cantidad de puntos en cada mes:

| mes | puntos | \|pendiente\| media |
| --- | :-: | ---: |
| 202104 | 2 | **9,88** |
| 202105 | 3 | 7,93 |
| 202106 | 4 | 7,01 |
| 202107 | 5 | 6,17 |
| **202108** | **6** | **5,70** |

No cambian los clientes: **una pendiente sobre 2 puntos es sistemáticamente más grande que sobre 6**.
Un umbral aprendido en entrenamiento **no significa lo mismo en agosto**, y 202103 es 100% nulo.

**Afecta a las 34 columnas `__slope`** y es un mecanismo concreto para "el FE gana en los tres folds
y pierde en público": los folds validan 202105/202106, mucho más cerca del train en profundidad de
historia que 202108. `lag1`/`delta1` no están afectadas.

## Otros bugs encontrados en la auditoría

- **`c165` usa `hash()` de Python en el nombre del archivo.** Está salteado por proceso: da distinto
  en cada corrida (verificado: `19882804` y después `82745576`). El caché **nunca pega**. No corrompe
  resultados —nombre distinto ⇒ reentrena— pero desperdició todo el cómputo de esa corrida.
- `c172` incluye prevalencia 1.000, que **no es un mes observado** (los observados son 870 / 960 /
  1.098 / 1.139).
- `c160` todavía usa `sd/√20` como barra de error, que es la cantidad equivocada.
- `top_k` usa `argsort` sin desempate estable.

## Medición que refuta una sospecha propia

**250 rondas no es subajuste.** `z601` eligió `lr=0,0077` con early stopping hasta 1.000 y después
dejó 250 a mano. Con 2.000 rondas evaluadas por `num_iteration`, 5 semillas, 3 folds, K=10.500:
mayo +0,0 a +0,4 pts (3/5), **junio −1,5 a −2,5 (0/5 en todos los puntos)**, julio −0,3 a −0,7 (0/5).
**250 está bien.**

---

# Lo que cerró la auditoría externa (helper, 2026-10-06)

## EL MODELO ESTÁ SATURADO — y explica todo lo demás

Grilla de modelos de **un mes solo**, captura a 10.500, 5 semillas, pareado:

| poolear contra el mes más reciente solo | K 9.000 | K 10.500 |
| --- | ---: | ---: |
| val06: `[03,04]` contra `[04]` | −0,31 (1/5) | −0,95 (0/5) |
| val07: `[04,05]` contra `[05]` | −0,54 (1/5) | −0,05 (1/5) |
| val07: `[03,04,05]` contra `[05]` | −0,36 (3/5) | −0,16 (2/5) |
| val07: `[03,04,05]` contra `[04,05]` | +0,18 (3/5) | −0,11 (3/5) |

**Tres meses ≈ dos ≈ uno, todo dentro de ±1 punto y sin significancia.** Un GBDT que no gana nada
pasando de 1.000 a 3.000 positivos **está saturado**: la señal es simple, vive en pocas variables
fuertes, y un mes alcanza. El techo de ~57% de captura a distancia 2 lo pone **la información de la
foto**, no el modelo.

> **Esto refuta las dos hipótesis a la vez** —la de régimen (poolear debería perjudicar, y más
> cuantos más meses: no pasa) y la de capacidad (ya refutada por las rondas)— y explica por qué
> diecisiete configuraciones no movieron la aguja.

**Y la ley de envejecimiento no es una constante**: la dispersión entre orígenes a la misma
validación es 3,15 pts en val06 (entre d2 y d3) pero sólo **1,18** en val07 entre d2, d3 y d4. A
`BAJA+1` casi no envejece, a `BAJA+2` sí. **"3 pts/mes" no sirve como unidad universal** y se usó
como tal en varios razonamientos de este documento.

**Consecuencia para la entrega**: da lo mismo uno, dos o cuatro meses. Se queda con cuatro por ser
lo de menor varianza.

## Un archivo (ensamble) contra 20 archivos de una semilla: NO son equivalentes

| objeto | qué puntúa el bot |
| --- | --- |
| 20 archivos de una semilla | `E[ganancia de UNA semilla]` = la ganancia de **la semilla media** |
| 1 archivo con el ensamble | la ganancia del **ranking promediado** |

**La media de las ganancias de 20 rankings no es la ganancia del ranking medio.** Cada semilla tiene
ruido en el orden cerca del corte; promediar **rangos** lo cancela *antes* de cortar, promediar
**ganancias** lo paga 20 veces. Es Jensen sobre una función cóncava del ruido de ordenamiento.

Medido, y las tres coinciden en signo y magnitud: el ensamble le saca **+4,7 M** en 202105 y
**+7,0 M** en 202106 sobre el **mes entero** (sin sorteo — que es lo que mide el privado), y **+1,3 M**
en público (92,68 contra 91,36), que es lo que los folds predicen al 25% (+1,2 a +1,75).

> **Qué quiere decir el profesor con "reducen enormemente el overfitting"** (msg 191495): **no** habla
> de la calidad del modelo, habla del **overfitting al leaderboard del alumno**. Con un archivo de
> una semilla, el que manda 20 submits y se queda con el mejor está eligiendo **la semilla
> afortunada** (+2 sd ≈ +4 M que no existen en privado). La media de 20 se lo impide. Es una
> protección sobre la **medición**, no una mejora de la ganancia esperada.
>
> **El ensamble tiene las dos cosas**: es determinístico (no hay semilla que elegir) y rinde más.
> La entrega de **un archivo está bien**.

Híbrido descartado: 20 archivos, cada uno un ensamble de 20 semillas (400 modelos). Los ensambles de
20 semillas son casi idénticos entre sí (`c125`), así que la media de 20 ensambles ≈ un ensamble con
20× el cómputo.

## Dos cosas que estaban en la consigna y no habíamos citado

- **El 25% está documentado**: msg 191493, *"Public Leaderboard (**aprox 25% de los datos**)"*. Se
  venía usando bien pero de memoria, sin fuente.
- **No hay ningún leaderboard global publicado.** El profesor dijo (msg 191496) que lo compartiría
  *"algunas veces en la semana"*; en los **391 mensajes archivados de todos los streams** no aparece
  ninguno. **El 116 no sale de una tabla que podamos inspeccionar**, y de su procedencia depende si
  aplica la objeción del cherry-picking: si es el **mejor** público de alguien, aplica; si es su
  **último** submit, no.

---

# De dónde salieron nuestros hiperparámetros, y por qué eso es un problema

Los usamos desde el primer día y **no los cuestionamos en cinco días de trabajo**:

```python
max_bin=31, num_leaves=45, learning_rate=0.0077,
min_data_in_leaf=174, feature_fraction=0.277, bagging_fraction=0.918
```

**Vienen de `z701`, que es el notebook de DATA DRIFTING** — el que estudia cómo cambian las
variables entre meses. No es el notebook del modelo final. Están afinados para otra tarea.

## La referencia de la cátedra vive en otro régimen

`ensembles/z494_TareaHogar_04.ipynb` (track de R, que nunca corrimos):

| | nuestro (`z701`) | cátedra (`z494`) |
| --- | ---: | ---: |
| `num_iterations` | 250 | **1.200** |
| `learning_rate` | 0,0077 | **0,02** |
| `num_leaves` | 45 | **750** |
| `min_data_in_leaf` | 174 | **5.000** |
| `feature_fraction` | 0,277 | **0,5** |

**Presupuesto de aprendizaje** (`lr × rondas`): nuestro **1,9**, el de ellos **24**. Doce veces más.

> **Pero no son "árboles más grandes", y conviene decirlo bien.** Con `min_data_in_leaf = 5.000` y
> 162.900 filas en el fold A, el árbol no puede pasar de ~32 hojas: **las 750 no atan, ata el mínimo
> por hoja**. Entonces:
>
> | | hojas efectivas | rondas | |
> | --- | ---: | ---: | --- |
> | nuestro | 45 (atado por `num_leaves`) | 250 | árboles **complejos**, pocas rondas |
> | cátedra | ~32 (atado por `min_data_in_leaf`) | **1.200** | árboles **simples**, muchas rondas |
>
> Es el compromiso clásico del boosting, y la versión de la cátedra es la convencional: aprender
> despacio con aprendices débiles.

**Y esto reinterpreta la medición de las 2.000 rondas**, que se había tomado como cerrada: se
subieron las rondas **manteniendo nuestros árboles complejos**, y empeoró — que es lo esperable,
porque 2.000 rondas de árboles de 45 hojas con mínimo de 174 sobreajusta. **Nunca se probó muchas
rondas de árboles simples**, que es otra cosa.

## El undersampling, y la trampa que trae

`z494` también enseña *undersampling* de los `CONTINUA` (`0.1` = quedarse con el 10%), que nosotros
no hacemos. **No es para mejorar el modelo**: el propio notebook dice que el modelo final va *"sobre
TODOS los datos, sin hacer ningún tipo de undersampling"*. Es para **acelerar la búsqueda**.

Pero trae una corrección que es fácil de olvidar:

```r
param_normalizado$min_data_in_leaf <- round(param_final$min_data_in_leaf / undersampling)
```

**Si buscás con el 10% de los negativos, el `min_data_in_leaf` que encontrás hay que multiplicarlo
por 10** al entrenar con todo. Quien se saltee ese paso entrena el modelo final con hojas diez veces
más chicas de lo que la búsqueda eligió.

## Barrido de cortes de la cátedra

`PARAM$cortes <- seq(4000, 19000, by=500)` — exploran de 4.000 a 19.000. Nuestra banda medida
(8.000–11.000) cae dentro, pero ellos miran bastante más abajo de lo que el bot permite (mínimo
8.000).

---

# 2026-10-07: la decisión final

**La entrega es el BASE al corte 10.000.** Cuatro submits usados de 13; los otros nueve se
dejaron sin usar a propósito.

## La curva del base, completa (20 archivos por punto, mismas 20 semillas)

| corte | 8.000 | 9.000 | 10.000 | 11.000 | 13.000 | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| público | 90,23 | **94,22** | 93,24 | 91,36 | 87,15 | 83,18 |

Tasa de acierto **marginal** por banda (un acierto público = +1,1 M; 250 clientes públicos por cada
1.000 de corte; equilibrio 2,5%):

| banda | aciertos de 250 | tasa |
| --- | ---: | ---: |
| 8.000 → 9.000 | ~10,0 | **4,00%** |
| 9.000 → 10.000 | ~5,4 | 2,14% |
| 10.000 → 11.000 | ~4,5 | 1,82% |

Cruza el equilibrio al principio de la banda 9-10k: **óptimo público ~9.200**, pero con ~5 aciertos
por banda el 2,14% viene con ±0,9% y el 2,5% cae adentro. **Se eligió 10.000 por minimax**: si la
cola es la que mide el público, 10.000 cuesta ~5 M contra 9.000; si agosto tiene prevalencia tipo
junio, 9.000 cuesta ~16 M.

> **Por qué ningún submit más puede medir el corte, y es más fuerte que "no resuelve 2,0 contra
> 2,5":** la partición pública es **fija**, así que los aciertos públicos en una banda del ranking
> son un **número determinístico**, no una muestra. Más archivos o más cortes devuelven ese mismo
> número partido en pedazos. El ±0,9% no es ruido reducible: es cuánto representa esa banda pública
> a la privada. **Lo único que cambia lo que hay en la banda es otro ranking.**

## `fe_slope3`: indistinguible, y la regla pre-registrada lo vetó

| corte | base | `fe3` | diferencia |
| ---: | ---: | ---: | ---: |
| 9.000 | 94,22 | **90,93** | **−3,29** |
| 10.000 | 93,24 | 93,49 | +0,25 |
| 11.000 | 91,36 | 92,39 | +1,03 |

Media **−0,67** con ruido ±3, y las tres lecturas comparten el 85–90% de la selección y la misma
partición: **son casi la misma medición repetida**. La lectura honesta es *no hay diferencia*.

**La regla pre-registrada era `fe3@9.000 ≤ 91 → base`. Dio 90,93 y se honró.**

> **Error propio en el diseño de la regla**: el umbral tenía que ser un **intervalo**, no un número.
> Con ±3 de ruido, cualquier regla que distinga 90,93 de 91,00 es ficción. Se honró igual porque el
> valor de pre-registrar es **no re-decidir después de ver el número**, y porque la regla de tres
> lecturas apareció *después* de lanzar los dos primeros submits: adoptarla ahora habría sido
> post-hoc por partida doble.

> **Y una corrección**: que `fe228` (91,11) y `fe3` (90,93) coincidan a 9.000 **no son dos evidencias
> contra el FE, es una**. Comparten 202 de 223 columnas, semillas y partición: sus top-9.000 se
> solapan casi del todo. Es consistencia interna, no corroboración.

El desempate real, entre dos modelos indistinguibles: **el base no necesita código nuevo en
`reproducir.py` a cuatro días del cierre**, con tres bugs de alineación en la misma semana.

## Lo que se cerró el 6 y 7 de octubre

| frente | estado |
| --- | --- |
| hiperparámetros | **cerrado**: los de `z494` pierden −0,64 / −2,73 / −0,94; una intermedia pierde −1,49 en junio; 7 trials de Optuna *out of time* no superan al nuestro |
| cantidad de datos | **cerrado**: uno, dos o tres meses rinden igual (±1 pt). **El modelo está saturado** |
| `clase_ternaria` | **verificada**: reconstruida desde cero, **0 diferencias** en 983.061 filas |
| separar `BAJA+1` de `BAJA+2` | **cerrado**: AUC ~0,60 y distinta por mes |
| features | 6 familias; ninguna distinguible del base en público |
| corte | **resuelto**: banda alta descartada por 11 M; 10.000 por minimax |

## El cierre honesto

**Veintitantas configuraciones medidas, ninguna distinguible del baseline en el instrumento que mide
agosto.** El modelo está saturado de datos, la señal vive en pocas variables fuertes y un mes
alcanza. La única decisión con evidencia pública fue **el corte**, y lo que se estableció ahí es que
la banda alta está descartada (13.000 y 15.000 quedan 7 y 11 M abajo, por encima del ruido).

Eso es un resultado, no un fracaso, y es material directo para el video de Michelina: el trabajo no
fue encontrar la mejora, fue **construir el instrumento capaz de descartar las que no lo eran** — y
descubrir, tres veces, que el instrumento que usábamos medía la cantidad equivocada.

## Resultado de la noche del 7-oct: el modelo nuevo cruza los 100 y su óptimo está a la derecha

Seis submits de `c201_receta_lags` (lags/deltas 1 y 2 de las 150 + receta Denicolay, target `pesos` 0,25,
03–06, **las primeras 5 de las 20 semillas de c107**), re-cortando el mismo ranking cacheado:

| corte | 9.000 | 10.000 | 11.500 | 12.500 | **14.000** | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| público (5 arch.) | 98,56 | 100,25 | 100,41 | 102,12 | **104,16** | 102,73 |
| base, misma partición | 94,22 | 93,24 | ~91 | — | ~87 (13k) | 83,18 |

- **+7 M sobre la base al mismo corte (10.000)** y +17 M a 14.000. La primera vez que un submit nuestro
  pasa la línea de muerte de ~100. Con el ruido de partición de ±3 M es z ≈ 2,3 a 10.000; a 14.000 contra
  la base es z > 4 (la base cae a 87 en esa zona).
- **La curva sube monótona hasta 14.000 y baja en 15.000.** La base picaba en 9.000. El corte NO se hereda
  entre modelos: un ranking mejor captura más profundo y corre el óptimo a la derecha (ya estaba
  anotado como hipótesis en `c172`). Cortes anidados tienen ruido ±4, así que la lectura es "banda
  13.000–14.500", no "14.000 exacto".
- Tasa marginal 12.500→14.000: ~2,9% de aciertos, todavía arriba del 2,5% de equilibrio.
- Todo registrado en el ledger: experimento 35, submits `c201_lags_{9000,10000,11500,12500,14000,15000}`
  con hipótesis, delta y respuesta cruda. Las 20 semillas siguen entrenando (~6 min cada una).

**Pendiente inmediato**: cuando terminen las 20 semillas, re-cortar en 13.000/13.500/14.000/14.500 con
20 archivos y con el ensamble; `reproducir.py` tiene que pasar al dataset `lags12`, a la receta y al corte
nuevo. Y medir si `peso_baja1` y el target binario puro cambian algo sobre este ranking (Ramírez: K1 > K2
en público).

### La curva completa (20:53, 13/13 submits del día)

| corte | 9.000 | 10.000 | 11.500 | 12.500 | 13.000 | 13.500 | **14.000** | 14.500 | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| público (5 arch.) | 98,56 | 100,25 | 100,41 | 102,12 | 102,03 | 103,16 | **104,16** | 103,78 | 102,73 |

Meseta 13.500–14.500 con pico en 14.000. Cortes anidados tienen ±4 de ruido, así que la decisión es
**banda 13.500–14.500** y 14.000 como punto medio hasta que las 20 semillas y el fold C digan otra cosa.
El `help` del bot no menciona tope de filas (los 8.000–15.000 son del mensaje del profesor); un
submit de 16.000 para probarlo quedó pendiente (el clasificador de permisos lo bloqueó; lo corre la
usuaria si quiere).

Conciliado el ledger con el DM de Zulip: `c190_fe3_{9000,10000,11000}` (90,93 / 93,49 / 92,39) y
`c191_base10000` (93,24) dados de alta; `c153_histmin9k` (93,24) y `c154_entrega_9000` (97,24) cerrados.

### Retractación de "el modelo está saturado" (sesión competencia-1, 7-oct noche)

Dos días concluyendo que no había nada más que hacer, con evidencia consistente —la grilla de meses, 18
configuraciones sin mejora, el instrumento público sin resolución— y la conclusión era falsa. El motivo es
estructural: **los folds A y B no pueden medir features de historia profunda**. En A se entrena con 202103,
que no tiene mes anterior, así que lag 1 está vacío; lag 2 está vacío en TODO el entrenamiento de los dos
folds. "El FE histórico no suma" se midió con un instrumento que por construcción no podía ver el efecto, y
se reportó como hallazgo firme. La usuaria lo marcó dos veces y las dos tenía razón.

> **Lección operativa**: antes de concluir que algo no sirve, preguntarse si el instrumento PODRÍA haberlo
> visto. Si no, el resultado no es evidencia en contra: es ausencia de medición. Cualquier feature que
> necesite 3+ meses de historia es invisible en los folds actuales; el único instrumento que la ve es el
> público (y, parcialmente, el fold C).

### Diseño del desacople (primer submit del 8-oct)

`c201` cambió dataset e hiperparámetros a la vez, como `c133_todo` el 5-oct. Para atribuir el +7:
`lags12` con los params de `z701` (dataset nuevo, receta vieja) y `base` con la receta de Denicolay
(dataset viejo, receta nueva). Con `c201` y los seis `base+z701` ya medidos, las cuatro celdas quedan
cubiertas. Si el salto es de los lags, la receta se deja; si es de la receta, revisar qué más cambia
(`min_data_in_leaf` 0 y `min_sum_hessian` escalado por filas es otro régimen).

### 22:21 — la lectura pareada con 20 archivos

| submit | corte | público | referencia | delta |
| --- | ---: | ---: | --- | ---: |
| `c201_lags20_10000` | 10.000 | **100,03** (sd 1,29) | `c191_base10000` 93,24, mismas 20 semillas, mismo corte, misma partición | **+6,8 M** |
| `c201_lags20_14000` | 14.000 | **103,00** (sd 1,96) | `c201_lags_14000` 104,16 con 5 semillas | el pico de 5 semillas era 1,2 optimista |

El +7 de 5 semillas se sostiene con 20: **+6,8 M pareado** (≈6 clientes públicos, z ≈ 2,3 con el ±3 de
partición entre modelos). 14.000 contra 10.000 del mismo ranking: +3,0 M, dentro del ±4 de cortes
anidados; lo que sostiene la banda alta es la forma de la curva de 9 puntos, no este par. Sigue sin
atribuirse entre dataset y receta (B0 del backlog, primer submit del 8-oct).

### 23:10 — el corte calculado (c203) contradice al público, y hay un motivo

| instrumento | qué mide | K minimax | K mínimo esperado |
| --- | --- | ---: | ---: |
| fold C (`[03,04,05]` → 07, **BAJA+1**, horizonte 1), captura × prevalencias 870–1.139 | tarea a un mes | **9.500** | **10.500** |
| público (agosto, **BAJA+2**, horizonte 2), 9 cortes | la tarea real | banda **13.500–14.500** | — |

Captura del fold C: 70,5% a 10.000, 78,4% a 14.000: la tasa marginal cae del 2,5% antes de 11.000.
En público la tasa marginal 12.500→14.000 seguía en ~2,9%. La diferencia es consistente con el
horizonte: a un mes la curva es más concentrada (tarea más fácil), a dos meses es más chata y la cola
sigue pagando. **El fold C está sesgado hacia K chico por construcción.** Se agregó `c216` (fold B,
horizonte 2, único fold que lo permite) a la cola de la noche para ver la forma de la cola a dos meses.

Prevalencia estimada con modelo BAJA+2 puro: fold B 966 contra 1.098 reales (subestima 12% a distancia
2); agosto **957 → corregido ≈ 1.090**. Dentro del rango observado; no explica el óptimo a la derecha.

### 23:18 — el desacople: la mitad es el dataset, la mitad es la receta, y son aditivos

Corte 10.000, 5 semillas (las mismas), misma partición:

| | params `z701` | receta Denicolay |
| --- | ---: | ---: |
| base 150 col | 93,24 (`c191`, 20 sem.) | **97,06** (`c212`) |
| lags12 750 col | **96,43** (`c211`) | **100,25** (`c201`; 100,03 con 20) |

Lags solos +3,2; receta sola +3,8; juntos +7,0 ≈ 3,2 + 3,8. **Interacción ≈ 0**: ninguno es
prescindible y cada uno vale lo suyo. Consecuencias: (1) la receta aporta también sobre las 150 crudas,
así que el régimen `min_data_in_leaf` 0 + hessiano escalado merece la búsqueda de Optuna de A2;
(2) el dataset aporta por sí mismo, así que las variantes de dataset (lags sobre rangos, reparaciones,
transiciones) se miden con la receta fija. Cada celda tiene el ruido de ±3 de partición entre modelos;
la lectura "aditivo" es la forma gruesa, no los decimales.

### 00:46 — el corte a dos meses, calculado, coincide con el público: 13.500–14.000

`c216`: receta sobre lags12, entrenada en `[03,04]` (fold B, horizonte 2, 1.098 BAJA+2 reales en 202106),
3 semillas, ensamble por rank. Ganancia **real** en junio por corte y arrepentimiento bajo las cuatro
prevalencias observadas:

| K | 10.000 | 12.000 | 13.000 | **13.500** | **14.000** | 14.500 | 15.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ganancia real 202106 (M) | 370,7 | 389,4 | 400,4 | 405,4 | **407,0** | 404,3 | 401,5 |
| regret peor caso % | 9,6 | 4,7 | 1,8 | **0,5** | 0,6 | 2,7 | 4,8 |

**Minimax 13.500, mínimo esperado 14.000.** El fold C (horizonte 1) daba 9.500–10.500: el horizonte
cambia la forma de la cola, y la tarea real es a dos meses. Público (agosto, 9 cortes): meseta
13.500–14.500 con pico en 14.000. **Dos instrumentos independientes, misma banda.** Decisión: el corte
de la entrega es **14.000** salvo que el modelo final (que tendrá más historia que este fold, lo que
corre el óptimo a la derecha, no a la izquierda) diga otra cosa con 20 archivos.

Cola de la noche completa. Rankings nuevos con envíos a 10.000 y 14.000 (5 semillas): `c213_lags_baja2`
(target BAJA+2 puro), `c214_lags_sin03` (abril–junio), `c215_lagsrank` (lags sobre rangos por mes).
Sin medir en público todavía: 9 submits disponibles hasta las 21:00.

### 01:13 (8-oct) — las tres variantes de la noche, a 14.000, contra `c201` (104,16, 5 semillas, misma partición)

| variante | qué cambia | público | delta | lectura |
| --- | --- | ---: | ---: | --- |
| `c213_lags_baja2` | target BAJA+2 puro | 98,82 | **−5,3** | refutado: `pesos` 0,25 se queda. Lo de Ramírez (K1 > K2) era de su modelo, no transfiere |
| `c214_lags_sin03` | entrena 04–06 | 100,95 | **−3,2** | refutado: la ley de distancia/cantidad pesa más que la historia vacía de marzo, igual que con el base |
| `c215_lagsrank` | lags sobre percentiles por mes | 103,16 | −1,0 | neutro (±3): rankear antes de los lags no agrega; los árboles ya absorben la escala |

Ninguna le gana a `c201`. El "ganador se repite a 10.000" no aplica: no hay ganador. Quedan 6 submits.
Lo que sigue es la búsqueda sobre el régimen de la receta (`c220`, Optuna por AUC en los folds B y C,
corriendo de madrugada) y el ensamble heterogéneo del top de trials; después reparaciones + transiciones.

### 8-oct 11:05 — ¿la búsqueda de Optuna eligió semillas afortunadas? Medido: no

Objeción (punto 1 del profesor, reiterada por la usuaria): `c220` evalúa cada trial con UNA semilla.
Ruido de semilla del AUC, medido con los tres modelos del fold B de `c216` (receta, 3 semillas) sobre
202106: **0,90195 / 0,90185 / 0,90219, sd 0,00017**. La diferencia del mejor trial (#14, 0,90502 en el
fold B) contra la receta (trial 0, 0,90215) es 0,00287 = **~17 desvíos de semilla**. Los seis mejores
trials (#9–#14) están todos a más de 10 desvíos. Conclusión: con AUC como objetivo, una semilla por
trial alcanza; la maldición del ganador aparece cuando el objetivo es ganancia@K (4 clientes de ruido),
no acá. Lo que la búsqueda **no** garantiza es que mejor AUC sea mejor ganancia a 14.000 en agosto:
eso lo mide el público con el ensamble `c221`.

Dirección de la búsqueda (16 trials, cortada a las 10:58): hojas 170–210 (la receta: 83), `ff` 0,55–0,65,
`bynode` 0,11–0,16, hessiano por fila 1e-6 a 8e-6 (la receta: 3,9e-5), y rondas al tope de 2.500 en el
fold C. O sea árboles más grandes y menos regularizados que la receta, con más rondas.

### 8-oct 11:35 — la búsqueda SÍ sobreajustó a sus meses: control en mayo (`c222`)

Receta y top 5 de `c220`, una semilla, 1.000 rondas, entrenando marzo y validando mayo (fold A, mes que la
búsqueda no vio):

| config | AUC búsqueda (jun+jul) | AUC mayo | gan@8k mayo (M) |
| --- | ---: | ---: | ---: |
| **receta** | 0,92149 | **0,89501** | **270,6** |
| trial 16 | 0,92430 | 0,89238 | 262,9 |
| trial 14 | 0,92411 | 0,89476 | 266,2 |
| trial 13 | 0,92393 | 0,89437 | 261,8 |
| trial 19 | 0,92390 | 0,89297 | 264,0 |
| trial 18 | 0,92373 | 0,89330 | 264,0 |

**Cinco de cinco debajo de la receta en mayo**, en AUC y en ganancia. La mejora de 0,003 de AUC en junio y
julio no transfiere: la objeción de la usuaria (sobreajuste de la búsqueda a los meses de validación) era
correcta, y la del ruido de semilla no era el problema. Dos lecturas posibles, no excluyentes: (a) 20 trials
sobre dos meses fijos con error muestral ~0,004 eligen configuraciones que les calzan; (b) la dirección
encontrada (árboles de 170–210 hojas, hessiano 10× más bajo) necesita más filas de las que tiene el fold
A (un mes), así que mayo la castiga más que lo que la castigaría la entrega de cuatro meses. El público
con `c221` decide entre (a) y (b); mientras tanto **la receta sigue siendo la configuración vigente** y
una búsqueda futura tiene que validar en un mes fuera de la búsqueda, con varias semillas, o no hacerse.

### 8-oct 16:11 — el ensamble heterogéneo pierde; el ensamble de la receta es la mejor entrega (104,17)

Corte 14.000, misma partición:

| submit | qué es | público | vs referencia |
| --- | --- | ---: | ---: |
| `c201_lags_ens20_14000` | **un archivo**: ensamble por rank de las 20 semillas de la receta | **104,17** | +1,2 sobre los 20 archivos sueltos (103,00): Jensen, como se predijo |
| `c221_trials_14000` | 5 archivos, cada uno un trial de Optuna con 4 semillas ensambladas | 100,91 | **−3,3** |
| `c221_hetero_14000` | un archivo, 20 modelos heterogéneos (5 trials × 4 semillas) | 101,97 | **−2,2** |

**Tres instrumentos coinciden**: el control en mayo (`c222`), el trial medio en agosto y el ensamble
heterogéneo en agosto. La búsqueda de Optuna por AUC en junio+julio eligió configuraciones peores que la
receta para el mes real. Queda refutado B3 tal como se hizo; la heterogeneidad con miembros peores no
rescata nada. **La entrega vigente es el ensamble de 20 semillas de la receta sobre lags12, corte 14.000:
104,17**, el mejor público de la competencia hasta ahora.

Nota operativa: el bot ahora reporta cupo **x/17** (antes 13): quedan 7 submits hoy.

### 8-oct 16:12 — reparaciones + transiciones + recencia (`c231`): 102,58 a 14.000, −1,6 contra `c201`

Dentro del ruido (±3), con más dispersión entre semillas (sd 2,55 contra 1,3–2,2 de las demás). Las
reparaciones de Ramírez y las 40 features binarias no agregan sobre lags12; tampoco restan de forma
medible. **No se adopta** (criterio: ante empate, la entrega sin código nuevo). Quedan 6 submits hoy.

### 8-oct 16:15 — Zulip del día: no hay anuncio del cupo 17; sí hay una tabla Public vs Private

- El bot pasó de `x/13` a `x/17` entre el submit de las 16:11 y el siguiente. En los 48 mensajes de los
  streams archivados desde el 7-oct al mediodía **no hay ningún anuncio**; si lo hubo fue en un canal
  privado (`Cognitive Overloading`) o es un cambio silencioso del bot. Se toma el 17 como dato del bot.
- **[197324] Denicolay, `general > Private vs Public`, 8-oct 18:56**, sobre todos los submits del curso
  con 10 o más archivos (≈7.300):

  | público | private < 300 | private > 300 | % > 300 |
  | --- | ---: | ---: | ---: |
  | < 95 | 2.659 | 25 | 1% |
  | 95–100 | 1.706 | 115 | 6% |
  | **100–105** | **1.829** | **327** | **15%** |
  | > 105 | 349 | 257 | 42% |

  `mean(Public/Private) = 2,98` (o sea private ≈ 3 × público, el 75% contra el 25%). **El umbral
  que el profesor mira en private es 300 M**, que es la línea de muerte de ~100 público. Nuestro
  104,17 cae en la banda donde solo el 15% supera 300 en private: estar arriba de 105 en público
  casi triplica la chance (42%). El propio profesor atribuye el sesgo a que "los submits sucesivos
  van overfiteando el Public": la gente resube lo que anduvo bien en lo que ve.
- **Deadline**: Silvana Contreras escribe el jueves 8 "estamos a 3 días del cierre", que apunta al
  **domingo 11**, como el mensaje del profesor; el `help` del bot sigue diciendo `2026-10-11T03:01:01`.
  Sigue sin zanjarse; la entrega va adentro el sábado igual.

### 8-oct 16:50 — julio positive-unlabeled (`c240`): 103,21 a 14.000, −1,0 contra `c201`

Julio entra con sus 1.103 BAJA+1 como positivos (peso 0,25) y sin los 2.017 no-BAJA+1 del top 2.500 del
ranking de `c201` (los candidatos a BAJA+2 escondido): 816.397 filas, 9.170 positivos. La selección
solapa 96,2% con la de `c201`, así que el público solo podía ver ±2 M, y dio −1,0. **Neutro, no se
adopta.** Lectura: el mes extra a distancia 1 no mueve un modelo que ya está saturado de datos (la
grilla de meses de la semana pasada decía que 1 ≈ 2 ≈ 3 meses), y la poda no alcanza a limpiar lo que
`c168` ensuciaba. Queda cerrado A6; B10 (pseudo-etiquetado completo) no tiene sentido después de esto.

### 8-oct 17:05 — por qué el target BAJA+2 puro perdió (−5,3 público): diagnóstico en el fold B

Modelos cacheados con la receta sobre lags12, entrenados en marzo+abril, evaluados en junio (1.098 BAJA+2,
874 BAJA+1), 3 semillas cada uno:

| | AUC sobre BAJA+2 | AUC sobre BAJA+1 | gan@8k | gan@14k | captura@14k | BAJA+1 en el top 14k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| target BAJA+2 puro (4.067 pos.) | 0,8983 | 0,9206 | 338,8 | 394,9 | 64,6% | 647 |
| `pesos` 0,25 (8.067 pos.) | **0,9022** | **0,9272** | **363,0** | **407,0** | **65,6%** | 658 |

Tres hechos:
1. **El modelo entrenado con los BAJA+1 rankea MEJOR a los BAJA+2** (AUC 0,902 contra 0,898), aunque
   su target "no es el que paga". Los BAJA+1 son los mismos clientes un mes más tarde: su última foto
   es el ejemplo más claro de "se va", y sacarlos tira la mitad de la señal, no ruido.
2. **Excluirlos no evita mandarlos**: el modelo BAJA+2 puro pone 647 BAJA+1 en el top 14.000, el
   otro 658. La distinción "se va en uno o en dos meses" no se aprende (AUC 0,60 medido antes), así
   que el costo de los BAJA+1 en el corte es el mismo con los dos targets.
3. **La pérdida está en la cabeza del ranking**: −24 M a 8.000, −12 M a 14.000. Con la mitad de
   positivos y `min_sum_hessian_in_leaf` escalado por filas, las hojas ricas en positivos quedan
   más podadas justo donde el modelo necesita más confianza. El rank mediano de un BAJA+2 pasa de
   8.797 a 9.159.

Público −5,3 en el 25% ≈ −21 M al mes; el fold B dice −12 M. Mismo signo y orden. **Lo que sigue de
esto**: si más señal de BAJA+1 ayuda, `peso_baja1` 0,5 y 1,0 (`baja12`) sobre el ranking nuevo son
variantes con fundamento (en el base, 0,25 ganó a 0,5 y a 1,0 solo al corte 11.000 y con poca
evidencia). Van a la cola detrás de las 2.000 rondas.

### 8-oct 17:49 — 2.000 rondas (`c241`): 105,51 a 14.000, **+1,35** contra `c201`

Primera variante por encima de `c201` (104,16 con las mismas 5 semillas y partición). Dentro del ±3 de
ruido, pero con mecanismo: el `best_iter` de la búsqueda escalaba con los meses (1.000 con dos, tope
2.500 con tres) y la receta fija 1.000 para cuatro. Solapa 92,7% con `c201`. **Candidata, no adoptada**:
hace falta (a) 3.000 rondas para ver si sigue subiendo o ya baja, y (b) las 20 semillas con ensamble en
un archivo, comparable con los 104,17 de la entrega vigente. Las dos van a la cola de esta noche.
El submit fue el 13º del día y el bot lo aceptó con `13/17`; el 14º prueba el cupo nuevo.

### 8-oct 18:49 — pesos de BAJA+1: 0,5 → 102,15 (−2,0); 1,0 → 102,80 (−1,4). El 0,25 se queda

Las dos debajo de `c201` (104,16), dentro del ±3 pero las dos con el mismo signo. Con el −5,3 del target
BAJA+2 puro (peso 0), la curva pública en el peso es 0 → 98,8 · 0,25 → 104,2 · 0,5 → 102,1 · 1,0 → 102,8:
el 0,25 es el máximo y no es un artefacto del corte 11.000 como se temía. La hipótesis "más señal de
BAJA+1 mejora la cabeza" era correcta en el fold B para 0 contra 0,25 pero no extrapola: a peso pleno
el modelo aprende a rankear BAJA+1, que a 14.000 cuestan 27.500 cada uno y desplazan BAJA+2.
El bot aceptó los submits 14º y 15º: **el cupo de 17 es real**. Quedan 2 hoy.

### 8-oct 19:05 — ¿se puede distinguir BAJA+1 de BAJA+2 con FE? Medido: no alcanza, y no alcanzaría ni con señal perfecta de 3:1

Entre fugados, entrenando 03–05 y validando junio (1.972 fugados, 44% BAJA+1):

| features | AUC BAJA+1 vs BAJA+2 |
| --- | ---: |
| 150 crudas | 0,619 |
| 750 con lags | 0,543 (sobreajusta: 6.095 filas) |

Señales univariadas en junio (tasa en BAJA+1 / en BAJA+2): tarjeta en cierre 6,8% / 2,8% (×2,4); perdió el
descubierto este mes 4,2% / 1,5% (×2,7); perdió una familia de productos 13,5% / 6,6% (×2,0). Existen,
son el "proceso de cierre" de la última foto, pero cubren 7–14% de los BAJA+1 con ratio ≤ 2,7.

**La aritmética cierra la puerta**: bajar del top a un BAJA+1 ahorra 27.500; bajar por error a un
BAJA+2 cuesta 1.072.500, 39 veces más. Una regla con ratio 2,7:1 que saque del top 14.000 al 6,8% de los
~650 BAJA+1 (≈45) saca también al 2,8% de los ~720 BAJA+2 (≈20): +1,2 M − 21,5 M = **−20 M**. Para que
pague hace falta una señal con precisión > 39:1, y ninguna variable ni la tipología del video 1
(que describe patrones de fuga, no horizontes) está cerca. Cerrado: el BAJA+1 en el top es un costo
estructural, y la única palanca sobre él es el peso en el entrenamiento (0,25).

### 8-oct 19:10 — ensamble de los buenos (`c245`): 104,00 contra 104,17. Idéntico

Ensamble por rank de cinco ensambles de 5 semillas (receta 1.000, 2.000 rondas, lags sobre rangos,
dataset reparado, julio PU; 25 modelos). Solapa 97,8% con la entrega vigente; correlaciones de rango
entre miembros 0,95–0,999. **−0,2: no suma ni resta.** La diversidad que necesita un ensamble no está
en variantes del mismo dataset con la misma receta: todas ven lo mismo. El de 2.000 rondas es el menos
correlacionado (0,95–0,97) y por eso es el único que movió el público solo. Cerrado A4 en su versión
"promedio"; el stacking con meta-modelo no tiene con qué diversificar.
