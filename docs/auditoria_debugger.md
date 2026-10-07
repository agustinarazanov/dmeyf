# Auditoría del pipeline de la competencia — sesión `debugger`

2026-10-04, 23:25 ART. Fue de solo lectura: no se modificó código, base, parquets ni `experimentos/`.
No se entrenó ningún modelo ni se mandó nada al bot. Los scripts `c103`, `c106`, `c112` y
`c116`–`c120` no están en disco como `.py` (solo quedan sus carpetas de experimento), así que
no se pudieron auditar como código.

**CONFIRMADO** = lo verifiqué corriendo algo; el snippet va al lado. **SOSPECHADO** = surge de leer el código.

---

## 0. Operativo, urgente: el último submit es el final

**CONFIRMADO.** `zulip/respuestas/192400.json` dice textual:

> ℹ️ This is now your final submission -- your LAST pre-deadline submit always is, with no separate step needed.

El deadline es `2026-10-05T03:01:01` UTC (00:01 ART). Hoy el final es `c117_v2_ens20_9000`
(95,78). **Cualquier submit "para medir" que se mande antes del deadline pasa a ser el final.** La
columna `submit.es_final` del ledger no controla nada de esto: el ledger puede marcar como final un
submit que el bot ya no considera final. Si el final tiene que ser uno distinto del último, hay que
volver a mandar ese archivo **al final de todo**.

---

## (a) Colisiones de clave de caché

### a1. Hoy no hay colisiones en disco — CONFIRMADO

Leí la línea `feature_names=` de **todos** los `experimentos/*/modelo_*.txt` y la comparé con las
predictoras que salen hoy de cada parquet (`FUERA_DE_X` actual), agrupando por prefijo de etiqueta.
**Ningún prefijo mezcla dos sets de features**, y todos los modelos base/v101/v2 coinciden
exactamente con su parquet. Además, en todos los scripts la etiqueta de archivo lleva un prefijo
que ya codifica la config (`{nombre}_{mes}_{semilla}_{clave}`). Revisé uno por uno c100, c102, c104,
c107, c108, c109, c110, c113 y c119, y no encontré dos configs distintas que caigan en el mismo nombre de archivo.

(Script: `scratchpad/feats.py`. Recorre los modelos, parsea `feature_names` y compara contra
`pq.read_schema(...)` filtrado por `c.FUERA_DE_X`.)

### a2. `clave()` no ve el contenido del dataset ni la lista de columnas — CONFIRMADO que ya pasó

`competencia.py:135`. La firma usa el **nombre** del parquet (o un string libre) y nada más. Si el
parquet se regenera, o si cambia `FUERA_DE_X`, `entrenar_o_cargar` (`:150`) carga el modelo viejo
sin avisar.

- **Ya pasó**: los modelos `fe_*` de `c102_comparacion/` y de `c104_diagnostico_fe/` tienen
  **297 columnas**, y `competencia_01_fe.parquet` hoy da **223** predictoras (los `ferank` tienen
  195). O sea, **el "FE pierde 46,9 M" de c102/c104 se midió sobre otro parquet que ya no existe.**
  Si hoy se vuelve a correr c102 o c104, LightGBM tira error por la cantidad de features (falla
  ruidoso, no silencioso). Pero el número reportado no es reproducible con el FE actual.
- El caso silencioso es este: si cambia el **contenido** de una columna sin cambiar cuántas son,
  por ejemplo al regenerar v101/v2 con otra definición de feature, se carga el modelo viejo y se
  reporta como nuevo. `Booster.predict` no valida nombres por defecto.
- **Arreglo sugerido**: meter en la firma `sorted(X.columns)` y el `mtime`/hash del parquet.
  Más barato: guardar los `feature_names` y validar después de cargar.

### a3. Los argumentos raros de `dataset` no colisionan, pero tienen un riesgo latente — SOSPECHADO

`c109:50` pasa `[len(X), semilla_sub]` en el lugar de `meses`. Hoy no colisiona porque m03, m04 y
m04rec tienen distinto `len(X)` y la etiqueta lleva el prefijo. Pero si cambia la semilla del
recorte, `semilla_sub` cambia y está en la clave, así que eso está bien. Lo que **no** entra en la
clave es la semilla del `rng` del recorte: es `c.SEMILLAS[1]`, que es la misma que `semilla_sub`, así
que hoy coincide. Es frágil.

---

## (b) Leakage

### b1. `c105_busqueda.py`: los gemelos BAJA+2 / BAJA+1 cruzan el split — CONFIRMADO (alta gravedad para c105/c106)

`c105:35,63`. `StratifiedShuffleSplit` reparte filas de **202103 y 202104 juntas**. Los 960 BAJA+2
de 202103 son exactamente los mismos 960 clientes que figuran como BAJA+1 en 202104:

```python
# devuelve (960, 960, 161881)
with a as (select numero_de_cliente id, clase_ternaria c from read_parquet('data/competencia_01.parquet') where foto_mes=202103),
     b as (select numero_de_cliente id, clase_ternaria c from read_parquet('data/competencia_01.parquet') where foto_mes=202104)
select count(*) filter (where a.c='BAJA+2'), count(*) filter (where a.c='BAJA+2' and b.c='BAJA+1'), count(*) filter (where b.c is not null)
from a left join b using(id)
```

Con 70/30, ~70% de los BAJA+2 de 202103 que caen en test tienen a su gemelo de 202104 en train.
Las features son casi iguales y, con `esquema='pesos'`, el gemelo lleva **etiqueta positiva**. El
objetivo premia sistemáticamente fusionar las clases con peso alto. Eso explica que el ganador de
c106 tenga `peso=0,91`, `ganancia_cv=1.008 M` contra `oot_media=318 M`, y la frase de c107 "ese
objetivo premia fusionar las clases del todo": **no es un rasgo del objetivo, es leakage**. Hay
además leakage de cliente en general: 161.881 clientes están en los dos meses. A eso se suma el
sesgo optimista de hacer early stopping sobre el mismo fold que se reporta (`:74-78`), y el corte
`argmax` post-hoc.
**Impacto**: invalida la comparación Optuna vs. base de c105/c106. No arruina ningún submit, porque
el final no usa esos params. **Arreglo**: `GroupKFold`/`StratifiedGroupKFold` por
`numero_de_cliente`, o validar out-of-time.

### b2. Ventanas SQL: no hay lookahead — CONFIRMADO por lectura

`fe.clausula_ventana()` (`fe_panel.py:216`) es `partition by numero_de_cliente order by foto_mes`
sin marco, o sea el default `RANGE UNBOUNDED PRECEDING .. CURRENT ROW`. c115 usa `v2` (2
preceding .. 1 preceding) y `v3` (2 preceding .. current row). En c101/c111/c115 no hay `following`
ni `lead()`, y todas las ventanas tienen `order by`. El único `lead` está en `_SQL_BASE`, que es la
construcción del target, correcto. `percent_rank` está particionado por `foto_mes`: es corte
transversal del mismo mes, sin futuro. `FUERA_DE_X` cubre `COLUMNAS_DE_BAJA`, `numero_de_cliente`,
`foto_mes`, `clase_ternaria`, `clase_binaria`, `peso` y las dos `F*iniciomora`, y los modelos en disco
confirman que ninguna entró (a1).

### b3. c115: las ventanas "fijas" siguen sin ser iguales en todos los meses — CONFIRMADO

El docstring de c115 dice que ventana fija = feature estable. Medido sobre `competencia_01_v2.parquet`:

| foto_mes | `w_ctrx_cae_a_cero` | `w_rojo_3m>=2` | `w_rojo_3m>=3` | `w_canales_cero_3m>=3` |
|---|---|---|---|---|
| 202103 | **1,06%** | **0** | **0** | **0** |
| 202104 | 0,17% | 14,4% | **0** | **0** |
| 202105 | 0,18% | 19,3% | 11,6% | 2,1% |
| 202108 | 0,14% | 17,5% | 9,8% | 2,0% |

- `ROWS 2 PRECEDING` sobre 202103/202104 tiene 1 y 2 filas: los umbrales de 3 meses son
  **imposibles en los dos meses de train de los folds A/B**. Es el mismo defecto de antes, ahora en dos meses en vez de en todos.
- `w_ctrx_cae_a_cero` (`c115:56`) hace `coalesce(lag(...),1) > 0`. En el primer mes de cada
  cliente (todo 202103, y cada alta), **cualquier ctrx=0 cuenta como "cayó a cero"**: ×6 en
  202103. Lo mismo pasa con `w_tc_cierre_nuevo` (`coalesce(lag,0)=0`).
- Pasa lo mismo con un cliente que tiene un hueco de meses o reingresa, porque el id se reutiliza:
  `ROWS` cuenta filas, no meses.

**Impacto**: el modelo de c116/c117 (el final actual) aprende en los folds una relación que en
202108 se ve distinta. No es leakage, es drift construido. Puede explicar parte de lo que hoy se
atribuye a las features.

### b4. `percent_rank` manda los NULL arriba de todo — CONFIRMADO

```python
duckdb.sql("select x, percent_rank() over (order by x) from (values (1),(2),(null),(3)) t(x)")
# -> NULL recibe 1.0
```

- `c115:69` `w_fee_por_trx_rk`: `ratio_seguro(..., ctrx_quarter)` es NULL cuando ctrx=0, y esos
  clientes quedan en rank 1,0. Medido: **~10% del bucket `>=0,9` son clientes con ctrx=0**, el grupo
  de mayor riesgo. **El lift que imprime c115 para `w_fee_por_trx_rk >= 0.9` está inflado por
  inactividad**: no es una señal de "costo relativo".
- `c101`/`fe.sql_rank` sobre todos los `m*`: en `Master_*` (~59% NULL) "no tiene tarjeta" queda
  con el rank del monto máximo. Para un árbol es separable (1,0 exacto), así que el daño es bajo.
  Para cualquier lectura humana del rank es engañoso.
- **Arreglo**: `order by x nulls first`, o `case when x is null then null else percent_rank() ... end`.

### b5. `sql_slope` sigue usando `cliente_antiguedad` como eje — SOSPECHADO (baja)

`fe_panel.py:203`. El CLAUDE.md dice que eso se corrigió en z402 porque se resetea en un
reingreso, pero el default del helper sigue siendo ese, y `c101` lo usa. Afecta solo al parquet FE.

---

## (c) Métrica

Revisado, **sin bugs**:
- `ganancia(…, envios)` usa `[envios-1]` y es consistente con `curva_de_corte` (`acum[cortes-1]`),
  `regret` (`curva[k-1]`), c100/c102/c107 (`[corte-1]`), c104 (`[10_999]`) y c108–c119 (`[CORTE-1]`).
- `top_k` devuelve exactamente `k` si `len(ids) >= k`, con el mismo orden de `argsort` que
  `ganancia_acumulada`.
- Empates: `argsort` (quicksort) no es estable, pero **medí empates en el valor de corte en 202108:
  1 fila**, tanto en la semilla 261431 como en el ensamble, para k=9.000 y k=11.000. Es irrelevante.
- `regret` normaliza por `curva.max()` sobre toda la curva, no solo sobre [8k, 15k]. Es correcto
  para regret.

Observaciones menores:
- `c113` **v4_reciente: la lectura local no mide la variante** — CONFIRMADO. `c113:67` hace
  `[m for m in [202105,202106] if m in meses_f] or meses_f`, que da `[]`, y el `or` lo reemplaza por
  los meses del fold (202103 / 202103+202104). `resumen.json` lo muestra: v1 y v4 dan
  **250,80 idénticos** en 202105, y v2 y v3 **246,18 idénticos**, porque con un solo mes
  `lambda**0 = 1`. **La columna `local_*` de v4_reciente en el ledger (`ganancia_val_media`
  325,27) es la de un modelo base pesos25 sin recencia.** Cambia un número ya reportado.
- `c104` mide a 11.000 fijo y `c102` en el corte minimax de cada config. Las dos cosas están
  declaradas, pero no se pueden comparar directamente.

## (d) Target y pesos

- `preparar(target='pesos')`: BAJA+2 → 1, BAJA+1 → `peso_baja1`, CONTINUA → 1. Bien.
- Recencia × clase (c110, c113, c119): `sub = data[data.MES.isin(meses)]` filtra con la misma
  máscara que `preparar`, así que **el orden es el mismo y el producto queda alineado**. Bien.
- La ganancia se mide siempre contra BAJA+2 en c100/c102/c104/c107/c108/c109/c110/c113 (`es_baja2`
  sale de `CLASE == "BAJA+2"` del mes de validación). En c119 se mide contra BAJA+1 a propósito y
  está declarado.
- `c105` `feval`: `verdad = y_real[te]` y `dte = X.iloc[te]` salen del mismo índice, así que
  **mide contra BAJA+2 y está bien alineado**. El problema de c105 es b1, no el feval.
- SOSPECHADO: con `lambda` chico (c110, 0,05) el peso total del mes viejo cae a ~0. `min_data_in_leaf`
  cuenta filas pero `min_sum_hessian_in_leaf` usa pesos, así que lambda cambia también la
  regularización efectiva. Es un confusor, no un bug.

## (e) Escritura de envíos

`competencia.py:159`:
- **El guard de `validos` es tautológico en todos los scripts**: `validos = set(ids)` y `top_k`
  elige de esos mismos `ids`. Nunca puede fallar. (Es la versión general del bug #1 y
  `verificar_seleccion` lo compensa. Pero **ningún script del repo llama a `verificar_seleccion`**:
  `grep` no la encuentra fuera de `competencia.py`. Solo la usa el código inline que no está en
  disco.)
- `astype("int64")` trunca floats con decimales y convierte NaN en `-9223372036854775808` sin
  error, porque el chequeo `"." in texto` mira el archivo ya casteado. Hoy no pasa (los ids salen
  de la columna id), pero si se pasaran scores por error, el CSV saldría "válido". Sugerido:
  `assert np.all(np.asarray(ids) == np.round(ids)) and not np.isnan(...).any()` antes del cast, y
  `ids > 0`.
- Duplicados y relectura: bien. La relectura compara sorted contra sorted, y está bien porque el
  orden del CSV no importa.

**Verificación cruzada de todos los CSV en disco** (CONFIRMADO): busqué para cada
`experimentos/*/envios/*.csv` la columna de scores (c107 / c117 / c120) cuyo top-k más se le parece.
Todos coinciden 91–100% con su modelo, o con uno hermano en el caso de las variantes de c113, que
no tienen parquet de scores. **La única excepción es `c118_v101_9000.csv` con 5,7%, el bug #1 ya
conocido.** `c120_mezcla` está alineado: el orden de filas de c117 coincide 0% con el de c107,
pero c120 hace merge por id (spearman por id = 1,0 contra cada fuente) y su CSV es exactamente el
top 9.000 de `mezcla`.

## (f) Ledger (`registro.py`)

- **`alta_resultado` duplica** — SOSPECHADO, por lectura. `resultado` no tiene `unique(submit_id)`
  y el insert no tiene `on conflict`. Si se re-parsea una respuesta (por ejemplo `ultimas`, que vuelve
  a guardar las últimas 5), se crea otra fila. Y `v_submits` hace `left join archivo` ×
  `left join resultado`: con dos resultados iguales, `count(a.id)` **se duplica** (`n_archivos` 2×),
  y con resultados distintos el submit aparece **dos veces** en `listar()`. Arreglo:
  `unique(submit_id)` + `on conflict do update`, o agregar `resultado` en una subconsulta.
- `alta_experimento` hace `on conflict (nombre) do update`: **volver a correr un script con otra
  config pisa la fila del experimento al que apuntan submits viejos**. Los nombres de c102
  (`base_baja2`, `fe_pesos25`…) no llevan prefijo de script y quedan en el mismo espacio de
  nombres que cualquier script futuro. Los campos que no se pasan quedan con el valor viejo, por
  ejemplo un `ganancia_val_sd` de una corrida anterior.
- `marcar_enviado` / `marcar_estado` no chequean `rowcount`: un nombre mal escrito no hace nada y no avisa.
- `c107:128` y `c113:118` hacen `update experimento set justificacion_corte=...`, pero **esa
  columna no está en `ESQUEMA`**. SOSPECHADO: se agregó a mano con `ALTER`. Sobre una base nueva,
  `crear_esquema` + c107 falla.
- `purgar(borrar_modelos=True)` usa `glob("modelo_s*.txt")`, pero los modelos se llaman
  `modelo_fin_*`, `modelo_val*`, etc., así que casi no borra nada. En cambio, en c108 sí matchearía
  `modelo_solo2021*`. Además `experimento.ruta` es la carpeta **compartida** por varios
  experimentos (los 4 de c113, los 8 de c102): descartar uno borraría modelos de los otros.
- `es_final` no tiene relación con lo que el bot considera final (ver §0).

## Zulip (`zulip/competencia.py`)

- Re-parseé las 27 respuestas guardadas con los regex actuales y la media sale bien en todas, incluida `-49.8575`.
- `PATRON_RECHAZO` incluye `error`, que matchearía cualquier mensaje con esa palabra (por ejemplo
  "std error"). Hoy no pasa. Baja.
- `validar_envio` detecta duplicados sobre la línea cruda: `" 123"` y `"123"` no se ven como
  duplicados. Con `np.savetxt` no ocurre. Baja.
- `escuchar(..., desde_id=message_id)` filtra por `id > message_id` del comando. Si hay dos
  submits seguidos y el bot contesta el primero tarde, la respuesta del primero se puede atribuir al
  segundo. No hay correlación por nombre del submit. SOSPECHADO, media: la respuesta no trae el
  nombre, pero sí `Submission ID`, y el comando `list` da el mapeo nombre → ID para verificar.

---

## Resumen por gravedad

| # | Qué | Estado | ¿Cambia algo reportado / submit? |
|---|---|---|---|
| 0 | El último submit antes de 03:01 UTC es el final | CONFIRMADO | **Sí: cualquier submit de medición reemplaza al final** |
| b1 | c105: leakage por gemelos BAJA+2/BAJA+1 en el split aleatorio | CONFIRMADO | Invalida Optuna vs. base (c105/c106) |
| a2 | clave() no ve el contenido ni las columnas; FE de c102/c104 medido con un parquet de 297 cols que ya no existe | CONFIRMADO | El "FE pierde 46,9 M" no es reproducible |
| c | c113 v4_reciente local = base pesos25 (250,80 == v1) | CONFIRMADO | `ganancia_val_media` de c113_v4_reciente es incorrecta |
| b3 | c115: ventanas de 3 meses imposibles en 202103/04; cae_a_cero ×6 en 202103 | CONFIRMADO | Afecta al modelo final actual (v2) |
| b4 | percent_rank: NULL = 1,0; el lift de fee_por_trx_rk está inflado por ctrx=0 | CONFIRMADO | Lectura de c115 |
| f | alta_resultado duplica; la vista multiplica n_archivos | SOSPECHADO | Ledger |
| e | `validos` es tautológico; verificar_seleccion no se llama en ningún script | CONFIRMADO | Fue el hueco del bug #1 |
| a1 | Colisiones de caché en disco | **no hay** | — |
