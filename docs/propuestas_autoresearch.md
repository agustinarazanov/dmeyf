# Propuestas de features — minado del diccionario (autoresearch)

Fecha: 2026-10-04. Fuente: `data/competencia_01.parquet` + `diccionario de datos.csv` (leído entero).
Sin modelos ni ganancia: sólo lift univariado sobre los meses etiquetados.

## Cómo se midió

- **Tasa base** = BAJA+2 / filas del mes. 202103–202106: 0,62% en promedio (960 / 1.139 / 870 / 1.098).
- **Lift** = tasa dentro del grupo / tasa base, pooled sobre los meses medidos.
- **Lift por mes**: el mismo cociente mes a mes (estabilidad).
- **Lift con ctrx bajo / alto**: lift *dentro del estrato* `ctrx_quarter < 28` (decil 1) y `>= 28`. Es la columna
  que dice si la feature aporta algo **además** de la inactividad, que es la señal univariada dominante
  (decil 1 de `ctrx_quarter`: ×5,8). Una feature con lift global alto pero ×1,0 dentro del estrato bajo
  es casi un sinónimo de "no transacciona".
- **Cob. 08** = % de filas de 202108 que caen en el grupo. Si difiere mucho de la cobertura en meses
  etiquetados, la feature no se comporta igual en el mes a predecir.
- Features de **historia** (las que miran `t-1`, `t-2`) se miden sólo en 202105–202106 y sólo sobre
  clientes con los 3 meses (`n3 = 3`), para que la ventana sea completa y comparable con 202108.
  Ver el problema 1 de calidad: con ventanas "desde el inicio del panel" la cobertura en 202108 se
  triplica sola.
- Todas las ventanas son `rows between k preceding and current row` sobre `partition by numero_de_cliente
  order by foto_mes`. Nada mira hacia adelante; todo se puede calcular en 202108.

Atajos usados en las expresiones (todas null-safe con `coalesce(...,0)`):

```sql
-- actividad del MES en todos los canales (ctrx_quarter es una ventana de 90 días; esto no)
canales_trx_mes = coalesce(chomebanking_transacciones,0) + coalesce(cmobile_app_trx,0)
                + coalesce(ccallcenter_transacciones,0) + coalesce(ccajas_transacciones,0)
                + coalesce(catm_trx,0) + coalesce(catm_trx_other,0)
                + coalesce(ctarjeta_debito_transacciones,0)
-- las dos tarjetas
tc_status_max   = greatest(coalesce(Visa_status,0), coalesce(Master_status,0))
ctarjetas       = coalesce(ctarjeta_visa,0) + coalesce(ctarjeta_master,0)
-- primacía
payroll_trx     = coalesce(cpayroll_trx,0) + coalesce(cpayroll2_trx,0)
debaut_total    = coalesce(ccuenta_debitos_automaticos,0) + coalesce(ctarjeta_visa_debitos_automaticos,0)
                + coalesce(ctarjeta_master_debitos_automaticos,0)
pagos_serv      = coalesce(cpagodeservicios,0) + coalesce(cpagomiscuentas,0)
-- patrimonio que el cliente tiene en el banco
activos_cliente = coalesce(mcaja_ahorro,0)+coalesce(mcaja_ahorro_adicional,0)+coalesce(mcaja_ahorro_dolares,0)
                + coalesce(mcuenta_corriente,0)+coalesce(mcuenta_corriente_adicional,0)
                + coalesce(mplazo_fijo_pesos,0)+coalesce(mplazo_fijo_dolares,0)
                + coalesce(minversion1_pesos,0)+coalesce(minversion1_dolares,0)+coalesce(minversion2,0)
-- ventanas
W  = (partition by numero_de_cliente order by foto_mes)
W2 = (W rows between 2 preceding and 1 preceding)       -- los dos meses anteriores
W3 = (W rows between 2 preceding and current row)       -- últimos tres meses incluido el actual
```

---

## Tabla principal — ordenada por lift

| # | Feature | Expresión SQL | Mecanismo | Filas (cob.) | Tasa | Lift | Lift por mes | ctrx bajo / alto | Cob. 08 |
|---|---|---|---|---:|---:|---:|---|---|---:|
| 1 | `ctrx_cae_a_cero` | `ctrx_quarter = 0 and lag(ctrx_quarter) over W > 0` | el último movimiento voluntario acaba de salir de la ventana de 90 días: la cuenta quedó muda | 841 (0,13%) | 8,92% | **14,3** | 16,2 / 11,0 / 14,7 | ×2,5 / — | 0,14% |
| 2 | `aq1_ctrx0` | `active_quarter = 1 and ctrx_quarter = 0` | incoherencia entre los dos contadores de actividad (el diccionario avisa que no miden lo mismo): figura activo sin un solo movimiento en cuentas | 2.302 (0,23% del panel) | 8,28% | **≈13** | — (chico, no partido por mes) | — | — |
| 3 | `rojo_sin_acuerdo` | `mcuentas_saldo < 0 and coalesce(cdescubierto_preacordado,0) = 0` | debe plata sin línea pactada: el banco le cobra caro y el cliente tiene motivo para irse (o el banco para echarlo) | 4.851 (0,74%) | 6,91% | **11,1** | 11,5 / 8,8 / 8,7 / 15,2 | ×2,6 / **×7,4** | 1,56% ⚠ |
| 4 | `tc_cierre_nuevo` | `tc_status_max >= 6 and coalesce(lag(tc_status_max) over W,0) = 0` | una tarjeta entra en proceso de cierre este mes: está desarmando la relación producto por producto | 2.802 (0,43%) | 6,35% | **10,2** | 10,1 / 13,5 / 8,7 / 7,2 | ×3,3 / **×8,2** | 0,51% |
| 5 | `perdio_tarjeta` | `ctarjetas < lag(ctarjetas) over W` | mismo mecanismo visto desde la cantidad de cuentas de tarjeta | 2.079 (0,32%) | 6,35% | 10,2 | 13,1 / 8,2 / 7,1 | ×3,5 / ×8,0 | 0,58% |
| 6 | `active_quarter_0` | `active_quarter = 0` | ningún movimiento voluntario en el trimestre; más estricto que `ctrx_quarter` porque excluye comisiones y cuotas | 8.566 (1,31%) | 6,65% | **10,7** | 10,9 / 13,0 / 9,4 / 9,1 | ×1,9 / — | 1,34% |
| 7 | `canales_cero_3de3` | `sum((canales_trx_mes = 0)::int) over W3 = 3` | tres meses sin usar **ningún** canal (HB, app, teléfono, caja, cajero propio o ajeno, débito): el banco ya no es su banco | 6.773 (2,10%)ʰ | 5,05% | **8,3** | 7,0 / 9,4 | ×1,5 / **×7,0** | 1,98% |
| 8 | `canales_cero_2de3` | `sum((canales_trx_mes = 0)::int) over W3 >= 2` | ídem, más cobertura | 10.914 (3,39%)ʰ | 4,65% | 7,7 | 6,3 / 8,7 | ×1,5 / ×5,1 | 3,21% |
| 9 | `paga_sin_usar_3de3` | `sum((coalesce(mcomisiones_mantenimiento,0) > 0 and ctrx_quarter < 28)::int) over W3 = 3` | le cobran mantenimiento del paquete tres meses seguidos y casi no lo usa: la ecuación costo/valor le dice que se vaya | 11.865 (3,68%)ʰ | 4,37% | **7,2** | 6,8 / 7,5 | ×1,2 / — | 3,05% |
| 10 | `ctrx_cae_fuerte_3m` | `ctrx_quarter < 0.6 * max(ctrx_quarter) over W2` | perdió más del 40% de su actividad contra su propio máximo reciente: desenganche relativo, no absoluto | 8.806 (2,73%)ʰ | 4,21% | **6,9** | 6,6 / 7,2 | ×1,5 / **×8,2** | 2,09% |
| 11 | `sin_tarjeta_credito` | `ctarjetas = 0` | un premium sin tarjeta no usa la mitad de lo que paga | 28.469 (4,35%) | 3,54% | 5,7 | 5,1 / 6,5 / 4,3 / 6,5 | ×1,9 / ×3,8 | 4,65% |
| 12 | `sin_primacia_ni_pagos` | `payroll_trx = 0 and debaut_total = 0 and pagos_serv = 0` | nada lo ata: ni sueldo, ni débitos, ni pagos de servicios; irse no le cuesta ningún trámite | 57.594 (8,81%) | 3,45% | 5,5 | 5,2 / 6,0 / 5,2 / 5,7 | ×1,3 / ×4,3 | 8,06% |
| 13 | `caja_vacia` | `percent_rank() over (partition by foto_mes order by mcaja_ahorro) < 0.10` | caja de ahorro del paquete vaciada (rankeado por mes para neutralizar inflación) | 65.409 (10%) | 3,23% | 5,2 | 4,9 / 5,3 / 4,9 / 5,5 | ×1,6 / ×3,7 | 10% |
| 14 | `digital_cero` | `coalesce(chomebanking_transacciones,0) + coalesce(cmobile_app_trx,0) = 0` | ni home banking ni app en el mes: no hay relación cotidiana | 18.845 (5,85%)ʰ | 3,12% | 5,1 | 4,6 / 5,6 | ×1,3 / ×2,4 | 5,56% |
| 15 | `fee_por_trx_top10` | `coalesce(mcomisiones_mantenimiento,0) > 0 and percent_rank() over (partition by foto_mes order by mcomisiones_mantenimiento / nullif(ctrx_quarter,0)) >= 0.90` | precio por uso: cuánto le cobra el banco por cada movimiento. Rankeado por mes porque el monto de comisión cambia de precio | 30.890 (9,59%)ʰ | 3,03% | 5,0 | 4,9 / 5,1 | ×1,3 / **×4,2** | 9,40% |
| 16 | `pasivos_margen_cero` | `coalesce(mpasivos_margen,0) <= 0` | el banco no gana nada con su dinero = no deja plata en el banco | 27.837 (4,26%) | 2,36% | 3,8 | 4,1 / 3,4 / 3,8 / 4,0 | **×1,8** / ×1,1 | 3,61% |
| 17 | `rojo_3de3` | `sum((mcuentas_saldo < 0)::int) over W3 = 3` | rojo persistente, no un descuido de un mes | 36.060 (11,2%)ʰ | 2,07% | 3,4 | 3,1 / 3,7 | ×1,8 / ×1,6 | 9,92% |
| 18 | `activos_bajos` | `percent_rank() over (partition by foto_mes order by activos_cliente) < 0.20` | patrimonio total en el banco en el quintil inferior | 130.814 (20%) | 2,04% | 3,3 | 3,3 / 3,3 / 3,2 / 3,3 | ×1,5 / ×2,5 | 20% |
| 19 | `plastico_vence_60d` | `least(coalesce(-Visa_Fvencimiento,99999), coalesce(-Master_Fvencimiento,99999)) < 60` | el plástico vence en menos de dos meses: momento natural de decidir no renovar | 1.037 (0,16%) | 1,74% | 2,8 | 2,1 / 3,3 / 2,8 / 2,8 | ×2,1 / ×2,0 | 0,18% |
| 20 | `mora_vieja` | `(Visa_delinquency = 1 and coalesce(Visa_Finiciomora,1) > 0) or (Master_delinquency = 1 and coalesce(Master_Finiciomora,1) > 0)` | mora real, sin los ceros del ciclo de Visa (ver problema 2). Reemplaza a `*_delinquency` crudo | 7.063 (1,08%) | 1,63% | 2,6 | 2,0 / 2,5 / 2,5 / 3,8 | ×1,4 / ×2,2 | 0,75% |
| 21 | `debaut_perdido_3m` | `debaut_total = 0 and max(debaut_total) over W2 > 0` | dio de baja los débitos automáticos: es el paso previo de quien muda sus pagos a otro banco | 7.061 (2,19%)ʰ | 1,46% | 2,4 | 2,4 / 2,3 | ×1,3 / ×2,4 | 1,82% |
| 22 | `payroll_perdido_3m` | `payroll_trx = 0 and max(payroll_trx) over W2 > 0` | dejó de cobrar el sueldo acá (portabilidad o cambio de empleo) | 6.384 (1,98%)ʰ | 1,35% | 2,2 | 2,0 / 2,3 | ×0,7 / **×4,0** | 2,02% |
| 23 | `prestamo_ultimas_cuotas` | `cprestamos_personales > 0 and mprestamos_personales < 0.68 * lag(mprestamos_personales) over W` | la deuda cae de golpe = precancelación o últimas cuotas: se desata el ancla del préstamo | 6.724 (1,03%) | 1,22% | 2,0 | 1,7 / 2,2 / 2,0 | **×4,5** / ×1,5 | 1,54% |
| 24 | `prestamo_cancelado` | `lag(cprestamos_personales) over W > 0 and coalesce(cprestamos_personales,0) = 0` | ídem, el préstamo terminó | 3.906 (0,60%) | 1,20% | 1,9 | 1,7 / 2,2 / 1,8 | ×2,4 / ×2,5 | 1,20% |
| 25 | `bajo_productos_3m` | `cproductos < max(cproductos) over W2` | cerró alguna familia de productos en los últimos dos meses | 22.619 (7,0%)ʰ | 1,18% | 1,9 | 2,0 / 1,9 | ×1,9 / ×2,0 | 7,75% |

ʰ = feature de historia, medida en 202105–202106 sobre clientes con 3 meses de panel.

**Variables continuas que conviene dar al modelo como tales** (deciles por mes; lift del decil extremo):

| Variable | Decil de mayor riesgo | Lift | Rango por mes | Nota |
|---|---|---:|---|---|
| `ctrx_quarter` | d1 (0–28) | 5,85 | 5,30–6,12 | monotónica, la base de todo |
| `percent_rank(mcaja_ahorro)` por mes | d1 | 5,19 | 4,91–5,46 | |
| `canales_trx_mes` | d1 (0–4) | 4,72 | 3,92–5,11 | |
| `mcomisiones_mantenimiento / ctrx_quarter` | d10 (> ~39) | 4,20 | **4,17–4,25** | la más estable de todo el estudio |
| `percent_rank(activos_cliente)` | d1–d2 | 3,48 / 3,08 | | caída abrupta en d3 (0,92) |
| `ctrx_quarter / max(ctrx_quarter) over W` | d1 (< 0,86) | 3,05 | 2,64–3,53 | |
| `(Visa+Master saldo) / (Visa+Master límite)` | d1 (≤ 0) | 2,90 | 2,58–3,18 | tarjeta sin saldo = no la usa |
| `(Visa+Master consumo) / límite` | d1–d2 | 2,15 / 2,02 | | |
| `mprestamos_personales / lag(...)` | d1 (< 0,68) | 1,91 | 1,68–2,19 | ver #23 |
| `cliente_antiguedad` | d1 (< 30 meses) | 1,52 | 1,38–1,70 | monotónica decreciente |
| `cliente_edad` | d10 (≥ 65) | 1,34 | 1,19–1,49 | **en U**: también sube en < 35 |
| `percent_rank(mrentabilidad_annual)` | d3–d4 | 1,47 / 1,64 | | **no monotónica**: las dos puntas protegen (0,55 / 0,52). Un árbol la corta bien; una regla lineal no |

### Hipótesis que NO funcionaron (también es información)

| Feature | Expresión | Lift | Qué dice |
|---|---|---:|---|
| `cuenta_puente` | `payroll > 0 and mtransferencias_emitidas >= 0.9 * payroll` | **0,29** | esperaba "cobra acá y se lo lleva a otro banco". Es al revés: el que mueve su sueldo es un cliente vivo |
| `forex_venta` | `cforex_sell > 0` | 0,14 | vender dólares no es liquidar la relación; es uso intensivo |
| `plazo_fijo_no_renovado` | `lag(cplazo_fijo) > 0 and cplazo_fijo = 0` | 0,78 | no renovar un plazo fijo no anticipa la baja |
| `rent_neg` | `mrentabilidad < 0` | 0,56 | el banco perdiendo plata con el cliente no lo hace irse |
| `cheque_rechazado` | `ccheques_*_rechazados > 0` | 0,71 | |
| `tcuentas <= 1` | | 0,99 | inútil: vale 1 en el 99,5% de las filas (ver problema 6) |
| `mobile_flag` (`tmobile_app = 1`) | | 0,81 | inútil: ver problema 4 |
| `tc_consumo_cayo_3m` | consumo < 20% del máximo de 2 meses | 1,36 | la tarjeta se apaga **después** de la cuenta, no antes |
| `dolares_retirados` | | 1,31 | |

---

## Problemas de calidad de datos

### 1. Ventanas "desde el inicio del panel" cambian de distribución en 202108 — **trampa de diseño**

Cualquier `max/sum ... over (rows between unbounded preceding ...)` acumula más historia cuanto más
tarde es el mes. Medido:

| Feature (ventana abierta) | Cob. 202103–06 | Cob. 202108 |
|---|---:|---:|
| `meses_en_rojo >= 3` | 6,6% | **19,8%** |
| activos < 20% del máximo histórico | 7,4% | **16,1%** |
| `cproductos < max histórico` | 5,1% | **14,2%** |
| consumo tarjeta < 20% del máximo | 5,9% | **15,3%** |
| `ctrx_quarter < 0,6 × máximo` | 2,0% | **5,6%** |

El modelo aprende el umbral en meses con poca historia y lo aplica en un mes con mucha: en 202108
marca el triple. **Usar siempre ventana fija** (`rows between 2 preceding and current row`) y entrenar
sólo con filas que la tengan completa. Con ventana fija las coberturas cierran (tabla principal, Cob. 08).

### 2. `Visa_delinquency` está contaminada en 202105 y 202108, no sólo `Visa_Finiciomora`

Los 3.222 (202105) y 3.249 (202108) ceros de `Visa_Finiciomora` son **exactamente** las filas con
`Visa_delinquency = 1` y `Finiciomora = 0`. La tasa de mora de Visa salta en esos dos meses:

| | 202103 | 202104 | **202105** | 202106 | 202107 | **202108** |
|---|---:|---:|---:|---:|---:|---:|
| `avg(Visa_delinquency)` | 0,95% | 0,92% | **2,79%** | 0,59% | 0,62% | **2,66%** |
| `avg(Master_delinquency)` | 0,77% | 0,66% | 0,56% | 0,44% | 0,46% | 0,38% |
| filas con `Visa_mpagado > 0` | 13.812 | 13.425 | **41.622** | 15.925 | 15.084 | **42.863** |

Y la mora "fresca" no predice: en 202105, las 3.222 con `Finiciomora = 0` tienen tasa BAJA+2 de
**0,81% (×1,5)** contra **1,60% (×3,0)** de la mora vieja del mismo mes. Si se reemplaza
`F*iniciomora` por `*_delinquency` crudo, **el modelo entrenado en 03/04/06 ve en 202108 el triple de
morosos de Visa, con la mitad de riesgo cada uno**. Usar `mora_vieja` (#20).

Interpretación probable: no es inyección sino **calendario del ciclo de la tarjeta**. La mediana de
`*_fultimo_cierre` por mes es 1 / 2 / **5** / 0 / 3 / **6**: en 202105 y 202108 el vencimiento cae antes
de la foto, así que aparecen los pagos del ciclo (`Visa_mpagado` ×3) y los que todavía no pagaron
figuran morosos desde el día 0. Sea cual sea la causa, **202108 se parece a 202105 en todo el bloque
de pagos de Visa y no a 202106**: si se valida en un solo mes, 202105 es el más parecido al de la competencia
en este bloque.

### 3. `tcallcenter` NO es adhesión: es exactamente `ccallcenter_transacciones > 0`

El diccionario dice "{0,1} indica si la persona está adherida al canal de banca telefónica". En los
datos `tcallcenter = 1 ⇔ ccallcenter_transacciones > 0` en las 983.061 filas, sin una sola excepción
(87.830 / 895.231). Es un flag de "transaccionó por teléfono este mes", 100% redundante con el contador.

### 4. `tmobile_app` no significa "se instaló la app"

714.457 filas tienen `tmobile_app = 0` **y** `cmobile_app_trx > 0`: transaccionan desde una app que,
según el flag, nunca instalaron. El flag vale 1 sólo en ~3% de la cartera. No usarlo como "es digital".

### 5. `internet` no es "usa HomeBanking o la app"

Vale 0 en el 96% de las filas, mientras que `chomebanking_transacciones > 0` en el 84%. Además toma
valores 0–4 y su distribución cambia: el valor 3 pasa de 368 (202106) a **5** (202107) y vuelve a 232
(202108). Su lift (×4,0) es inestable (3,4 / 3,2 / 3,0 / **6,5**). No confiar.

### 6. Campos con dominio distinto al declarado

- `tcuentas`: el diccionario dice {0,1,2}; nunca vale 0 y vale 2 sólo en 4.778 filas (0,5%). Casi constante.
- `ccaja_seguridad`: el diccionario dice "{0,1} son los únicos dos valores posibles"; llega a **6**
  (2.896 filas > 1).
- `Visa_Fvencimiento` / `Master_Fvencimiento`: "días **para** el vencimiento", pero son **negativos**
  (mediana −1.300, sube +30 por mes). El valor absoluto es la distancia al vencimiento. Leído con el
  signo del nombre, "menor" significa "más lejos".
- `Visa_mpagado` / `Master_mpagado` vs `*_mpagospesos`: `mpagospesos` es negativo en ~92% de los
  tarjetahabientes (convención de signo de débito), mientras `mpagado > 0` sólo en 5–9%. **No** son el
  total y su parte en pesos: miden cosas distintas. Para "cuánto pagó" usar `-mpagospesos`.
- `mcomisiones` = `mcomisiones_mantenimiento + mcomisiones_otras` sólo en ~66% de las filas.
- `active_quarter = 1` con `ctrx_quarter = 0`: 2.302 filas (incoherente según el diccionario) — y es
  el subgrupo de **mayor tasa de todo el panel (8,3%)**. Vale como feature (#2).

### 7. `ccajas_depositos` desaparece en 202105

Filas con `ccajas_depositos > 0`: 6.486 / 2.175 / **0** / 831 / 5.276 / 5.828. En paralelo
`ccajas_otras > 0` sube de 2.057 a 4.973 / 6.490 / 5.966 y vuelve a 1.971 en 202107. Es una
**reclasificación** de depósitos por caja como "otras" entre abril y junio. Sumar
`ccajas_depositos + ccajas_otras` o usar sólo `ccajas_transacciones`.

### 8. `rojo_sin_acuerdo` se duplica en 202107–08 (⚠ en la tabla)

Filas con `mcuentas_saldo < 0 and cdescubierto_preacordado = 0`: 1.213 / 1.205 / 1.237 / 1.196 /
**2.377 / 2.566**. Al mismo tiempo, los que tienen acuerdo bajan ~1.000 (158.241 → 157.263) en julio.
Es decir: en julio unos mil clientes **pierden el acuerdo de descubierto** y los que quedan en rojo pasan
a "sin acuerdo". Puede ser una política del banco (recorte de líneas) y no un cambio de conducta. Con
lift ×11 en los meses de entrenamiento, en 202108 la regla marca el doble de gente y es probable que el
lift sea menor. Usarla, pero sin confiar en que la tasa de 6,9% se mantenga.

### 9. Otros drifts de escala en los meses del entrenamiento

- `mcomisiones_otras < 0` (devoluciones): 8.399 filas en 202103, decae a ~1.000 en 202107–08. Un
  modelo entrenado en marzo aprende un patrón de devoluciones que ya no existe.
- `mcomisiones_mantenimiento < 0`: 1.218 / **4.486 / 4.515** / 1.317 / 1.214 / 1.012 — devoluciones
  masivas en abril y mayo.
- `mrentabilidad < 0`: 25,7% → 17,8% de la cartera entre 202103 y 202108; mediana +48%. Usar rank por mes.
- `Visa_fultimo_cierre` y `Master_fultimo_cierre` valen **0** para el 73–77% de la cartera en 202106
  (cierre el mismo día de la foto). Cualquier feature de "días desde cierre" depende del calendario,
  no del cliente.

---

## Lectura de conjunto

1. **El estrato importa más que el lift global.** La mayoría de las features con lift global alto
   pierden casi todo dentro de `ctrx_quarter < 28` (×1,2–1,9): son la misma inactividad vista de otro
   ángulo. Las que mantienen **×7–8 entre los clientes activos** son las que anticipan a alguien que
   el modelo de actividad todavía no ve: `rojo_sin_acuerdo`, `tc_cierre_nuevo`, `ctrx_cae_fuerte_3m`,
   `canales_cero_3de3`. Y `payroll_perdido_3m` (×4,0 entre activos, ×0,7 entre inactivos) sólo
   tiene sentido para quien todavía usa el banco.
2. **Entre los inactivos**, lo que agrega es contractual: `prestamo_ultimas_cuotas` (×4,5), y
   `pasivos_margen_cero` (×1,8).
3. **Precio por uso** (`mcomisiones_mantenimiento / ctrx_quarter`, rankeado por mes) es la señal más
   estable de todo el estudio: lift del decil superior 4,17–4,25 en los cuatro meses.
4. `tc_cierre_nuevo` y `perdio_tarjeta` tienen el mecanismo más directo pero también el riesgo más
   alto de ser **casi contemporáneos** a la baja. No lo evalué (no entreno modelos); conviene
   chequearlo en el protocolo pareado.
