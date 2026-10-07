"""Features v2: las de v101 corregidas + las que propuso autoresearch.

Tres arreglos sobre la version anterior, los tres verificados:

1. VENTANAS FIJAS de 3 meses en vez de `unbounded preceding`. Con unbounded la
   feature NO ES LA MISMA en cada mes: 'rojo >= 3 meses' cubre 0% en 202103 y
   19,8% en 202108 por pura construccion. Con ventana fija queda ~10% estable.
   Era un defecto mio, y explica mejor que mi hipotesis anterior por que las
   features historicas no median nada en los folds.

2. MORA: `*_delinquency` tambien esta contaminada en 202105/202108. Las
   3.222/3.249 filas con Finiciomora=0 son exactamente delinquency=1, y tienen
   la mitad de riesgo (lift 1,30 contra 2,82 de la mora de verdad). Sacar
   F*iniciomora y usar delinquency crudo MUEVE el problema, no lo arregla.
   Se usa `delinquency=1 and coalesce(Finiciomora,1) > 0`.

3. Se agregan las candidatas de autoresearch que mantienen senal DENTRO de los
   clientes activos, que es lo que distingue una senal real de un sinonimo de
   inactividad.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fe_panel as fe

DESTINO = "data/competencia_01_v2.parquet"

SQL = """
with b as (
  select *
    , coalesce(chomebanking_transacciones,0) + coalesce(cmobile_app_trx,0)
      + coalesce(ccallcenter_transacciones,0) + coalesce(ccajas_transacciones,0)
      + coalesce(catm_trx,0) + coalesce(catm_trx_other,0)
      + coalesce(ctarjeta_debito_transacciones,0)                      as canales_mes
    , greatest(coalesce(Visa_status,0), coalesce(Master_status,0))     as tc_status_max
    , coalesce(cpayroll_trx,0) + coalesce(cpayroll2_trx,0)             as payroll_trx
    , coalesce(ccuenta_debitos_automaticos,0)
      + coalesce(ctarjeta_visa_debitos_automaticos,0)
      + coalesce(ctarjeta_master_debitos_automaticos,0)                as debaut
    , coalesce(cpagodeservicios,0) + coalesce(cpagomiscuentas,0)       as pagos_serv
    , (coalesce(Visa_delinquency,0)=1 and coalesce(Visa_Finiciomora,1)>0)::int
      + (coalesce(Master_delinquency,0)=1 and coalesce(Master_Finiciomora,1)>0)::int as mora_vieja
  from crudo
)
select * exclude (canales_mes, tc_status_max, payroll_trx, debaut, pagos_serv, mora_vieja)
  -- saldo en rojo
  , (mcuentas_saldo < 0)::int                                          as w_rojo
  , (mcuentas_saldo < 0 and coalesce(cdescubierto_preacordado,0)=0)::int as w_rojo_sin_acuerdo
  , sum((mcuentas_saldo < 0)::int) over v3                             as w_rojo_3m
  -- apagon de actividad (ventanas FIJAS)
  , canales_mes                                                        as w_canales_mes
  , sum((canales_mes = 0)::int) over v3                                as w_canales_cero_3m
  , (ctrx_quarter = 0 and coalesce(lag(ctrx_quarter) over v, 1) > 0)::int as w_ctrx_cae_a_cero
  , (active_quarter = 1 and ctrx_quarter = 0)::int                     as w_aq1_ctrx0
  , (ctrx_quarter < 0.6 * nullif(max(ctrx_quarter) over v2, 0))::int   as w_ctrx_cae_fuerte
  , min(ctrx_quarter) over v3                                          as w_ctrx_min_3m
  -- tarjetas: estado y mora SIN contaminar
  , (tc_status_max in (6,7,9))::int                                    as w_tc_cerrando
  , (tc_status_max >= 6 and coalesce(lag(tc_status_max) over v, 0) = 0)::int as w_tc_cierre_nuevo
  , mora_vieja                                                         as w_mora_vieja
  , suma_sin_null(Visa_msaldototal, Master_msaldototal)                as w_tc_saldo
  -- primacia
  , (payroll_trx > 0)::int                                             as w_cobra_sueldo
  , ((payroll_trx>0)::int + (debaut>0)::int + (pagos_serv>0)::int)     as w_anclas
  -- costo relativo: comision por transaccion, rankeada dentro del mes
  , percent_rank() over (partition by foto_mes
      order by ratio_seguro(mcomisiones_mantenimiento, ctrx_quarter))  as w_fee_por_trx_rk
  , (coalesce(catm_trx,0) + coalesce(catm_trx_other,0) > 0)::int       as w_usa_cajero
from b
window
  v  as (partition by numero_de_cliente order by foto_mes),
  v2 as (partition by numero_de_cliente order by foto_mes rows between 2 preceding and 1 preceding),
  v3 as (partition by numero_de_cliente order by foto_mes rows between 2 preceding and current row)
"""


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    con = fe.conectar()
    con.execute(f"""create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / 'data/competencia_01.parquet'}')""")
    antes = con.sql("select count(*) from (describe crudo)").fetchone()[0]
    con.execute(f"create or replace table v2 as {SQL}")
    dsp = con.sql("select count(*) from (describe v2)").fetchone()[0]
    print(f"{antes} -> {dsp} columnas (+{dsp - antes})\n")

    base = con.sql("""select avg((clase_ternaria='BAJA+2')::int) from v2
                      where clase_ternaria is not null""").fetchone()[0]
    print(f"{'feature':26} {'cob.08':>7} {'lift':>6}  {'lift si ctrx>=28':>17}")
    for col in ("w_rojo_sin_acuerdo", "w_rojo_3m >= 2", "w_ctrx_cae_a_cero", "w_aq1_ctrx0",
                "w_ctrx_cae_fuerte = 1", "w_canales_cero_3m >= 3", "w_tc_cierre_nuevo",
                "w_tc_cerrando", "w_mora_vieja >= 1", "w_anclas = 0",
                "w_fee_por_trx_rk >= 0.9"):
        r = con.sql(f"""select
            (select round(100.0*avg(({col})::int),1) from v2 where foto_mes=202108) cob,
            round(avg((clase_ternaria='BAJA+2')::int)/{base},2) lift,
            (select round(avg((clase_ternaria='BAJA+2')::int)
                / (select avg((clase_ternaria='BAJA+2')::int) from v2
                   where clase_ternaria is not null and ctrx_quarter>=28),2)
             from v2 where clase_ternaria is not null and ctrx_quarter>=28 and {col}) lift_act
            from v2 where clase_ternaria is not null and {col}""").fetchone()
        print(f"  {col:24} {r[0]:>6}% {r[1]:>6}  {str(r[2]):>17}")

    con.execute(f"copy v2 to '{raiz / DESTINO}' (format parquet, compression zstd)")
    con.close()
    print(f"\nescrito {DESTINO} en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
