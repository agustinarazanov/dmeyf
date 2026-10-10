"""Las features del analisis de Miranda (v101), llevadas al modelo.

El FE a granel de z402 (147 columnas) perdio 40 M. Esto es lo contrario: 14
columnas, cada una con una hipotesis de negocio detras, sacadas de lo que el EDA
de v101 encontro sobre los que se van.

Las que traen informacion NUEVA de verdad son las acumuladas —meses_en_rojo,
meses_sin_trx—, que una fila sola no puede tener. El resto son cambios de
sistema de coordenadas: interacciones que al arbol le costarian dos cortes.

  rojo                 mcuentas_saldo < 0. En 52-58% de los BAJA+2 contra ~20%
                       de los CONTINUA, estable los cuatro meses.
  meses_en_rojo        cuantos meses lleva en rojo (acumulado, sin mirar futuro)
  rojo_sin_acuerdo     en rojo Y sin descubierto preacordado. Los que se quedan
                       tienen 95,5% de acuerdo; los que se van, 59-89%.
  primacia             sueldo acreditado o servicios domiciliados. El gradiente
                       por patron va de 4% a 62%.
  usa_cajero           catm_trx NO cuenta los cajeros ajenos: 13.634 clientes
                       usan solo ajenos y figuran como 'no usa cajero'.
  tc_cerrando          Visa/Master_status en {6,7,9}, la tarjeta cerrandose.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fe_panel as fe

DESTINO = "data/competencia_01_v101.parquet"

SQL = """
select *
  -- saldo en rojo: la señal grande que el EDA encontro
  , (mcuentas_saldo < 0)::int                                      as v_rojo
  , sum((mcuentas_saldo < 0)::int) over historia                   as v_meses_en_rojo
  , (mcuentas_saldo < 0 and ifnull(cdescubierto_preacordado,0) = 0)::int as v_rojo_sin_acuerdo
  , ratio_seguro(mcuentas_saldo, abs(mcuentas_saldo) + 1)          as v_rojo_intensidad
  -- primacia: sueldo y servicios domiciliados
  , (ifnull(cpayroll_trx,0) + ifnull(cpayroll2_trx,0) > 0)::int    as v_cobra_sueldo
  , (ifnull(cpagomiscuentas,0) > 0)::int                           as v_domicilia
  , ((ifnull(cpayroll_trx,0) + ifnull(cpayroll2_trx,0) > 0)::int
     + (ifnull(cpagomiscuentas,0) > 0)::int)                       as v_anclas
  -- actividad: nivel, historia y apagon
  , sum((ifnull(ctrx_quarter,0) = 0)::int) over historia           as v_meses_sin_trx
  , min(ifnull(ctrx_quarter,0)) over historia                      as v_ctrx_minimo
  , ctrx_quarter - lag(ctrx_quarter) over historia                 as v_ctrx_delta
  -- cajeros: catm_trx no cuenta los ajenos
  , (ifnull(catm_trx,0) + ifnull(catm_trx_other,0) > 0)::int       as v_usa_cajero
  -- tarjetas: estado y mora, colapsando las dos
  , (greatest(ifnull(Visa_status,0), ifnull(Master_status,0)) in (6,7,9))::int as v_tc_cerrando
  , greatest(ifnull(Visa_delinquency,0), ifnull(Master_delinquency,0))        as v_tc_mora
  , suma_sin_null(Visa_msaldototal, Master_msaldototal)             as v_tc_saldo
from crudo
"""


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    con = fe.conectar()
    con.execute(f"""create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / 'data/competencia_01.parquet'}')""")
    antes = con.sql("select count(*) from (describe crudo)").fetchone()[0]
    con.execute(f"create or replace table v101 as {SQL} {fe.clausula_ventana()}")
    dsp = con.sql("select count(*) from (describe v101)").fetchone()[0]
    print(f"{antes} -> {dsp} columnas (+{dsp - antes})")

    print("\nlift de cada flag contra la tasa base de BAJA+2, en los meses etiquetados:")
    base = con.sql("""select avg((clase_ternaria='BAJA+2')::int) from v101
                      where clase_ternaria is not null""").fetchone()[0]
    for col in ("v_rojo", "v_rojo_sin_acuerdo", "v_meses_en_rojo >= 3", "v_anclas = 0",
                "v_meses_sin_trx >= 2", "v_tc_cerrando", "v_usa_cajero = 0"):
        r = con.sql(f"""select avg((clase_ternaria='BAJA+2')::int) t, count(*) n from v101
                        where clase_ternaria is not null and {col}""").fetchone()
        print(f"  {col:24} {r[1]:>8,} filas  tasa {r[0]:.3%}  lift {r[0]/base:>5.2f}x")

    con.execute(f"copy v101 to '{raiz / DESTINO}' (format parquet, compression zstd)")
    con.close()
    print(f"\nescrito {DESTINO} en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
