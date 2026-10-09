"""Sumas, conteos, rachas y crecimientos, encima de lags12. Lo que c260 (cocientes) no cubre.

Cuatro familias, cada una con un mecanismo:
  SUMAS   el arbol ve Visa y Master por separado (22 campos gemelos); el total de deuda, de ingreso, de
          inversion o de consumo es una variable que no existe en ninguna columna y hay que construirla.
  CONTEOS "la columna que suma uno por atributo": cuantos canales usa, cuantos productos tiene activos,
          cuantos seguros, cuantas tarjetas en cierre. Un cliente que pasa de 4 canales a 1 se esta yendo.
  RACHAS  cuantas fotos seguidas (hasta 3, la historia disponible en train) se cumple algo: en rojo,
          sin sueldo, sin consumo de tarjeta, sin operar, cayendo el saldo. Es la version acumulada de
          la transicion de c231.
  CRECIMIENTO x / lag1(x) - 1 para 8 montos y contadores clave: division contra el propio pasado,
          inmune al nivel de inflacion (el delta en pesos no lo es).
Sumas y conteos llevan lag1/delta1/lag2/delta2; rachas y crecimientos ya son temporales.

Diccionario (diccionario de datos.csv) de lo que entra, ademas de lo citado en c260:
  mpayroll2 "acreditaron fuera de archivo de los empleadores"; mprestamos_prendarios/_hipotecarios "deuda restante";
  mplazo_fijo_dolares (convertido a pesos) / _pesos; minversion1_pesos / _dolares / minversion2 "monto de inversiones";
  mcuenta_debitos_automaticos "debitos automaticos en las cuentas (no tarjetas)"; mttarjeta_visa/master_debitos_automaticos
  "debitos automaticos en la tarjeta"; mpagodeservicios / mpagomiscuentas "pagos de servicios del mes";
  c*_transacciones / c*_trx "cantidad de transacciones del mes por canal" (homebanking, mobile app, call center,
  cajeros propios, cajeros ajenos, cajas de sucursal); ctarjeta_visa/master_transacciones "transacciones con la tarjeta";
  cprestamos_personales / cplazo_fijo / cinversion1 / cinversion2 / cseguro_* "cantidad vigente";
  cpayroll_trx "acreditaciones de haberes en el mes"; ccuenta_debitos_automaticos "cantidad de debitos automaticos";
  Visa_status / Master_status {0, 6, 7, 9}: abierta, en cierre, cierre avanzado, cerrada.
  OJO: tcallcenter == (ccallcenter_transacciones > 0) en todo el panel; se usa el contador, no el flag.

Salida: data/competencia_01_agregados.parquet
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe  # noqa: E402
import competencia as c  # noqa: E402

ORIGEN = c.DATOS / "competencia_01_lags12.parquet"
DESTINO = c.DATOS / "competencia_01_agregados.parquet"


def z(x):            # null-safe para sumar: Visa_*/Master_* nulos = no tiene la tarjeta
    return f"coalesce({x}, 0)"


def pos(x):          # 1 si el contador es > 0
    return f"(case when coalesce({x}, 0) > 0 then 1 else 0 end)"


SUMAS = {
    "s_saldo_tarjetas":   f"{z('Visa_msaldototal')} + {z('Master_msaldototal')}",
    "s_limite_tarjetas":  f"{z('Visa_mlimitecompra')} + {z('Master_mlimitecompra')}",
    "s_consumo_tarjetas": f"{z('mtarjeta_visa_consumo')} + {z('mtarjeta_master_consumo')}",
    "s_prestamos":        f"{z('mprestamos_personales')} + {z('mprestamos_prendarios')} + {z('mprestamos_hipotecarios')}",
    "s_deuda_total":      f"{z('mprestamos_personales')} + {z('mprestamos_prendarios')} + {z('mprestamos_hipotecarios')} + {z('Visa_msaldototal')} + {z('Master_msaldototal')}",
    "s_ingresos":         f"{z('mpayroll')} + {z('mpayroll2')}",
    "s_inversiones":      f"{z('mplazo_fijo_dolares')} + {z('mplazo_fijo_pesos')} + {z('minversion1_pesos')} + {z('minversion1_dolares')} + {z('minversion2')}",
    "s_patrimonio":       f"{z('mcuentas_saldo')} + {z('mplazo_fijo_dolares')} + {z('mplazo_fijo_pesos')} + {z('minversion1_pesos')} + {z('minversion1_dolares')} + {z('minversion2')}"
                          f" - ({z('mprestamos_personales')} + {z('mprestamos_prendarios')} + {z('mprestamos_hipotecarios')} + {z('Visa_msaldototal')} + {z('Master_msaldototal')})",
    "s_debitos_auto":     f"{z('mcuenta_debitos_automaticos')} + {z('mttarjeta_visa_debitos_automaticos')} + {z('mttarjeta_master_debitos_automaticos')}",
    "s_pagos_servicios":  f"{z('mpagodeservicios')} + {z('mpagomiscuentas')}",
}
CANALES = ["chomebanking_transacciones", "cmobile_app_trx", "ccallcenter_transacciones", "catm_trx", "catm_trx_other", "ccajas_transacciones"]
PRODUCTOS = ["ctarjeta_visa_transacciones", "ctarjeta_master_transacciones", "cprestamos_personales", "cplazo_fijo",
             "cinversion1", "cinversion2", "cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales",
             "cpayroll_trx", "ccuenta_debitos_automaticos", "ctransferencias_emitidas", "ccheques_depositados"]
CONTEOS = {
    "n_canales":          " + ".join(pos(x) for x in CANALES),
    "n_productos_activos": " + ".join(pos(x) for x in PRODUCTOS),
    "n_seguros":          " + ".join(z(x) for x in ["cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"]),
    "n_tarjetas_cierre":  "(case when Visa_status in (6, 7, 9) then 1 else 0 end) + (case when Master_status in (6, 7, 9) then 1 else 0 end)",
    "n_trx_canales":      " + ".join(z(x) for x in CANALES),
}
RACHAS = {   # condicion sobre la foto; la racha cuenta fotos consecutivas hacia atras (0..3)
    "r_rojo":        "coalesce(mcuentas_saldo, 0) < 0",
    "r_sin_sueldo":  "coalesce(cpayroll_trx, 0) = 0",
    "r_sin_consumo": f"{z('mtarjeta_visa_consumo')} + {z('mtarjeta_master_consumo')} = 0",
    "r_sin_operar":  " + ".join(z(x) for x in CANALES) + " = 0",
    "r_cae_saldo":   "mcuentas_saldo < lag(mcuentas_saldo, 1) over historia",
}
CRECIMIENTO = ["mcuentas_saldo", "mpayroll", "ctrx_quarter", "mprestamos_personales", "mcaja_ahorro",
               "mrentabilidad", "mcomisiones", "Visa_msaldototal"]




def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); a = ap.parse_args()
    if DESTINO.exists() and not a.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute("set memory_limit = '4GB'; set threads = 2")
    W = fe.clausula_ventana("historia")
    # paso 1: sumas, conteos, condiciones de racha (0/1), crecimientos
    paso1 = [f"({e}) :: FLOAT as {n}" for n, e in {**SUMAS, **CONTEOS}.items()]
    paso1 += [f"(case when {e} then 1 else 0 end) :: TINYINT as {n}__c" for n, e in RACHAS.items()]
    paso1 += [f"({x} / nullif(lag({x}, 1) over historia, 0) - 1) :: FLOAT as {x}__crec1" for x in CRECIMIENTO]
    # paso 2: rachas encadenadas y lags/deltas de sumas y conteos
    paso2 = []
    for n in RACHAS:
        paso2.append(f"(case when {n}__c = 1 then 1 + (case when lag({n}__c, 1) over historia = 1 then 1 + "
                     f"(case when lag({n}__c, 2) over historia = 1 then 1 + (case when lag({n}__c, 3) over historia = 1 then 1 else 0 end) else 0 end) else 0 end) else 0 end) :: TINYINT as {n}")
    for x in list(SUMAS) + list(CONTEOS):
        paso2.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        paso2.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        paso2.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        paso2.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = (
        f"with p1 as (select *{fe._fragmento(paso1)}\nfrom read_parquet('{ORIGEN}'){W})"
        f"\nselect * exclude ({', '.join(n + '__c' for n in RACHAS)}){fe._fragmento(paso2)}\nfrom p1{W}"
    )
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas [{time.time() - t0:.0f}s]")
    print(con.execute(f"""select {fe.MES}, round(avg(n_canales), 2) canales, round(avg(n_productos_activos), 2) productos,
        round(avg(r_rojo), 2) r_rojo, round(avg(r_sin_operar), 2) r_sin_operar, round(median(s_deuda_total)) deuda,
        round(avg(mcuentas_saldo__crec1), 2) crec_saldo from read_parquet('{DESTINO}') group by 1 order by 1""").df().to_string(index=False))
    print(con.execute(f"""select r_rojo, count(*) n, round(avg(case when {fe.CLASE} = 'BAJA+2' then 1.0 else 0 end) * 100, 2) pct_baja2
        from read_parquet('{DESTINO}') where {fe.MES} = 202106 group by 1 order by 1""").df().to_string(index=False))


if __name__ == "__main__":
    main()
