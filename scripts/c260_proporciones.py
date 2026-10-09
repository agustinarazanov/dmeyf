"""Proporciones con sentido de negocio, encima de lags12: cocientes pesos/pesos que no sufren inflacion.

El profesor sugirio "calcular proporciones" y "tomar en cuenta la inflacion". Un arbol con max_bin 31
aproxima mal un cociente de dos columnas (necesita muchas hojas para una diagonal), y un cociente
pesos/pesos es invariante a la inflacion, asi que el umbral aprendido en marzo vale en agosto.

Diccionario (diccionario de datos.csv) de las columnas que entran:
  Visa_msaldototal / Master_msaldototal   "Saldo total de la tarjeta, para ese mes"
  Visa_mlimitecompra / Master_mlimitecompra  "Limite de compra, valor muy importante"
  mtarjeta_visa_consumo / mtarjeta_master_consumo  "consumos efectuados durante el mes con la tarjeta"
  Visa_mconsumototal   "consumos (pesos y dolares) efectuados por el cliente durante ese mes"
  Visa_mpagado         "Monto total de todos los pagos efectuados por el cliente"
  Visa_mpagominimo     "pago minimo necesario para no ser moroso"
  mcuentas_saldo       "Saldo total de TODAS las cuentas del cliente"
  mcaja_ahorro / mcuenta_corriente  montos de la caja de ahorro / cuenta corriente del paquete
  mpayroll             "Monto total que le acreditaron los empleadores al cliente durante el mes"
  mprestamos_personales  "deuda restante de todos los prestamos personales"
  ctrx_quarter         "movimientos voluntarios en las cuentas (no tarjeta) en los ultimos 90 dias"
  cproductos           "familias de productos"
  (cdescubierto_preacordado es un flag, no un monto: no hay cociente posible con el acuerdo)

Nulos: Visa_*/Master_* nulos significan "no tiene esa tarjeta" (estructural); el cociente queda nulo.
Denominadores 0 -> nulo (nullif). Cada proporcion lleva lag1/delta1/lag2/delta2 como el resto.

Salida: data/competencia_01_props.parquet (lags12 + 11 x 5 = 55 columnas).
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
DESTINO = c.DATOS / "competencia_01_props.parquet"

LIM = "nullif(coalesce(Visa_mlimitecompra, 0) + coalesce(Master_mlimitecompra, 0), 0)"
PROPS = {
    "p_util_visa":        "Visa_msaldototal / nullif(Visa_mlimitecompra, 0)",
    "p_util_master":      "Master_msaldototal / nullif(Master_mlimitecompra, 0)",
    "p_util_total":       f"(coalesce(Visa_msaldototal, 0) + coalesce(Master_msaldototal, 0)) / {LIM}",
    "p_consumo_limite":   f"(coalesce(mtarjeta_visa_consumo, 0) + coalesce(mtarjeta_master_consumo, 0)) / {LIM}",
    "p_saldo_sueldo":     "mcuentas_saldo / nullif(mpayroll, 0)",
    "p_prestamo_sueldo":  "mprestamos_personales / nullif(mpayroll, 0)",
    "p_consumo_sueldo":   "Visa_mconsumototal / nullif(mpayroll, 0)",
    "p_pago_saldo_visa":  "Visa_mpagado / nullif(greatest(Visa_msaldototal, 0), 0)",   # saldo acreedor (< 0) -> nulo; la version medida dividia por el saldo con signo
    "p_minimo_saldo_visa": "Visa_mpagominimo / nullif(greatest(Visa_msaldototal, 0), 0)",
    "p_ahorro_share":     "mcaja_ahorro / nullif(abs(mcaja_ahorro) + abs(mcuenta_corriente), 0)",
    "p_trx_producto":     "ctrx_quarter / nullif(cproductos, 0)",
}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); a = ap.parse_args()
    if DESTINO.exists() and not a.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute("set memory_limit = '4GB'; set threads = 2")
    base = ", ".join(f"({expr}) :: FLOAT as {n}" for n, expr in PROPS.items())
    lags = []
    for x in PROPS:
        lags.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        lags.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        lags.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        lags.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = (
        f"with j as (select *, {base} from read_parquet('{ORIGEN}'))"
        "\nselect *" + fe._fragmento(lags) + "\nfrom j" + fe.clausula_ventana("historia")
    )
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas [{time.time() - t0:.0f}s]")
    print(con.execute(f"""select {fe.MES}, round(avg(p_util_total), 3) util, round(median(p_saldo_sueldo), 2) saldo_sueldo,
        count(p_util_visa) * 1.0 / count(*) con_visa from read_parquet('{DESTINO}') group by 1 order by 1""").df().to_string(index=False))


if __name__ == "__main__":
    main()
