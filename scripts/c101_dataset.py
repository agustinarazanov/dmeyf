"""Genera data/competencia_01_fe.parquet.

El archivo NO existe en disco: el competencia_01_fe.csv que esta ahi es copia
byte a byte del base (el COPY de z402 exportaba la tabla sin tocar). Esto lo
regenera y ademas extiende el rankeo dentro del mes a TODOS los montos, no a 9:
entrenamos en marzo-junio y predecimos agosto, con la mediana de mcuentas_saldo
saltando +46% en el medio. El rank dentro del mes es el antidoto.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fe_panel as fe

ORIGEN = "data/competencia_01.parquet"
DESTINO = "data/competencia_01_fe.parquet"

# los campos de comportamiento, donde la historia dice algo
CAMPOS = [
    "ctrx_quarter", "cproductos", "mrentabilidad", "mrentabilidad_annual",
    "mcuentas_saldo", "mpayroll", "cpayroll_trx", "mcaja_ahorro", "mcuenta_corriente",
    "mtarjeta_visa_consumo", "ctarjeta_visa_transacciones",
    "mtarjeta_master_consumo", "ctarjeta_master_transacciones",
    "mcomisiones", "mcomisiones_mantenimiento", "ccomisiones_mantenimiento",
    "matm", "catm_trx", "chomebanking_transacciones", "ccallcenter_transacciones",
    "ccajas_transacciones", "mtransferencias_recibidas", "mtransferencias_emitidas",
    "mpagomiscuentas", "cpagomiscuentas", "mprestamos_personales",
    "cprestamos_personales", "mpasivos_margen", "mactivos_margen",
    "Visa_msaldototal", "Visa_mpagominimo", "Master_msaldototal",
    "active_quarter", "cliente_antiguedad",
]


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    con = fe.conectar()
    con.execute(f"""
        create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / ORIGEN}')
    """)

    columnas = [f[0] for f in con.sql("describe crudo").fetchall()]
    montos = [c for c in columnas if c.startswith("m") and c not in ("mes",)]
    campos = [c for c in CAMPOS if c in columnas]
    print(f"{len(columnas)} columnas de entrada | {len(campos)} campos con historia | "
          f"{len(montos)} montos a rankear")

    consulta = (
        "select *"
        + fe.sql_lag(campos, 1)
        + fe.sql_delta(campos, 1)
        + fe.sql_slope(campos)        # la pendiente: 'se esta apagando o creciendo'
        + fe.sql_rank(montos)         # rank dentro del mes: el antidoto a la inflacion
        + " from crudo"
        + fe.clausula_ventana()
    )
    con.execute(f"create or replace table fe as {consulta}")

    n_dsp = con.sql("select count(*) from (describe fe)").fetchone()[0]
    filas = con.sql("select count(*) from fe").fetchone()[0]
    print(f"{len(columnas)} -> {n_dsp} columnas (+{n_dsp - len(columnas)}), {filas:,} filas")

    con.execute(f"copy fe to '{raiz / DESTINO}' (format parquet, compression zstd)")
    con.close()
    mb = (raiz / DESTINO).stat().st_size / 1e6
    print(f"escrito {DESTINO}: {mb:,.0f} MB en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
