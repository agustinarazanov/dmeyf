"""Dataset B5+B7+A5 del backlog: reparaciones de datos + transiciones + recencia, encima de lags12.

Sobre competencia_01.parquet (150 crudas) se aplica, en este orden:

1. REPARACIONES (Ramirez D01, Denicolay En Limpio 1.1):
   - ccajas_depositos NULL en 202105 (todo cero ese mes) y ccajas_depositos + ccajas_otras sumadas en
     `ccajas_dep_otras` (reclasificacion abril-junio) -> se conservan las dos crudas y se agrega la suma
   - *_mconsumototal eliminadas (duplican *_mconsumospesos)
   - *_Fvencimiento < -1.000.000 -> NULL (centinela)
   - *_mfinanciacion_limite > 10 x *_mlimitecompra -> NULL
2. FLAG DE BLOQUE NULO: `bloque_nulo` = 1 si las variables de negocio del bloque estan todas en NULL a la
   vez (35.301 filas, Zulip [186635]); sus lags/deltas se dejan como salen pero el flag los explica.
3. TRANSICIONES (Clara Rodriguez, Zulip [189814]), binarias e inmunes a la inflacion, para 12 montos
   clave: `x__a_cero` (x = 0 y lag1 > 0), `x__a_negativo` (x < 0 y lag1 >= 0), `x__desde_cero` (x > 0 y lag1 = 0).
4. RECENCIA (meses desde el ultimo evento, acotada a la historia disponible): sueldo, transaccion en cuentas
   (ctrx_quarter > 0), saldo positivo, consumo de tarjeta. Se calcula como "cuantas fotos consecutivas
   hacia atras sin el evento", con ventana fija de 3 (0..3) para que signifique lo mismo en todos los meses.
5. LAGS/DELTAS 1 y 2 de todas las predictoras reparadas, como c200.

Salida: data/competencia_01_rep.parquet
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe  # noqa: E402
import competencia as c  # noqa: E402

ORIGEN = c.DATOS / "competencia_01.parquet"
DESTINO = c.DATOS / "competencia_01_rep.parquet"

MONTOS_TRANSICION = ["mcaja_ahorro", "mcuentas_saldo", "mpasivos_margen", "mpayroll", "mcuenta_corriente",
                     "mtarjeta_visa_consumo", "mtarjeta_master_consumo", "mprestamos_personales",
                     "mplazo_fijo_pesos", "mtransferencias_recibidas", "mcomisiones_mantenimiento", "mrentabilidad"]
EVENTOS_RECENCIA = {
    "sueldo": "coalesce(cpayroll_trx, 0) > 0",
    "trx": "coalesce(ctrx_quarter, 0) > 0",
    "saldo_pos": "coalesce(mcuentas_saldo, 0) > 0",
    "consumo_tc": "coalesce(mtarjeta_visa_consumo, 0) + coalesce(mtarjeta_master_consumo, 0) > 0",
}
BLOQUE = ["mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen",
          "mcuenta_corriente", "mcaja_ahorro", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo",
          "mtarjeta_master_consumo", "mpayroll", "mcomisiones_mantenimiento", "mcomisiones_otras"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); args = ap.parse_args()
    if DESTINO.exists() and not args.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute("set memory_limit = '3GB'; set threads = 2")   # convive con un entrenamiento en curso
    con.execute(f"create or replace view base as select * from read_parquet('{ORIGEN}')")
    campos = [x for x in fe.columnas_numericas(con, "base", excluir=tuple(c.ROTAS_EN_202108)) if x not in c.FUERA_DE_X]
    sacar = [x for x in campos if x.endswith("_mconsumototal")]
    campos_rep = [x for x in campos if x not in sacar] + ["ccajas_dep_otras", "bloque_nulo"]

    rep = (
        "with reparada as (\n"
        f"  select * exclude ({', '.join(sacar + ['ccajas_depositos', 'Visa_Fvencimiento', 'Master_Fvencimiento', 'Visa_mfinanciacion_limite', 'Master_mfinanciacion_limite'])})\n"
        f"    , case when {fe.MES} = 202105 then null else ccajas_depositos end as ccajas_depositos\n"
        f"    , coalesce(case when {fe.MES} = 202105 then null else ccajas_depositos end, 0) + coalesce(ccajas_otras, 0) as ccajas_dep_otras\n"
        "    , case when Visa_Fvencimiento < -1000000 then null else Visa_Fvencimiento end as Visa_Fvencimiento\n"
        "    , case when Master_Fvencimiento < -1000000 then null else Master_Fvencimiento end as Master_Fvencimiento\n"
        "    , case when Visa_mfinanciacion_limite > 10 * coalesce(Visa_mlimitecompra, 0) then null else Visa_mfinanciacion_limite end as Visa_mfinanciacion_limite\n"
        "    , case when Master_mfinanciacion_limite > 10 * coalesce(Master_mlimitecompra, 0) then null else Master_mfinanciacion_limite end as Master_mfinanciacion_limite\n"
        f"    , case when {' and '.join(f'{x} is null' for x in BLOQUE)} then 1 else 0 end as bloque_nulo\n"
        "  from base)"
    )
    trans = []
    for x in MONTOS_TRANSICION:
        trans += [
            f"(coalesce({x}, 0) = 0 and lag({x}, 1) over historia > 0) :: TINYINT as {x}__a_cero",
            f"(coalesce({x}, 0) < 0 and lag({x}, 1) over historia >= 0) :: TINYINT as {x}__a_negativo",
            f"(coalesce({x}, 0) > 0 and lag({x}, 1) over historia = 0) :: TINYINT as {x}__desde_cero",
        ]
    rec = []
    for nombre, cond in EVENTOS_RECENCIA.items():
        # meses consecutivos hacia atras sin el evento, acotado a 3 (ventana fija): 0 = este mes lo tuvo
        rec.append(
            f"case when ({cond}) then 0"
            f" when lag(({cond})::int, 1) over historia = 1 then 1"
            f" when lag(({cond})::int, 2) over historia = 1 then 2"
            f" else 3 end :: TINYINT as rec_{nombre}"
        )
    lags = []
    for x in campos_rep:
        lags.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        lags.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        lags.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        lags.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = rep + "\nselect *" + fe._fragmento(trans + rec + lags) + "\nfrom reparada" + fe.clausula_ventana("historia")
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    bn = con.execute(f"select sum(bloque_nulo) from read_parquet('{DESTINO}')").fetchone()[0]
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas | bloque_nulo en {bn:,} filas | "
          f"{len(trans)} transiciones, {len(rec)} recencias, {len(lags)} lags/deltas [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
