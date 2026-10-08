"""Reparacion con mecanismo: el banco cambio el PRECIO del mantenimiento en julio y agosto.

z701 midio PSI 0,547 / 0,660 en `mcomisiones_mantenimiento` para 202107 / 202108 (el drift mas grande de
todas las columnas de negocio): entre los que se quedan, el porcentaje que paga es plano (~30%) pero el
monto sube +22% (2.117 -> 2.598). Es un cambio de tarifa, no de conducta, y cae justo en el mes a
predecir. El modelo aprende umbrales de "cuanto paga" en 03-06 y los aplica a un precio distinto.

Deflactar TODOS los montos por mediana mensual perdio (c127, firma mayo/junio). Aca solo se tocan las
dos columnas de comision, dividiendo por la mediana mensual entre los que pagan (> 0), de modo que
"paga la tarifa estandar" valga 1,0 en todos los meses. El resto del dataset es lags12 tal cual
(se reemplazan las columnas y se recalculan sus lags/deltas).

Salida: data/competencia_01_comdefl.parquet
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
DESTINO = c.DATOS / "competencia_01_comdefl.parquet"
COLS = ["mcomisiones_mantenimiento", "mcomisiones"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); a = ap.parse_args()
    if DESTINO.exists() and not a.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute("set memory_limit = '4GB'; set threads = 2")
    derivadas = [f"{x}__{s}" for x in COLS for s in ("lag1", "delta1", "lag2", "delta2")]
    con.execute(f"create or replace view base as select * exclude ({', '.join(COLS + derivadas)}) from read_parquet('{ORIGEN}')")
    con.execute(f"create or replace view crudo as select {fe.ID}, {fe.MES}, {', '.join(COLS)} from read_parquet('{c.DATOS / 'competencia_01.parquet'}')")
    medianas = con.execute(f"""select {fe.MES}, {', '.join(f'median(case when {x} > 0 then {x} end) as {x}' for x in COLS)}
                               from crudo group by 1 order by 1""").df()
    print(medianas.to_string(index=False))
    defl = ", ".join(
        f"(c.{x} / m.{x}) :: DOUBLE as {x}" for x in COLS)
    lags = []
    for x in COLS:
        lags.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        lags.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        lags.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        lags.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = (
        f"with m as (select {fe.MES}, {', '.join(f'median(case when {x} > 0 then {x} end) as {x}' for x in COLS)} from crudo group by 1),"
        f"\nd as (select c.{fe.ID}, c.{fe.MES}, {defl} from crudo c join m using ({fe.MES})),"
        f"\nj as (select b.*, {', '.join(f'd.{x}' for x in COLS)} from base b join d using ({fe.ID}, {fe.MES}))"
        "\nselect *" + fe._fragmento(lags) + "\nfrom j" + fe.clausula_ventana("historia")
    )
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
