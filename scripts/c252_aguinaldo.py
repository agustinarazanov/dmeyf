"""Reparacion con mecanismo: el aguinaldo de junio contamina los lags de AGOSTO.

En 202106 mpayroll trae el medio aguinaldo (mediana +36% contra abril). Eso no molesta al modelo
sin historia, pero con lags si: en agosto, lag2 = junio y delta2 = agosto - junio da una caida del
-35% a TODOS los que cobran sueldo (medido: mediana de delta2/mpayroll en 202108 = -35,1%), un patron
que en los meses de entrenamiento no existe (en junio la misma columna da +36%). Es un artefacto del
calendario que cae justo en el mes a predecir.

Correccion minima: mpayroll y mpayroll2 de 202106 divididos por 1,5 (el SAC es medio sueldo) ANTES de
calcular lags y deltas. Nada mas cambia respecto de lags12.

Salida: data/competencia_01_aguinaldo.parquet
"""
import argparse, sys, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe
import competencia as c
ORIGEN = c.DATOS / "competencia_01_lags12.parquet"
DESTINO = c.DATOS / "competencia_01_aguinaldo.parquet"
COLS = ["mpayroll", "mpayroll2"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); a = ap.parse_args()
    if DESTINO.exists() and not a.forzar: raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time(); con = fe.conectar(); con.execute("set memory_limit = '4GB'; set threads = 2")
    derivadas = [f"{x}__{s}" for x in COLS for s in ("lag1", "delta1", "lag2", "delta2")]
    con.execute(f"create or replace view base as select * exclude ({', '.join(COLS + derivadas)}) from read_parquet('{ORIGEN}')")
    con.execute(f"create or replace view crudo as select {fe.ID}, {fe.MES}, {', '.join(COLS)} from read_parquet('{c.DATOS / 'competencia_01.parquet'}')")
    corr = ", ".join(f"case when {fe.MES} = 202106 then {x} / 1.5 else {x} end as {x}" for x in COLS)
    lags = []
    for x in COLS:
        lags += [f"lag({x}, 1) over historia :: FLOAT as {x}__lag1", f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1",
                 f"lag({x}, 2) over historia :: FLOAT as {x}__lag2", f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2"]
    sql = (f"with d as (select {fe.ID}, {fe.MES}, {corr} from crudo),"
           f"\nj as (select b.*, {', '.join(f'd.{x}' for x in COLS)} from base b join d using ({fe.ID}, {fe.MES}))"
           "\nselect *" + fe._fragmento(lags) + "\nfrom j" + fe.clausula_ventana("historia"))
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    chk = con.execute(f"select {fe.MES}, median(mpayroll__delta2 / mpayroll) from read_parquet('{DESTINO}') where mpayroll > 0 and {fe.MES} >= 202106 group by 1 order by 1").fetchall()
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas | mediana delta2/mpayroll corregida: {chk} [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
