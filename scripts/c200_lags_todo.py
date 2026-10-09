"""
NOTA (review monday-45, 9-oct): ccajas_depositos tambien esta casi muerta en 202106 (831 no-cero contra 2.175-6.486 los
otros meses); el delta2 de agosto apunta a junio y queda inflado para ~5.800 clientes. 0,23% de la ganancia: no se
regenero el dataset a dos dias del cierre. Si se vuelve a construir, anular 202106 tambien.
Dataset con lag 1, delta 1, lag 2 y delta 2 de TODAS las predictoras.

Es la receta que Denicolay dejo en Zulip (J-Clase 08 > En Limpio, 2026-10-06):
"es superador agregar al dataset lags y delta lags de orden 1 y 2", y la que
Ramirez (farpp8/dmeyf2026-competencia01, DECISIONES.md D06) midio en publico:
86,74 -> 98,59 con los mismos hiperparametros. Nuestro competencia_01_fe.parquet
tiene historia de 21 variables, no de 150, y nunca tuvo lag 2.

Tambien aplica el unico arreglo de data quality de esa receta:
ccajas_depositos -> NULL en 202105 (todos sus valores son cero ese mes).

Las columnas nuevas van en FLOAT (4 bytes): 600 columnas x 983k filas en DOUBLE
no entran comodas en 16 GB junto con el Dataset de LightGBM.

Salida: data/competencia_01_lags12.parquet
"""
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe  # noqa: E402
import competencia as c  # noqa: E402

ORIGEN = c.DATOS / "competencia_01.parquet"
DESTINO = c.DATOS / "competencia_01_lags12.parquet"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); args = ap.parse_args()
    if DESTINO.exists() and not args.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; regenerarlo invalida los modelos cacheados que lo usaron. --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute(f"create or replace view base as select * from read_parquet('{ORIGEN}')")
    campos = fe.columnas_numericas(con, "base", excluir=tuple(c.ROTAS_EN_202108))
    campos = [x for x in campos if x not in c.FUERA_DE_X]
    print(f"  {len(campos)} predictoras -> {4 * len(campos)} columnas de historia")

    nuevas = []
    for x in campos:
        nuevas.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        nuevas.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        nuevas.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        nuevas.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")

    # La anulacion va en una CTE para que los lags/deltas se calculen sobre la columna ya
    # corregida. OJO: el archivo generado el 2026-10-07 (el que entreno c201) tenia los lags
    # sobre los ceros crudos de 202105; esta version los corrige. Regenerar cambia el dataset.
    sql = (
        "with limpia as (select * exclude (ccajas_depositos)"
        f"\n  , case when {fe.MES} = 202105 then null else ccajas_depositos end as ccajas_depositos"
        "\n  from base)"
        "\nselect *" + fe._fragmento(nuevas) + "\nfrom limpia" + fe.clausula_ventana("historia")
    )
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
