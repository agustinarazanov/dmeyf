"""Dataset B6 del backlog: lags y deltas 1 y 2 sobre MONTOS RANKEADOS por mes.

Un delta en pesos entre junio (aguinaldo) y julio mide calendario; un delta de percentil
dentro del mes mide conducta. Abregu (B4 + B5 de su combo final) hace los lags sobre el
ranking. Aca: los 73 montos (prefijo `m`, mas los Visa_m*/Master_m*) se reemplazan por su
percent_rank dentro de foto_mes (los NULL quedan NULL, no van a 1,0), y lag1/delta1/lag2/delta2
se calculan sobre esa version para los montos y sobre el valor crudo para el resto.
Incluye la correccion de ccajas_depositos (NULL en 202105) antes de los lags.

Salida: data/competencia_01_lagsrank.parquet
"""
import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import fe_panel as fe  # noqa: E402
import competencia as c  # noqa: E402

ORIGEN = c.DATOS / "competencia_01.parquet"
DESTINO = c.DATOS / "competencia_01_lagsrank.parquet"


def es_monto(col: str) -> bool:
    return col.startswith("m") or col.startswith("Visa_m") or col.startswith("Master_m")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); args = ap.parse_args()
    if DESTINO.exists() and not args.forzar:
        raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time()
    con = fe.conectar()
    con.execute(f"create or replace view base as select * from read_parquet('{ORIGEN}')")
    campos = [x for x in fe.columnas_numericas(con, "base", excluir=tuple(c.ROTAS_EN_202108)) if x not in c.FUERA_DE_X]
    montos = [x for x in campos if es_monto(x)]
    print(f"  {len(campos)} predictoras, {len(montos)} montos rankeados por mes")

    # 1) limpia: ccajas_depositos NULL en 202105
    # 2) rankeada: cada monto reemplazado por su percent_rank dentro del mes (NULL se queda NULL)
    rank_expr = [
        f"case when {x} is null then null else percent_rank() over (partition by {fe.MES} order by {x}) end as {x}"
        for x in montos
    ]
    nuevas = []
    for x in campos:
        nuevas.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        nuevas.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        nuevas.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        nuevas.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = (
        "with limpia as (select * exclude (ccajas_depositos)"
        f"\n  , case when {fe.MES} = 202105 then null else ccajas_depositos end as ccajas_depositos"
        "\n  from base),"
        f"\nrankeada as (select * exclude ({', '.join(montos)})" + fe._fragmento(rank_expr) + "\n  from limpia)"
        "\nselect *" + fe._fragmento(nuevas) + "\nfrom rankeada" + fe.clausula_ventana("historia")
    )
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
