"""¿Perder productos ayuda al modelo?

Unica candidata de la teoria de negocio que paso el filtro duro: el lift SUBE al
condicionar por clientes activos (2,21 -> 2,10 global, y 6,94 -> 7,34 para los
que pierden 2+). Casi todo lo demas que probamos era inactividad disfrazada.

Es `relationship breadth` de la literatura de attrition bancaria, y es
informacion que un arbol NO puede derivar de una fila sola: necesita el mes
anterior. `cproductos` esta como nivel; el delta no.

Ventanas FIJAS de 3 meses, no unbounded: con unbounded la feature no significa
lo mismo en 202103 que en 202108 (ese bug ya nos costo una ronda).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fe_panel as fe
import competencia as c

DESTINO = "data/competencia_01_prod.parquet"
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
TARGET, PESO, NBR, CORTE = "pesos", 0.25, 250, 11_000
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c121_productos"
CARPETA.mkdir(parents=True, exist_ok=True)

SQL = """
with paso1 as (
  select *
    , lag(cproductos) over (partition by numero_de_cliente order by foto_mes) as p_lag
  from crudo
)
select * exclude (p_lag)
  , cproductos - p_lag                                        as p_delta
  , cproductos - min(cproductos) over v3                      as p_vs_min3
  , max(cproductos) over v3 - cproductos                      as p_caida_desde_max3
  , sum(greatest(0, coalesce(p_lag, cproductos) - cproductos)) over v3 as p_perdidos_3m
  , (cproductos < coalesce(p_lag, cproductos))::int           as p_perdio
from paso1
window
  v3 as (partition by numero_de_cliente order by foto_mes rows between 2 preceding and current row)
"""


def construir(raiz: Path) -> None:
    con = fe.conectar()
    con.execute(f"""create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / 'data/competencia_01.parquet'}')""")
    con.execute(f"create or replace table prod as {SQL}")
    base = con.sql("""select avg((clase_ternaria='BAJA+2')::int) from prod
                      where clase_ternaria is not null""").fetchone()[0]
    act = con.sql("""select avg((clase_ternaria='BAJA+2')::int) from prod
                     where clase_ternaria is not null and ctrx_quarter>=28""").fetchone()[0]
    print(f"{'feature':26} {'cob.08':>7} {'lift':>6} {'lift activos':>13}")
    for col in ("p_perdio = 1", "p_delta <= -2", "p_perdidos_3m >= 1", "p_perdidos_3m >= 2",
                "p_caida_desde_max3 >= 1", "p_caida_desde_max3 >= 2"):
        r = con.sql(f"""select
            (select round(100.0*avg(({col})::int),1) from prod where foto_mes=202108) cob,
            round(avg((clase_ternaria='BAJA+2')::int)/{base},2) l,
            (select round(avg((clase_ternaria='BAJA+2')::int)/{act},2) from prod
             where clase_ternaria is not null and ctrx_quarter>=28 and {col}) la
            from prod where clase_ternaria is not null and {col}""").fetchone()
        print(f"  {col:24} {r[0]:>6}% {r[1]:>6} {str(r[2]):>13}")
    con.execute(f"copy prod to '{raiz / DESTINO}' (format parquet, compression zstd)")
    con.close()


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    construir(raiz)
    np.random.seed(c.SEMILLAS[0])
    sem = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar("competencia_01_prod.parquet")
    todas = c.columnas_predictoras(data)
    nuevas = [x for x in todas if x.startswith("p_")]
    sets = {"base": [x for x in todas if not x.startswith("p_")], "con_productos": todas}
    print(f"\n{len(sets['base'])} columnas base + {len(nuevas)} de productos\n")

    res = {}
    for etq, pred in sets.items():
        obs = []
        for meses, mv in FOLDS:
            X, y, w = c.preparar(data, meses, TARGET, PESO)
            X = X[pred]
            val = data[data[c.fe.MES] == mv]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            for s in sem:
                cl = c.clave(PARAMS, meses, TARGET, PESO, f"{etq}{len(pred)}", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"{etq}_{mv}_{s}_{cl}")
                obs.append(c.ganancia_acumulada(m.predict(val[pred]), es)[CORTE - 1])
        res[etq] = np.array(obs)
        print(f"  {etq:14} {len(pred):>3} cols  media={np.mean(obs)/1e6:>6,.1f}M  "
              f"[{time.time()-t0:.0f}s]")

    d = res["con_productos"] - res["base"]
    print(f"\npareado por (fold, semilla): {d.mean()/1e6:+.1f} M  gana en {(d>0).sum()}/{len(d)}  "
          f"p={wilcoxon(d).pvalue:.4f}")
    pd.DataFrame(res).to_parquet(CARPETA / "por_semilla.parquet", index=False)

    import lightgbm as lgb
    mod = sorted(CARPETA.glob("con_productos_202106_261431_*.txt"))
    if mod:
        m = lgb.Booster(model_file=str(mod[0]))
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        imp = (imp / imp.sum() * 100).sort_values(ascending=False)
        print(f"\ndonde cayeron, de {len(imp)} columnas:")
        for n in nuevas:
            if n in imp.index:
                print(f"  {n:22} {imp[n]:>5.2f}%  puesto {list(imp.index).index(n)+1}")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
