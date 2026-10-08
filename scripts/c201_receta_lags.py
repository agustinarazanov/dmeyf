"""Lags de todo (c200) + la receta de hiperparametros de Denicolay.

HIPOTESIS: la entrega base (150 columnas, params de z701, 93,24 publico a 10.000
con 20 archivos) esta debajo de la linea de muerte (~100) porque le falta la
historia de todas las variables y porque sus hiperparametros son de otra tarea.
Dos compañeros con las mismas reglas lo midieron en publico: Ramirez
86,74 -> 98,59 (lags de todo) -> 103,18 (+ esta receta); Abregu 88 -> 110.

La receta, textual de J-Clase 08 > En Limpio (2026-10-06) y de
farpp8/dmeyf2026-competencia01 (06_receta_denicolay.py):
  max_bin 31 - min_data_in_leaf 0 - feature_fraction 0,5 - feature_fraction_bynode 0,2
  learning_rate 0,005 - sin bagging - 83 hojas - 1.000 rondas
  min_sum_hessian_in_leaf = 12,791817 x filas / 326.184

Lo que NO cambia respecto de la entrega, para que el delta sea atribuible:
target pesos 0,25, meses 202103-202106, las mismas 20 semillas de c107,
ensamble por rank, FUERA_DE_X.

Cada modelo se cachea por semilla; los scores de 202108 se guardan por semilla
para re-cortar sin reentrenar. Se puede interrumpir y retomar.

    python c201_receta_lags.py --semillas 5        # entrena/carga 5 y escribe envios
    python c201_receta_lags.py --semillas 20 --cortes 10000 11000 11500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))   # la raiz del repo: competencia.py y fe_panel.py
import competencia as c  # noqa: E402

NOMBRE = "c201_receta_lags"
DATASET = "competencia_01_lags12.parquet"
MESES = [202103, 202104, 202105, 202106]
TARGET, PESO_BAJA1 = "pesos", 0.25
NBR = 1_000
CARPETA = c.EXPERIMENTOS / NOMBRE


def params_receta(n_filas: int) -> dict:
    return {
        "objective": "binary", "boosting_type": "gbdt", "metric": "auc",
        "boost_from_average": True, "feature_pre_filter": False,
        "max_bin": 31, "min_data_in_leaf": 0,
        "feature_fraction": 0.5, "feature_fraction_bynode": 0.2,
        "learning_rate": 0.005, "num_leaves": 83,
        "min_sum_hessian_in_leaf": 12.791817 * n_filas / 326_184,
        "bagging_fraction": 1.0, "bagging_freq": 0,
        "num_threads": 8, "force_col_wise": True, "deterministic": True,
    }


def semillas_c107(n: int) -> list[int]:
    np.random.seed(c.SEMILLAS[0])
    extra = np.random.choice(1_000_000, size=15, replace=False).tolist()
    return (c.SEMILLAS + [int(x) for x in extra])[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--cortes", type=int, nargs="+", default=[10_000, 11_000, 11_500])
    args = ap.parse_args()
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)

    d = c.cargar(DATASET)
    pred = c.columnas_predictoras(d)
    X, y, w = c.preparar(d, MESES, TARGET, PESO_BAJA1)
    X = X[pred]
    fut = d[d[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    validos = set(ids.tolist())
    Xfut = fut[pred]
    del d
    P = params_receta(len(X))
    print(f"  {len(pred)} predictoras | train {len(X):,} filas, {int(y.sum()):,} positivos "
          f"| min_sum_hessian {P['min_sum_hessian_in_leaf']:.2f} [{time.time()-t0:.0f}s]")

    scores: dict[int, np.ndarray] = {}
    for i, s in enumerate(semillas_c107(args.semillas), 1):
        ruta = CARPETA / f"scores_202108_s{s}.parquet"
        if ruta.exists():
            t = pd.read_parquet(ruta)
            assert (t[c.fe.ID].to_numpy() == ids).all(), f"{ruta.name}: los ids cacheados no son los del dataset actual"
            scores[s] = t["score"].to_numpy()
        else:
            etiqueta = f"receta_{s}_{c.clave(P, MESES, TARGET, PESO_BAJA1, DATASET, NBR, s)}"
            m = c.entrenar_o_cargar(P, X, y, w, NBR, s, CARPETA, etiqueta)
            scores[s] = m.predict(Xfut)
            pd.DataFrame({c.fe.ID: ids, "score": scores[s]}).to_parquet(ruta)
        print(f"  semilla {s} ({i}/{args.semillas}) [{time.time()-t0:.0f}s]", flush=True)

    ens = c.ensamble_por_rank(scores)
    pd.DataFrame({c.fe.ID: ids, "ensamble": ens}).to_parquet(CARPETA / "scores_202108.parquet")
    ref = c.cargar("competencia_01.parquet")
    for K in args.cortes:
        E = CARPETA / f"envios_{K}"
        E.mkdir(exist_ok=True)
        for j, s in enumerate(scores):
            sel = c.top_k(scores[s], ids, K)
            if j == 0:
                c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{NOMBRE}_{K}_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens, ids, K), E / f"{NOMBRE}_{K}_ens.csv", validos, referencia=ref)
        print(f"  corte {K:6,}: {len(scores)} archivos de semilla + 1 ensamble")
    print(f"  listo [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
