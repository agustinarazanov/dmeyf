"""Ensamble heterogeneo: el top-k de trials de c220, cada uno con varias semillas, en 03-06.

Lo que Denicolay pide para la entrega ("10+ modelos con hiperparametros distintos, promediando")
y lo que a Abregu le rindio +6,8 (semillerio sobre params de Optuna). Cada (trial, semilla) es un
modelo cacheado; los scores de 202108 se guardan por modelo; el ensamble es por rank sobre todos.
Tambien se escribe el ensamble de cada trial solo, para medir si la heterogeneidad aporta sobre
el mejor trial.

    python scripts/c221_ensamble_heterogeneo.py --top 5 --semillas 4 --rondas 1500 --cortes 14000

--rondas: num_iterations para el modelo final en 4 meses. Los best_iter de la busqueda fueron ~1.000
en el fold B (2 meses) y ~2.500 (tope) en el fold C (3 meses); con mas filas conviene mas rondas.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent / "src"))
import competencia as c  # noqa: E402
from c201_receta_lags import DATASET, semillas_c107  # noqa: E402

NOMBRE = "c221_ensamble"
CARPETA = c.EXPERIMENTOS / NOMBRE
STUDY = c.EXPERIMENTOS / "c220_optuna" / "study.db"
MESES = [202103, 202104, 202105, 202106]


def params_de(trial, n_filas: int) -> dict:
    p = trial.params
    return {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
        "min_data_in_leaf": 0, "learning_rate": 0.005, "bagging_freq": 0,
        "num_threads": 8, "force_col_wise": True, "deterministic": True,
        "num_leaves": p["num_leaves"], "feature_fraction": p["feature_fraction"],
        "feature_fraction_bynode": p["feature_fraction_bynode"],
        "min_sum_hessian_in_leaf": p["hessian_por_fila"] * n_filas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--semillas", type=int, default=4)
    ap.add_argument("--rondas", type=int, default=1_500)
    ap.add_argument("--cortes", type=int, nargs="+", default=[14_000])
    args = ap.parse_args()
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)

    study = optuna.load_study(study_name="c220_optuna", storage=f"sqlite:///{STUDY}")
    top = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:args.top]
    print("  trials elegidos:")
    for t in top:
        print(f"    #{t.number} auc {t.value:.5f} {t.params}")

    d = c.cargar(DATASET)
    pred = c.columnas_predictoras(d)
    X, y, w = c.preparar(d, MESES, "pesos", 0.25)
    X = X[pred]
    fut = d[d[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    Xfut = fut[pred]
    del d

    scores = {}
    for t in top:
        P = params_de(t, len(X))
        for s in semillas_c107(args.semillas):
            ruta = CARPETA / f"scores_202108_t{t.number}_s{s}.parquet"
            if ruta.exists():
                tt = pd.read_parquet(ruta)
                assert (tt[c.fe.ID].to_numpy() == ids).all()
                scores[(t.number, s)] = tt["score"].to_numpy()
            else:
                etq = f"t{t.number}_s{s}_{c.clave(P, MESES, 'pesos', 0.25, DATASET, args.rondas, s)}"
                m = c.entrenar_o_cargar(P, X, y, w, args.rondas, s, CARPETA, etq)
                scores[(t.number, s)] = m.predict(Xfut)
                pd.DataFrame({c.fe.ID: ids, "score": scores[(t.number, s)]}).to_parquet(ruta)
            print(f"  trial {t.number} semilla {s} [{time.time()-t0:.0f}s]", flush=True)

    ref = c.cargar("competencia_01.parquet")
    validos = set(ids.tolist())
    ens_todo = c.ensamble_por_rank(scores)
    pd.DataFrame({c.fe.ID: ids, "ensamble": ens_todo}).to_parquet(CARPETA / "scores_202108_ens_hetero.parquet")
    for K in args.cortes:
        E = CARPETA / f"envios_{K}"
        E.mkdir(exist_ok=True)
        # un archivo por trial (ensamble de sus semillas): el bot promedia -> mide "trial medio"
        for t in top:
            e_t = c.ensamble_por_rank({k: v for k, v in scores.items() if k[0] == t.number})
            sel = c.top_k(e_t, ids, K)
            c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{NOMBRE}_{K}_t{t.number}.csv", validos)
        # un archivo por semilla del ensamble heterogeneo (todos los trials, esa semilla)
        for s in semillas_c107(args.semillas):
            e_s = c.ensamble_por_rank({k: v for k, v in scores.items() if k[1] == s})
            c.escribir_envios(c.top_k(e_s, ids, K), E / f"{NOMBRE}_{K}_hetero_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens_todo, ids, K), E / f"{NOMBRE}_{K}_hetero_todo.csv", validos, referencia=ref)
        print(f"  corte {K:6,}: {len(top)} archivos por trial, {args.semillas} del heterogeneo por semilla, 1 total")
    print(f"  listo [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
