"""Revalida el top-N de la busqueda limpia, FUERA de muestra y a VARIOS cortes.

Dos lecciones de hoy, incorporadas:

1. El ganador de la busqueda no es el ganador del futuro (z301, z601). La
   busqueda entrega candidatos; el ganador se elige revalidando.
2. El ranking de configuraciones CAMBIA segun el corte. `pesos25` le ganaba a
   `baja2` por +8,6 M al corte 11.000 y por -0,1 M al corte 9.000. Evaluar a un
   solo corte fue el error que obligo a retractar ese resultado.

Se compara contra el baseline heredado de z701, que es el que hay que vencer.
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import optuna

import competencia as c
import registro as r

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
CORTES = [9_000, 10_000, 11_000, 12_000]
N_SEMILLAS = 5
FIJOS = {"objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
         "boost_from_average": True, "feature_pre_filter": False, "verbose": -1,
         "bagging_freq": 1}
BUSCADOS = ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction",
            "bagging_fraction", "lambda_l1", "lambda_l2", "min_gain_to_split", "max_bin")
BASELINE = {**FIJOS, "max_bin": 31, "num_leaves": 45, "learning_rate": 0.0077,
            "min_data_in_leaf": 174, "feature_fraction": 0.277, "bagging_fraction": 0.918}
CARPETA = c.EXPERIMENTOS / "c123_revalidar"
CARPETA.mkdir(parents=True, exist_ok=True)


def evaluar(data, pred, params, esquema, peso, nbr, etq, semillas):
    """Devuelve {corte: array de ganancias por (fold, semilla)}."""
    por_corte = {k: [] for k in CORTES}
    for meses, mv in FOLDS:
        X, y, w = c.preparar(data, meses, esquema, peso)
        X = X[pred]
        val = data[data[c.fe.MES] == mv]
        es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        for s in semillas:
            cl = c.clave(params, meses, esquema, peso, etq, nbr, s)
            m = c.entrenar_o_cargar(params, X, y, w, nbr, s, CARPETA, f"{etq}_{mv}_{s}_{cl}")
            acum = c.ganancia_acumulada(m.predict(val[pred]), es)
            for k in CORTES:
                por_corte[k].append(acum[k - 1])
    return {k: np.array(v) for k, v in por_corte.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()

    est = optuna.load_study(study_name="c122_base", storage=r.dsn_url())
    hechos = [t for t in est.trials if t.value is not None]
    top = sorted(hechos, key=lambda t: t.value, reverse=True)[: args.top]
    print(f"{len(hechos)} trials; revalidando el top {len(top)} a {len(CORTES)} cortes\n")

    data = c.cargar("competencia_01.parquet")
    pred = c.columnas_predictoras(data)
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS[:N_SEMILLAS]

    cand = [("baseline_z701", BASELINE, "pesos", 0.25, 250)]
    for t in top:
        cand.append((f"t{t.number}", {**FIJOS, **{k: t.params[k] for k in BUSCADOS if k in t.params}},
                     t.params.get("esquema", "baja2"), t.params.get("peso_baja1", 0.0),
                     t.user_attrs.get("best_iter", 250)))

    res = {}
    for etq, params, esq, peso, nbr in cand:
        res[etq] = evaluar(data, pred, params, esq, peso, nbr, etq, semillas)
        linea = "  ".join(f"{k//1000}k:{res[etq][k].mean()/1e6:>6,.1f}" for k in CORTES)
        print(f"  {etq:14} esq={esq:6} peso={peso:.2f} nbr={nbr:>4}  {linea}  [{time.time()-t0:.0f}s]")

    print(f"\n=== pareado contra baseline_z701, por corte ===")
    print(f"{'candidato':14} " + "  ".join(f"{k//1000}k".rjust(16) for k in CORTES))
    filas = []
    for etq in res:
        if etq == "baseline_z701":
            continue
        celdas = []
        for k in CORTES:
            d = res[etq][k] - res["baseline_z701"][k]
            p = wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
            celdas.append(f"{d.mean()/1e6:+6.1f} {(d>0).sum():>2}/{len(d)} {p:.2f}")
            filas.append({"cand": etq, "corte": k, "delta": d.mean(), "gana": (d > 0).sum(), "p": p})
        print(f"  {etq:14} " + "  ".join(c_.rjust(16) for c_ in celdas))
    pd.DataFrame(filas).to_parquet(CARPETA / "revalidacion.parquet", index=False)

    df = pd.DataFrame(filas)
    gana_en_todos = df.groupby("cand").delta.min()
    print(f"\ncandidato con mejor PEOR corte: {gana_en_todos.idxmax()} "
          f"({gana_en_todos.max()/1e6:+.1f} M en su peor corte)")
    if gana_en_todos.max() <= 0:
        print("-> NINGUN candidato de la busqueda le gana al baseline en todos los cortes.")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
