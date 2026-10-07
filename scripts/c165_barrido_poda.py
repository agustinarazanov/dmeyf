"""Barrido de agresividad de la poda por canaritos, y evaluacion en los dos folds.

c161 con desvios=4 deja pasar el 83% de las variables: demasiado flojo. Pero el mejor
canarito queda en la posicion ~100 de 267, o sea solo ~100 variables reales le ganan a
TODO el ruido. Acá se calcula la importancia UNA vez por semilla y se derivan los
sobrevivientes para varios umbrales, incluido el estricto ("mejor que el mejor canarito").

Despues se evalua cada set podado en los dos folds, contra el FE completo y contra el base.
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

PARAMS = {"objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
          "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
          "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
          "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
          "verbose": -1}
MESES = [202103, 202104, 202105, 202106]
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
CORTE = 9_000
CAR = c.EXPERIMENTOS / "c165_poda"


def ranking_con_canaritos(data, pred, semilla, ratio=0.2):
    X, y, w = c.preparar(data, MESES, "pesos", 0.25)
    X = X[pred]
    rng = np.random.default_rng(semilla)
    n_can = max(1, int(len(pred) * ratio))
    cols = [f"canarito{i:04d}" for i in range(n_can)]
    X = pd.concat([X, pd.DataFrame(rng.uniform(size=(len(X), n_can)),
                                   columns=cols, index=X.index)], axis=1)
    m = lgb.train({**PARAMS, "seed": semilla}, lgb.Dataset(X, label=y, weight=w),
                  num_boost_round=250)
    imp = pd.DataFrame({"feature": m.feature_name(), "gain": m.feature_importance("gain")})
    imp = imp.sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)
    imp["pos"] = np.arange(1, len(imp) + 1)
    imp["es_canarito"] = imp.feature.isin(cols)
    return imp


def main():
    t0 = time.time()
    CAR.mkdir(parents=True, exist_ok=True)
    data = c.cargar("competencia_01_fe.parquet")
    pred = c.columnas_predictoras(data)
    base_cols = set(c.columnas_predictoras(c.cargar("competencia_01.parquet")))
    sem_poda = c.SEMILLAS[:5]
    print(f"competencia_01_fe.parquet: {len(pred)} predictoras\n")

    rks = [ranking_con_canaritos(data, pred, s) for s in sem_poda]
    print(f"  importancias calculadas [{time.time()-t0:.0f}s]\n")

    criterios = {"estricto (mejor que TODO el ruido)": None, "mediana (desvios=0)": 0.0,
                 "desvios=1": 1.0, "desvios=2": 2.0, "desvios=4": 4.0}
    sets = {}
    print(f"  {'criterio':36s} {'vivos':>6} {'de base':>8} {'del FE':>7} {'gain=0':>7}")
    for nom, desv in criterios.items():
        votos = {}
        for imp in rks:
            pc = imp.loc[imp.es_canarito, "pos"]
            umbral = pc.min() if desv is None else pc.median() + desv * pc.std()
            for f in imp[(~imp.es_canarito) & (imp.pos < umbral)].feature:
                votos[f] = votos.get(f, 0) + 1
        vivos = sorted([f for f, v in votos.items() if v >= 3])
        sets[nom] = vivos
        g0 = sum(1 for f in vivos if all(
            imp.loc[imp.feature == f, "gain"].iloc[0] == 0 for imp in rks))
        nb = sum(1 for f in vivos if f in base_cols)
        print(f"  {nom:36s} {len(vivos):6d} {nb:8d} {len(vivos)-nb:7d} {g0:7d}")

    sets["FE completo (sin podar)"] = pred
    sets["base (150)"] = sorted(base_cols)

    print(f"\n=== evaluacion en los dos folds, corte {CORTE:,}, 10 semillas ===\n")
    sem = c.SEMILLAS + [112909, 314159, 562991, 733517, 951413]
    dbase = c.cargar("competencia_01.parquet")
    res = {}
    for nom, cols in sets.items():
        d = dbase if nom == "base (150)" else data
        cols = [x for x in cols if x in d.columns]
        for tr, va in FOLDS:
            X, y, w = c.preparar(d, tr, "pesos", 0.25)
            v = d[d[c.fe.MES] == va]
            es = (v["clase_ternaria"].to_numpy() == "BAJA+2")      # de SU propio v
            g = []
            for s in sem:
                cl = c.clave(PARAMS, tr, "pesos", 0.25, f"{nom}{len(cols)}", 250, s)
                m = c.entrenar_o_cargar(PARAMS, X[cols], y, w, 250, s, CAR,
                                        f"{abs(hash(nom))%10**8}_{va}_{s}_{cl}")
                g.append(c.ganancia_acumulada(m.predict(v[cols]), es)[CORTE - 1] / 1e6)
            res[(nom, va)] = np.array(g)
        print(f"  {nom:36s} {len(cols):4d} col | 202105 {res[(nom,202105)].mean():6.1f}M"
              f" | 202106 {res[(nom,202106)].mean():6.1f}M"
              f" | media {np.mean([res[(nom,202105)].mean(), res[(nom,202106)].mean()]):6.1f}M"
              f"   [{time.time()-t0:.0f}s]")

    print(f"\n=== contra el base, pareado por semilla ===\n")
    for nom in sets:
        if nom == "base (150)":
            continue
        linea = f"  {nom:36s}"
        for va in (202105, 202106):
            a, b = res[(nom, va)], res[("base (150)", va)]
            p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            linea += f" | {va}: {a.mean()-b.mean():+6.1f}M {int((a>b).sum()):2d}/10 p={p:.3f}"
        print(linea)
    print(f"\nlisto en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
