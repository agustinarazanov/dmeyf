"""Fase 2b: revalidar el top-N de Optuna FUERA del mes en que se busco.

z301 y z601 ya mostraron que el ganador de la busqueda no es el ganador del
futuro: el rank 1 de Optuna termino 4to en mayo. La busqueda entrega
CANDIDATOS, no un ganador; el ganador se elige revalidando.

Hace falta especialmente aca: la busqueda optimiza CV dentro de 202103-202104 y
encima elige el corte post-hoc sobre el mismo mes. Las dos cosas favorecen al
target fusionado (mas positivos, y la fusion no paga el cambio de escala). Los
folds out of time del c102 preferian peso 0,25, no 0,9.
"""

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c
import registro as r

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
DATASETS = {"base": "competencia_01.parquet", "fe": "competencia_01_fe.parquet"}
FIJOS = {"objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
         "boost_from_average": True, "feature_pre_filter": False, "verbose": -1,
         "bagging_freq": 1}
BUSCADOS = ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction",
            "bagging_fraction", "lambda_l1", "lambda_l2", "min_gain_to_split", "max_bin")


def params_de(trial) -> dict:
    return {**FIJOS, **{k: trial.params[k] for k in BUSCADOS if k in trial.params}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="base")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--semillas", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    estudio = optuna.load_study(study_name=f"c105_{args.dataset}", storage=r.dsn_url())
    hechos = [t for t in estudio.trials if t.value is not None]
    top = sorted(hechos, key=lambda t: t.value, reverse=True)[: args.top]
    print(f"{len(hechos)} trials; revalidando el top {len(top)} fuera de muestra\n")

    data = c.cargar(DATASETS[args.dataset])
    pred = c.columnas_predictoras(data)
    carpeta = c.EXPERIMENTOS / f"c106_{args.dataset}"
    carpeta.mkdir(parents=True, exist_ok=True)
    semillas = c.SEMILLAS[: args.semillas]

    resultados = []
    for rank, trial in enumerate(top, 1):
        params = params_de(trial)
        esquema = trial.params.get("esquema", "baja2")
        peso = trial.params.get("peso_baja1", 0.0)
        nbr = trial.user_attrs.get("best_iter", 250)
        nombre = f"{args.dataset}_t{trial.number}"

        escenarios, obs = {}, []
        for meses, mes_val in FOLDS:
            X, y, w = c.preparar(data, meses, esquema, peso)
            X = X[pred]
            val = data[data[c.fe.MES] == mes_val]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            acums = []
            for s in semillas:
                cl = c.clave(params, meses, esquema, peso, args.dataset, nbr, s)
                m = c.entrenar_o_cargar(params, X, y, w, nbr, s, carpeta,
                                        f"{nombre}_{mes_val}_{s}_{cl}")
                acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
            escenarios[f"{mes_val}"] = np.mean(acums, axis=0)
            obs.append((mes_val, acums))

        corte, tabla = c.corte_minimax(escenarios)
        plano = np.array([a[corte - 1] for _, acs in obs for a in acs])
        resultados.append({
            "rank_cv": rank, "trial": trial.number, "ganancia_cv": trial.value,
            "esquema": esquema, "peso": round(peso, 3), "num_boost_round": nbr,
            "corte": corte, "regret_peor": float(tabla.loc[corte, "peor_caso"]),
            "oot_media": plano.mean(), "oot_sd": plano.std(),
            "oot_202105": escenarios["202105"][corte - 1],
            "oot_202106": escenarios["202106"][corte - 1],
            "obs": plano, "params": params,
        })
        print(f"  rank_cv {rank} (trial {trial.number}): esq={esquema} peso={peso:.2f} "
              f"nbr={nbr} corte={corte:,} -> oot {plano.mean()/1e6:,.1f}M "
              f"(sd {plano.std()/1e6:,.1f}) [{time.time()-t0:.0f}s]")

    df = pd.DataFrame([{k: v for k, v in r_.items() if k not in ("obs", "params")}
                       for r_ in resultados])
    df["rank_oot"] = df.oot_media.rank(ascending=False).astype(int)
    df.to_parquet(carpeta / "revalidacion.parquet", index=False)

    print("\n=== el ranking de la busqueda vs el del futuro ===")
    m = df.copy()
    for col in ("ganancia_cv", "oot_media", "oot_sd", "oot_202105", "oot_202106"):
        m[col] = (m[col] / 1e6).round(1)
    m["regret_peor"] = (m.regret_peor * 100).round(2)
    print(m[["rank_cv", "rank_oot", "trial", "ganancia_cv", "esquema", "peso", "corte",
             "oot_202105", "oot_202106", "oot_media", "oot_sd", "regret_peor"]].to_string(index=False))

    rho = df.rank_cv.corr(df.rank_oot, method="spearman")
    print(f"\ncorrelacion de Spearman entre el ranking CV y el out of time: {rho:+.2f}")
    if df.loc[df.rank_oot == 1, "rank_cv"].iloc[0] != 1:
        print("  -> el ganador de la busqueda NO es el ganador del futuro (otra vez)")

    mejor = resultados[int(df.oot_media.idxmax())]
    base = resultados[0]
    d = mejor["obs"] - base["obs"]
    if not np.allclose(d, 0):
        print(f"\nmejor out of time = trial {mejor['trial']}: +{d.mean()/1e6:,.1f} M sobre el "
              f"rank 1 de CV, gana en {(d>0).sum()}/{len(d)}, p={wilcoxon(d).pvalue:.4f}")
    (carpeta / "ganador.json").write_text(json.dumps(
        {k: v for k, v in mejor.items() if k != "obs"}, indent=2, default=str))
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
