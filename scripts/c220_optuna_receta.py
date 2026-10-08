"""Optuna sobre el regimen de la receta de Denicolay, en lags12, con el objetivo correcto.

El desacople (7-oct 23:18) mostro que la receta vale +3,8 por si sola; esta busqueda explora su
vecindad, que es lo que Denicolay deja como "bayesianos": num_leaves, min_sum_hessian_in_leaf y
num_iterations (aca por early stopping), mas feature_fraction / bynode como hizo Abregu (0,40).

Objetivo: AUC media en los dos folds limpios que tienen historia en el entrenamiento, B ([03,04] -> 06,
BAJA+2) y C ([03,04,05] -> 07, BAJA+1). Sin undersampling (Denicolay, En Limpio 1.7). Una semilla por
trial, la 1 de c107; el ensamble heterogeneo del top-k de trials se entrena aparte (c221) en 03-06.
El AUC no depende de la prevalencia ni del corte, asi que no premia fusionar clases (el error de c105).

Storage sqlite en experimentos/c220_optuna/study.db: resumible, `--trials N` son N trials mas.

    python scripts/c220_optuna_receta.py --trials 40
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))
import competencia as c  # noqa: E402
from c201_receta_lags import DATASET, semillas_c107  # noqa: E402

NOMBRE = "c220_optuna"
CARPETA = c.EXPERIMENTOS / NOMBRE
FOLDS = [([202103, 202104], 202106, "BAJA+2"), ([202103, 202104, 202105], 202107, "BAJA+1")]
MAX_ROUNDS, PACIENCIA = 2_500, 150


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)
    d = c.cargar(DATASET)
    pred = c.columnas_predictoras(d)
    folds = []
    for meses, mes_val, positivo in FOLDS:
        X, y, w = c.preparar(d, meses, "pesos", 0.25)
        val = d[d[c.fe.MES] == mes_val]
        folds.append((lgb.Dataset(X[pred], label=y, weight=w, free_raw_data=False),
                      lgb.Dataset(val[pred], label=(val[c.fe.CLASE].to_numpy() == positivo).astype("int8"),
                                  free_raw_data=False), len(X)))
    del d
    print(f"  {len(pred)} predictoras, folds listos [{time.time()-t0:.0f}s]", flush=True)
    semilla = semillas_c107(1)[0]

    def objetivo(trial: optuna.Trial) -> float:
        base = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
            "min_data_in_leaf": 0, "learning_rate": 0.005, "bagging_freq": 0,
            "num_threads": 8, "force_col_wise": True, "deterministic": True, "verbose": -1,
            "seed": semilla,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 0.7),
            "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.1, 0.5),
            "hessian_por_fila": trial.suggest_float("hessian_por_fila", 1e-6, 3e-4, log=True),
        }
        hpf = base.pop("hessian_por_fila")      # la receta escala el hessiano por filas: 12,79/326.184 = 3,9e-5
        aucs, iters = [], []
        for dtr, dval, n in folds:
            p = {**base, "min_sum_hessian_in_leaf": hpf * n}
            m = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
            aucs.append(m.best_score["valid_0"]["auc"])
            iters.append(m.best_iteration)
        trial.set_user_attr("auc_por_fold", aucs)
        trial.set_user_attr("best_iter_por_fold", iters)
        trial.set_user_attr("hessian_por_fila", hpf)
        print(f"  trial {trial.number}: auc {np.mean(aucs):.5f} {aucs} iters {iters} "
              f"leaves {base['num_leaves']} ff {base['feature_fraction']:.2f} bynode {base['feature_fraction_bynode']:.2f} "
              f"hpf {hpf:.2e} [{time.time()-t0:.0f}s]", flush=True)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", study_name=NOMBRE,
                                storage=f"sqlite:///{CARPETA / 'study.db'}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=8))
    if len(study.trials) == 0:
        study.enqueue_trial({"num_leaves": 83, "feature_fraction": 0.5, "feature_fraction_bynode": 0.2,
                             "hessian_por_fila": 12.791817 / 326_184})       # la receta tal cual, como trial 0
    study.optimize(objetivo, n_trials=args.trials)
    print("\n  top 5:")
    for t in sorted(study.trials, key=lambda t: -(t.value or 0))[:5]:
        print(f"    #{t.number} auc {t.value:.5f} {t.params} iters {t.user_attrs.get('best_iter_por_fold')}")
    print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
