"""La busqueda que nunca se corrio: Optuna contra el objetivo CORRECTO.

Las dos busquedas anteriores (c105, c122) optimizaban ganancia en CV DENTRO del mes.
Eso premia fusionar las clases -mas positivos, y la fusion no paga el cambio de escala-
asi que el TPE se fue a peso_baja1 ~0,8, que es justo lo que los folds rechazan.
Optimizaba bien hacia el blanco equivocado.

Aca el objetivo es la ganancia OUT OF TIME en los dos folds limpios, a corte fijo,
promediada sobre varias semillas. Es caro por trial pero es la cantidad que importa.

  objetivo = media sobre {fold A, fold B} x {semillas} de la ganancia a K=10.500

peso_baja1 entra en el espacio: que lo decida la busqueda con el objetivo bueno.
"""

import argparse
import time

import lightgbm as lgb
import numpy as np
import optuna

import competencia as c
import registro

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
CORTE = 10_500
SEMILLAS_TRIAL = 3          # varias por trial: evita elegir el trial afortunado
ESTUDIO = "c180_oot"


def objetivo(trial, data, pred):
    p = {
        "objective": "binary", "boosting_type": "gbdt", "verbose": -1,
        "first_metric_only": True, "boost_from_average": True, "feature_pre_filter": False,
        "max_bin": trial.suggest_int("max_bin", 15, 127),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.3, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
    }
    nbr = trial.suggest_int("num_boost_round", 100, 1500, log=True)
    peso = trial.suggest_float("peso_baja1", 0.0, 1.0)

    ganancias = []
    for meses, mes_val in FOLDS:
        X, y, w = c.preparar(data, meses, "pesos", peso)
        X = X[pred]
        v = data[data[c.fe.MES] == mes_val]
        es = (v["clase_ternaria"].to_numpy() == "BAJA+2")       # de SU propio v
        for s in c.SEMILLAS[:SEMILLAS_TRIAL]:
            m = lgb.train({**p, "seed": s}, lgb.Dataset(X, label=y, weight=w),
                          num_boost_round=nbr)
            g = c.ganancia_acumulada(m.predict(v[pred]), es)[CORTE - 1] / 1e6
            ganancias.append(g)
    # media de las dos medias por fold: que un fold no domine por tener mas plata
    por_fold = [np.mean(ganancias[i::len(FOLDS)]) for i in range(len(FOLDS))]
    trial.set_user_attr("por_fold", [float(x) for x in por_fold])
    return float(np.mean(por_fold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    data = c.cargar("competencia_01.parquet")
    pred = c.columnas_predictoras(data)
    print(f"{len(pred)} predictoras | objetivo: ganancia OOT a {CORTE:,} en los 2 folds limpios, "
          f"{SEMILLAS_TRIAL} semillas por trial\n")

    est = optuna.create_study(study_name=ESTUDIO, storage=registro.dsn_url(),
                              direction="maximize", load_if_exists=True)
    base = {"max_bin": 31, "num_leaves": 45, "learning_rate": 0.0077,
            "min_data_in_leaf": 174, "feature_fraction": 0.277, "bagging_fraction": 0.918,
            "lambda_l1": 1e-4, "lambda_l2": 1e-4, "min_gain_to_split": 0.0,
            "num_boost_round": 250, "peso_baja1": 0.25}
    if not est.trials:
        est.enqueue_trial(base)          # el modelo actual, como piso de referencia
    est.optimize(lambda t: objetivo(t, data, pred), n_trials=args.trials,
                 show_progress_bar=False)

    print(f"\nmejor: {est.best_value:,.1f} M   (trial {est.best_trial.number})")
    for k, v in est.best_trial.params.items():
        print(f"    {k:20s} {v}")
    print(f"    por fold: {est.best_trial.user_attrs.get('por_fold')}")
    print(f"\n{len(est.trials)} trials en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
