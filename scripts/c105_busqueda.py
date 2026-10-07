"""Fase 2: busqueda de hiperparametros con Optuna, resumible y paralelizable.

El study vive en Postgres, no en sqlite: asi se puede correr este script en
varios procesos a la vez contra el mismo study y retomarlo si se corta.

    python c105_busqueda.py --dataset base --trials 40
    python c105_busqueda.py --dataset base --trials 40   # otra terminal, suma

El objetivo es la ganancia en CV sobre los meses de train. NO es la decision
final: z301 y z601 ya mostraron que el ganador de la busqueda no es el ganador
del futuro. El top-N se revalida out of time en c106.
"""

import argparse
import logging

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import StratifiedShuffleSplit

import competencia as c
import registro as r

MESES_TRAIN = [202103, 202104]
DATASETS = {"base": "competencia_01.parquet", "fe": "competencia_01_fe.parquet"}
HISTORIA = ("__lag1", "__delta1", "__slope")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("busqueda")


def hacer_objetivo(data, pred, semilla):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=semilla)

    def objetivo(trial):
        esquema = trial.suggest_categorical("esquema", ["baja2", "pesos"])
        peso = trial.suggest_float("peso_baja1", 0.05, 1.0) if esquema == "pesos" else 0.0
        lr = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        params = {
            "objective": "binary", "metric": "custom", "boosting_type": "gbdt",
            "first_metric_only": True, "boost_from_average": True,
            "feature_pre_filter": False, "verbose": -1, "seed": semilla,
            "learning_rate": lr,
            "num_leaves": trial.suggest_int("num_leaves", 8, 200),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 2000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 10.0),
            "max_bin": trial.suggest_int("max_bin", 15, 255, log=True),
        }
        X, y, w = c.preparar(data, MESES_TRAIN, esquema, peso)
        X = X[pred]
        # Se entrena con el target que diga el trial, pero la ganancia se mide
        # SIEMPRE contra BAJA+2: es lo unico que paga.
        y_real = c.preparar(data, MESES_TRAIN, "baja2")[1].astype(bool)

        ganancias, iteraciones, cortes = [], [], []
        for tr, te in sss.split(X, y_real):
            verdad = y_real[te]

            def feval(y_pred, _dset, verdad=verdad):
                # early stopping sobre la ganancia, no sobre la log-loss
                return "ganancia", float(c.ganancia_acumulada(y_pred, verdad).max() / 0.3), True

            dtr = lgb.Dataset(X.iloc[tr], label=y[tr], weight=w[tr])
            dte = lgb.Dataset(X.iloc[te], label=y[te], reference=dtr)
            booster = lgb.train(
                params, dtr, num_boost_round=1500, valid_sets=[dte], feval=feval,
                callbacks=[lgb.early_stopping(int(50 + 5 / lr), verbose=False,
                                              first_metric_only=True)])
            acum = c.ganancia_acumulada(
                booster.predict(X.iloc[te], num_iteration=booster.best_iteration), verdad)
            ganancias.append(acum.max() / 0.3)
            iteraciones.append(booster.best_iteration)
            cortes.append(int(np.argmax(acum)) + 1)

        trial.set_user_attr("best_iter", int(np.median(iteraciones)))
        trial.set_user_attr("corte_cv", int(np.median(cortes) / 0.3))
        trial.set_user_attr("esquema", esquema)
        trial.set_user_attr("peso_baja1", peso)
        return float(np.mean(ganancias))

    return objetivo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="base")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--solo-ranks", action="store_true")
    args = ap.parse_args()

    data = c.cargar(DATASETS[args.dataset])
    pred = c.columnas_predictoras(data)
    if args.solo_ranks:
        pred = [x for x in pred if not x.endswith(HISTORIA)]
    nombre = args.dataset + ("_ranks" if args.solo_ranks else "")
    log.info(f"{nombre}: {len(pred)} predictoras, train {MESES_TRAIN}")

    estudio = optuna.create_study(
        study_name=f"c105_{nombre}", direction="maximize",
        storage=r.dsn_url(), load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=c.SEMILLAS[0]))
    hechos = len(estudio.trials)
    log.info(f"study c105_{nombre}: {hechos} trials ya hechos")

    estudio.optimize(hacer_objetivo(data, pred, c.SEMILLAS[0]),
                     n_trials=args.trials, show_progress_bar=False)

    log.info(f"mejor ganancia CV: {estudio.best_value:,.0f}")
    log.info(f"mejores params: {estudio.best_params}")
    log.info(f"best_iter: {estudio.best_trial.user_attrs.get('best_iter')}")


if __name__ == "__main__":
    main()
