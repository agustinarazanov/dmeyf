"""Control de sobreajuste de la busqueda: la receta y el top-5 de c220 en un mes que Optuna no vio.

c220 eligio hiperparametros maximizando AUC en junio (fold B) y julio (fold C). Con ~1.100 positivos
por mes el error muestral del AUC (~0,004) es del orden de la mejora encontrada (0,003), asi que el
mejor trial puede estar ajustado a esos dos meses. El fold A ([202103] -> 202105, BAJA+2) no participo
de la busqueda: si el orden receta < top-5 se mantiene en mayo, la mejora generaliza; si se mezcla, no.

Una semilla, rondas fijas (1.000, las de la receta) para que la comparacion sea solo de hiperparametros.
Reporta AUC y ganancia a 8.000 / 10.000 / 14.000 en mayo (870 positivos: el K* de mayo es ~8.000, asi
que la ganancia a 14.000 es mala para todos; lo que importa es el ORDEN entre configuraciones).

    python scripts/c222_fold_A_control.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))
import competencia as c  # noqa: E402
from c201_receta_lags import DATASET, params_receta, semillas_c107  # noqa: E402
from c221_ensamble_heterogeneo import params_de, STUDY  # noqa: E402

NOMBRE = "c222_fold_A"
CARPETA = c.EXPERIMENTOS / NOMBRE
NBR = 1_000


def main() -> None:
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)
    study = optuna.load_study(study_name="c220_optuna", storage=f"sqlite:///{STUDY}")
    top = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
    d = c.cargar(DATASET)
    pred = c.columnas_predictoras(d)
    X, y, w = c.preparar(d, [202103], "pesos", 0.25)
    X = X[pred]
    val = d[d[c.fe.MES] == 202105]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
    Xv = val[pred]
    del d
    s = semillas_c107(1)[0]
    configs = [("receta", {**params_receta(len(X)), "num_threads": 4})]
    configs += [(f"trial {t.number} (auc busqueda {t.value:.5f})", {**params_de(t, len(X)), "num_threads": 4}) for t in top]
    print(f"  fold A: train 202103 ({len(X):,} filas) -> 202105 ({int(es.sum())} BAJA+2) [{time.time()-t0:.0f}s]", flush=True)
    print(f"  {'config':38s} {'AUC mayo':>9s} {'gan@8k':>8s} {'gan@10k':>8s} {'gan@14k':>8s}", flush=True)
    for nombre, P in configs:
        etq = f"{nombre.split(' (')[0].replace(' ', '')}_{c.clave(P, [202103], 'pesos', 0.25, DATASET, NBR, s)}"
        m = c.entrenar_o_cargar(P, X, y, w, NBR, s, CARPETA, etq)
        sc = m.predict(Xv)
        g = c.ganancia_acumulada(sc, es)
        print(f"  {nombre:38s} {roc_auc_score(es, sc):9.5f} {g[7999]/1e6:8.1f} {g[9999]/1e6:8.1f} {g[13999]/1e6:8.1f}   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
