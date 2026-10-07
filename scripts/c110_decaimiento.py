"""Si los meses viejos estorban, ¿cortarlos o pesarlos menos?

c109 dejo claro que manda el mes y no los positivos: 202104 recortado a los
mismos 960 positivos que 202103 igual le saca +34,2 M (10/10, p=0,002). La
pregunta practica es si conviene tirar los meses viejos o solo bajarles el peso.

Se barre lambda: cada fila se multiplica por lambda^(antiguedad en meses). Con
lambda=1 todos los meses pesan igual (lo que hacemos hoy); con lambda chico el
mes mas viejo casi no cuenta. El peso de clase (BAJA+1 = 0,25) se mantiene y se
multiplica, no se reemplaza.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

MESES, MES_VAL, CORTE = [202103, 202104], 202106, 11_000
TARGET, PESO, NBR = "pesos", 0.25, 250
LAMBDAS = [0.05, 0.15, 0.30, 0.50, 0.75, 1.00]
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c110_decaimiento"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar()
    pred = c.columnas_predictoras(data)
    val = data[data[c.fe.MES] == MES_VAL]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")

    sub = data[data[c.fe.MES].isin(MESES)]
    X, y, w_clase = c.preparar(data, MESES, TARGET, PESO)
    X = X[pred]
    # antiguedad en meses respecto del mes mas nuevo del train
    edad = (max(MESES) // 100 * 12 + max(MESES) % 100) - (
        sub[c.fe.MES] // 100 * 12 + sub[c.fe.MES] % 100)
    edad = edad.to_numpy()
    print(f"train {MESES} -> {MES_VAL} | filas por edad: "
          f"{dict(zip(*np.unique(edad, return_counts=True)))}\n")

    res = {}
    for lam in LAMBDAS:
        w = w_clase * (lam ** edad)
        acums = []
        for s in semillas:
            cl = c.clave(PARAMS, MESES, TARGET, PESO, f"lam{lam}", NBR, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"lam{lam}_{s}_{cl}")
            acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
        obs = np.array([a[CORTE - 1] for a in acums])
        res[lam] = obs
        print(f"  lambda={lam:.2f}  peso del mes viejo={lam:.2f}  @11k={obs.mean()/1e6:>7,.1f}M "
              f"(sd {obs.std()/1e6:.1f})  [{time.time()-t0:.0f}s]")

    print("\n=== pareado por semilla contra lambda=1 (todos los meses igual) ===")
    ref = res[1.00]
    mejor, mejor_d = None, -np.inf
    for lam, obs in res.items():
        d = obs - ref
        if np.allclose(d, 0):
            continue
        p = wilcoxon(d).pvalue
        print(f"  lambda={lam:.2f}: {d.mean()/1e6:+6.1f} M  gana en {(d>0).sum()}/{len(d)}  p={p:.4f}")
        if d.mean() > mejor_d:
            mejor, mejor_d = lam, d.mean()
    print(f"\nmejor lambda: {mejor} ({mejor_d/1e6:+,.1f} M sobre pesar todo igual)")
    print("referencia c108: entrenar SOLO en 202104 daba +10,6 M (18/20)")
    pd.DataFrame(res).to_parquet(CARPETA / "por_semilla.parquet", index=False)
    print(f"listo en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
