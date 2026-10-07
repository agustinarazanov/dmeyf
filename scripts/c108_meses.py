"""Sumar meses viejos, ¿ayuda o estorba?

En c104 entrenar solo en 202104 dio 402,6 M y entrenar en 202103+202104 dio
396,4 M: el mes mas viejo parecia restar. Si es real, el modelo final —que
entrena con los cuatro meses— esta cargando lastre, y z701 ya avisaba que para
agosto convienen los meses mas cercanos a la inferencia.

Validacion fija en 202106 para que lo unico que cambie sean los meses de train.
La regla max(train) <= validacion-2 deja solo 202103 y 202104 como candidatos,
asi que esto es la pregunta en miniatura, no la respuesta completa.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

MES_VAL = 202106
CORTE = 11_000
TARGET, PESO = "pesos", 0.25
NUM_BOOST_ROUND = 250
CONJUNTOS = {"solo 202103": [202103], "solo 202104": [202104],
             "202103+202104": [202103, 202104]}
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
np.random.seed(c.SEMILLAS[0])
SEMILLAS = c.SEMILLAS + np.random.choice(1_000_000, size=15, replace=False).tolist()

CARPETA = c.EXPERIMENTOS / "c108_meses"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    data = c.cargar()
    pred = c.columnas_predictoras(data)
    val = data[data[c.fe.MES] == MES_VAL]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
    print(f"validacion fija: {MES_VAL}, {es.sum()} BAJA+2, corte {CORTE:,}\n")

    res = {}
    for etq, meses in CONJUNTOS.items():
        X, y, w = c.preparar(data, meses, TARGET, PESO)
        X = X[pred]
        acums = []
        for s in SEMILLAS:
            cl = c.clave(PARAMS, meses, TARGET, PESO, "base", NUM_BOOST_ROUND, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NUM_BOOST_ROUND, s, CARPETA,
                                    f"{etq.replace(' ','')}_{s}_{cl}")
            acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
        a = np.mean(acums, axis=0)
        res[etq] = {"obs": np.array([x[CORTE - 1] for x in acums]),
                    "kopt": int(np.argmax(a)) + 1, "max": a.max(), "filas": len(X)}
        print(f"  {etq:16} filas={res[etq]['filas']:>7,}  @{CORTE//1000}k="
              f"{res[etq]['obs'].mean()/1e6:>7,.1f}M  K*={res[etq]['kopt']:>6,}  "
              f"max={res[etq]['max']/1e6:>7,.1f}M  [{time.time()-t0:.0f}s]")

    print("\n=== pareado por semilla contra '202103+202104' (lo que usamos hoy) ===")
    ref = res["202103+202104"]["obs"]
    for etq, v in res.items():
        d = v["obs"] - ref
        if np.allclose(d, 0):
            continue
        print(f"  {etq:16} {d.mean()/1e6:+7.1f} M   gana en {(d>0).sum()}/{len(d)}   "
              f"p={wilcoxon(d).pvalue:.4f}")
    sd = np.std([v["obs"].mean() for v in res.values()])
    print(f"\nsd entre semillas dentro de un conjunto: "
          f"{np.mean([v['obs'].std() for v in res.values()])/1e6:,.1f} M")
    print(f"diferencia entre conjuntos: {sd/1e6:,.1f} M")
    pd.DataFrame({k: v["obs"] for k, v in res.items()}).to_parquet(
        CARPETA / "por_semilla.parquet", index=False)
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
