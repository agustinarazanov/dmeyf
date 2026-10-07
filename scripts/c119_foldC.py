"""Fold C: validar en 202107 contra el target BAJA+1.

BAJA+1 necesita UN mes de futuro, BAJA+2 necesita dos. Por eso 202107 tiene
1.103 BAJA+1 etiquetados y cero BAJA+2, y por eso habilita un fold que no
existia:

    fold A:  train [202103]                  -> 202105  (BAJA+2)
    fold B:  train [202103,202104]           -> 202106  (BAJA+2)
    fold C:  train [202103,202104,202105]    -> 202107  (BAJA+1)   <- nuevo

Es valido: BAJA+1 en 202107 significa "esta en 202107, no esta en 202108", y
202108 lo TENEMOS. Es dato observado. Y los positivos de train (BAJA+2 hasta
202105) ya se fueron para 202107, asi que los conjuntos son disjuntos.

PARA QUE SIRVE Y PARA QUE NO
  - Sirve para ORDENAR configuraciones, y sobre todo para contestar la pregunta
    de la recencia: es el unico fold que permite entrenar con 202105. El fold B
    obliga a entrenar con marzo y abril, los dos meses MAS VIEJOS del panel.
  - NO sirve para el corte ni para el nivel: la ganancia paga por BAJA+2, y aca
    el target es BAJA+1. El numero que sale NO son pesos, es un proxy.
  - Tiene un sesgo conocido: favorece las señales TARDIAS. El cierre de tarjeta
    es x11 en la ultima foto y x4,6 un mes antes.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

MES_VAL, CORTE, NBR = 202107, 11_000, 250
TARGET, PESO = "pesos", 0.25
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c119_foldC"
CARPETA.mkdir(parents=True, exist_ok=True)

# la pregunta que solo este fold puede contestar
VARIANTES = {
    "todos los meses":      ([202103, 202104, 202105], 1.00),
    "decaimiento l=0,15":   ([202103, 202104, 202105], 0.15),
    "solo el mas reciente": ([202105], 1.00),
    "los dos recientes":    ([202104, 202105], 1.00),
}


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    sem = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar("competencia_01.parquet")
    pred = c.columnas_predictoras(data)
    val = data[data[c.fe.MES] == MES_VAL]
    # el target del fold C: BAJA+1, que en 202107 SI esta etiquetado
    es_baja1 = (val[c.fe.CLASE].to_numpy() == "BAJA+1")
    print(f"fold C: valida {MES_VAL}, {es_baja1.sum()} BAJA+1 de {len(val):,} filas "
          f"({es_baja1.mean():.2%})\n")

    res = {}
    for etq, (meses, lam) in VARIANTES.items():
        X, y, w = c.preparar(data, meses, TARGET, PESO)
        X = X[pred]
        sub = data[data[c.fe.MES].isin(meses)]
        edad = ((max(meses)//100*12 + max(meses)%100)
                - (sub[c.fe.MES]//100*12 + sub[c.fe.MES]%100)).to_numpy()
        w = w * (lam ** edad)
        obs = []
        for s in sem:
            cl = c.clave(PARAMS, meses, TARGET, PESO, f"foldC{lam}", NBR, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                    f"{etq.replace(' ','_')}_{s}_{cl}")
            # proxy: misma mecanica de ganancia pero pagando por BAJA+1
            obs.append(c.ganancia_acumulada(m.predict(val[pred]), es_baja1)[CORTE - 1])
        res[etq] = np.array(obs)
        print(f"  {etq:22} {len(X):>7,} filas  proxy={np.mean(obs)/1e6:>7,.1f}  "
              f"(sd {np.std(obs)/1e6:.1f})  [{time.time()-t0:.0f}s]")

    print("\n=== pareado por semilla contra 'todos los meses' ===")
    ref = res["todos los meses"]
    for etq, v in res.items():
        d = v - ref
        if np.allclose(d, 0):
            continue
        print(f"  {etq:22} {d.mean()/1e6:+7.1f}  gana en {(d>0).sum()}/{len(d)}  "
              f"p={wilcoxon(d).pvalue:.4f}")
    pd.DataFrame(res).to_parquet(CARPETA / "por_semilla.parquet", index=False)
    print("\nRECORDAR: el numero NO son pesos (el target es BAJA+1). Sirve para ordenar.")
    print("Contraste con el fold B (BAJA+2, 202106), donde el decaimiento daba +8,0 M (10/10).")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
