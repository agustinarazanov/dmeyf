"""¿202104 gana por ser mas RECIENTE o por tener mas POSITIVOS?

202104 tiene 1.139 BAJA+2 y 202103 tiene 960: las dos explicaciones estan
confundidas, igual que pasaba con el corte. Se separa submuestreando 202104
hasta dejarlo con los mismos 960 positivos que 202103.

  - si 202104 submuestreado sigue ganando -> es el MES (recencia/calidad)
  - si cae al nivel de 202103             -> eran los positivos de mas
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

MES_VAL, CORTE = 202106, 11_000
TARGET, PESO, NBR = "pesos", 0.25, 250
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c109_recencia"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar()
    pred = c.columnas_predictoras(data)
    val = data[data[c.fe.MES] == MES_VAL]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")

    n03 = int((data[data[c.fe.MES] == 202103][c.fe.CLASE] == "BAJA+2").sum())
    n04 = int((data[data[c.fe.MES] == 202104][c.fe.CLASE] == "BAJA+2").sum())
    print(f"202103 tiene {n03} BAJA+2 | 202104 tiene {n04} | se recorta 202104 a {n03}\n")

    def evaluar(etq, filas, semilla_sub=None):
        X, y, w = c.preparar(filas, [202103, 202104], TARGET, PESO)
        X = X[pred]
        acums = []
        for s in semillas:
            cl = c.clave(PARAMS, [len(X), semilla_sub or 0], TARGET, PESO, etq, NBR, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"{etq}_{s}_{cl}")
            acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
        obs = np.array([a[CORTE - 1] for a in acums])
        pos = int((filas[c.fe.CLASE] == "BAJA+2").sum())
        print(f"  {etq:22} filas={len(X):>7,} pos={pos:>5,}  @11k={obs.mean()/1e6:>7,.1f}M "
              f"(sd {obs.std()/1e6:.1f})  [{time.time()-t0:.0f}s]")
        return obs

    m03 = data[data[c.fe.MES] == 202103]
    m04 = data[data[c.fe.MES] == 202104]
    # recorte: se tiran positivos de 202104 al azar hasta igualar a 202103
    rng = np.random.default_rng(c.SEMILLAS[1])
    pos04 = m04.index[m04[c.fe.CLASE] == "BAJA+2"].to_numpy()
    tirar = rng.choice(pos04, size=n04 - n03, replace=False)
    m04_rec = m04.drop(index=tirar)

    res = {
        "202103 (960 pos)": evaluar("m03", m03),
        "202104 (1139 pos)": evaluar("m04", m04),
        "202104 recortado": evaluar("m04rec", m04_rec, semilla_sub=int(c.SEMILLAS[1])),
    }

    print("\n=== pareado por semilla ===")
    for a, b in [("202104 recortado", "202103 (960 pos)"), ("202104 (1139 pos)", "202104 recortado")]:
        d = res[a] - res[b]
        print(f"  {a} vs {b}: {d.mean()/1e6:+6.1f} M  gana en {(d>0).sum()}/{len(d)}  "
              f"p={wilcoxon(d).pvalue:.4f}")

    perdido = (res["202104 (1139 pos)"] - res["202104 recortado"]).mean()
    ganado = (res["202104 recortado"] - res["202103 (960 pos)"]).mean()
    print(f"\ncon los mismos 960 positivos, 202104 le saca {ganado/1e6:+,.1f} M a 202103")
    print(f"los 179 positivos extra valen {perdido/1e6:+,.1f} M")
    print(f"\n-> {'manda el MES' if abs(ganado) > abs(perdido) else 'mandan los POSITIVOS'}")
    pd.DataFrame(res).to_parquet(CARPETA / "por_semilla.parquet", index=False)
    print(f"listo en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
