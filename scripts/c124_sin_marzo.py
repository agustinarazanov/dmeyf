"""¿El modelo final deberia entrenar sin 202103?

«Sacarle peso a marzo» se sostuvo en todas las mediciones: 202104 le gana a
202103 por +34,2 M con los positivos IGUALADOS (10/10, p=0,002), y entrenar con
un mes solo le gana a entrenar con dos (+10,6 M, 18/20). Pero el modelo final
sigue entrenando con los cuatro meses, marzo incluido.

No se puede validar directo (haria falta la etiqueta de 202108), asi que se mide
en los dos folds que tenemos, los dos con el mismo diseno:

  fold B: train <= 202104, valida 202106 con target BAJA+2  (ganancia real)
  fold C: train <= 202105, valida 202107 con target BAJA+1  (proxy; el unico
          fold que permite entrenar con 202105, el mes mas nuevo)

En cada uno se compara «con marzo» contra «sin marzo», a cuatro cortes.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

CORTES = [9_000, 10_000, 11_000, 12_000]
TARGET, PESO, NBR = "pesos", 0.25, 250
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
# (nombre del fold, mes de validacion, clase que paga, variantes de train)
ESCENARIOS = [
    ("fold B (202106, BAJA+2)", 202106, "BAJA+2",
     {"con marzo": [202103, 202104], "sin marzo": [202104]}),
    ("fold C (202107, BAJA+1)", 202107, "BAJA+1",
     {"con marzo": [202103, 202104, 202105], "sin marzo": [202104, 202105],
      "solo 202105": [202105]}),
]
CARPETA = c.EXPERIMENTOS / "c124_sin_marzo"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    sem = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar("competencia_01.parquet")
    pred = c.columnas_predictoras(data)
    filas = []

    for nombre, mes_val, paga, variantes in ESCENARIOS:
        val = data[data[c.fe.MES] == mes_val]
        es = (val[c.fe.CLASE].to_numpy() == paga)
        print(f"\n{nombre}: {es.sum()} positivos de {len(val):,}")
        res = {}
        for etq, meses in variantes.items():
            X, y, w = c.preparar(data, meses, TARGET, PESO)
            X = X[pred]
            acums = []
            for s in sem:
                cl = c.clave(PARAMS, meses, TARGET, PESO, "c124", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                        f"{mes_val}_{etq.replace(' ','')}_{s}_{cl}")
                acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
            res[etq] = {k: np.array([a[k - 1] for a in acums]) for k in CORTES}
            linea = "  ".join(f"{k//1000}k:{res[etq][k].mean()/1e6:>6,.1f}" for k in CORTES)
            print(f"  {etq:12} {len(X):>7,} filas  {linea}  [{time.time()-t0:.0f}s]")

        ref = res["con marzo"]
        for etq in variantes:
            if etq == "con marzo":
                continue
            celdas = []
            for k in CORTES:
                d = res[etq][k] - ref[k]
                p = wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
                celdas.append(f"{d.mean()/1e6:+6.1f} {(d>0).sum():>2}/{len(d)} p={p:.2f}")
                filas.append({"fold": nombre, "variante": etq, "corte": k,
                              "delta": d.mean(), "gana": int((d > 0).sum()), "p": p})
            print(f"    {etq} vs con marzo: " + " | ".join(celdas))

    df = pd.DataFrame(filas)
    df.to_parquet(CARPETA / "resultado.parquet", index=False)
    print("\n=== resumen: delta de 'sin marzo' contra 'con marzo' ===")
    sm = df[df.variante == "sin marzo"]
    print(sm.pivot(index="corte", columns="fold", values="delta").div(1e6).round(1).to_string())
    print(f"\ngana en los dos folds y en los {len(CORTES)} cortes: "
          f"{'SI' if (sm.delta > 0).all() else 'NO'}")
    print(f"peor corte de 'sin marzo': {sm.delta.min()/1e6:+.1f} M")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
