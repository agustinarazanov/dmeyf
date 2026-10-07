"""Elegir el corte que maximiza la ganancia PRIVADA esperada, no el argmax publico.

Elegimos 9.000 por ser el argmax de la curva publica. Pero el publico es un sorteo del
25% y z301 midio que publico y privado correlacionan -1 para un modelo fijo: el K donde
el publico rindio mas es justo el K donde esa banda tuvo EXCESO de aciertos visibles,
que son los que FALTAN en el privado.

Dos fuentes de evidencia:
  A) la curva publica real de agosto, medida con 20 archivos por punto (error 0,33 M)
  B) los dos folds etiquetados, donde SI se puede ver el privado y medir el sesgo

El consejo de la catedra ("quedarse del lado alto del corte") se verifica en B y se
aplica a A.
"""

import time

import numpy as np
import pandas as pd

import competencia as c

# la curva publica medida el 2026-10-06, 20 archivos por punto, mismo modelo y semillas
PUBLICO = {8_000: (90.2275, 2.2529), 9_000: (94.2232, 2.0755), 11_000: (91.3550, 1.4873),
           13_000: (87.1530, 1.7933), 15_000: (83.1834, 1.8796)}
N_ARCHIVOS = 20
PARAMS = {"objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
          "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
          "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
          "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1}
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]


def aportes_ordenados(data, meses_train, mes_val, semilla):
    """El vector de aporte del mes, en orden de ranking del modelo."""
    pred = c.columnas_predictoras(data)
    X, y, w = c.preparar(data, meses_train, "pesos", 0.25)
    v = data[data[c.fe.MES] == mes_val]
    es = (v["clase_ternaria"].to_numpy() == "BAJA+2")        # de SU propio v
    clave = c.clave(PARAMS, meses_train, "pesos", 0.25, "base150", 250, semilla)
    m = c.entrenar_o_cargar(PARAMS, X[pred], y, w, 250, semilla,
                            c.EXPERIMENTOS / "c137_combo",
                            f"base_{mes_val}_{semilla}_{clave}")
    return c.aporte(es)[np.argsort(-m.predict(v[pred]))]


def sesgo_del_argmax(ap, Ks, repeticiones=2000, semilla=314159):
    """Cuanto se equivoca el argmax publico, y cuanto cuesta seguirlo."""
    n = len(ap)
    tot = np.cumsum(ap)
    k_real = Ks[np.argmax(tot[Ks - 1])]
    rng = np.random.default_rng(semilla)
    argmax, priv_argmax = np.empty(repeticiones, dtype=int), np.empty(repeticiones)
    priv_fijo = {K: np.empty(repeticiones) for K in Ks}
    for r in range(repeticiones):
        mk = np.zeros(n, dtype=bool)
        mk[rng.choice(n, size=int(round(n * 0.25)), replace=False)] = True
        cp, cq = np.cumsum(ap * mk), np.cumsum(ap * ~mk)
        k = Ks[np.argmax(cp[Ks - 1])]
        argmax[r], priv_argmax[r] = k, cq[k - 1]
        for K in Ks:
            priv_fijo[K][r] = cq[K - 1]
    # OJO: "el mejor K fijo" se elige MIRANDO la respuesta, asi que compararlo contra
    # el argmax es circular y siempre gana. Lo que vale es comparar contra Ks elegidos
    # SIN mirar: por eso se reportan los candidatos razonables, no el optimo con hindsight.
    candidatos = [K for K in (8000, 9000, 9250, 10000, 11000, 12000) if K in priv_fijo]
    return {"k_real": int(k_real), "argmax_medio": float(argmax.mean()),
            "argmax_sd": float(argmax.std()), "sesgo": float(argmax.mean() - k_real),
            "priv_argmax": float(priv_argmax.mean() / 1e6),
            "priv_candidatos": {int(K): float(priv_fijo[K].mean() / 1e6) for K in candidatos}}


def main():
    t0 = time.time()
    data = c.cargar("competencia_01.parquet")
    Ks = np.arange(6000, 16001, 250)

    print("=== A) que tan bien estima el argmax publico al optimo real ===\n")
    res = {}
    for meses, val in FOLDS:
        ap = aportes_ordenados(data, meses, val, c.SEMILLAS[0])
        r = sesgo_del_argmax(ap, Ks)
        res[val] = r
        n_baja = int((data[data[c.fe.MES] == val]["clase_ternaria"] == "BAJA+2").sum())
        print(f"  fold {val} ({n_baja} BAJA+2)")
        print(f"    optimo real          K = {r['k_real']:,}")
        print(f"    argmax publico       K = {r['argmax_medio']:,.0f}  (sd {r['argmax_sd']:,.0f})"
              f"   sesgo {r['sesgo']:+,.0f}")
        print(f"    seguir el argmax     {r['priv_argmax']:7.2f} M de ganancia privada")
        for K, g in r["priv_candidatos"].items():
            print(f"      K fijo en {K:6,}   {g:7.2f} M   ({g - r['priv_argmax']:+5.2f} contra el argmax)")
        print()

    print("=== B) asimetria: quedarse corto contra pasarse ===\n")
    for meses, val in FOLDS:
        ap = aportes_ordenados(data, meses, val, c.SEMILLAS[0])
        tot = np.cumsum(ap)
        k = res[val]["k_real"]
        print(f"  fold {val}, alrededor de K={k:,}:")
        for delta in (-3000, -2000, -1000, 0, 1000, 2000, 3000):
            K = k + delta
            if 0 < K <= len(ap):
                print(f"    {delta:+6,}  ->  {(tot[K-1]-tot[k-1])/1e6:+7.2f} M")
        print()

    print("=== C) la curva publica de AGOSTO y el ajuste cuadratico ===\n")
    ks = np.array(sorted(PUBLICO)); ys = np.array([PUBLICO[k][0] for k in ks])
    se = np.array([PUBLICO[k][1] / np.sqrt(N_ARCHIVOS) for k in ks])
    coef = np.polyfit(ks, ys, 2, w=1 / se)
    vertice = -coef[1] / (2 * coef[0])
    print(f"  puntos medidos: " + "  ".join(f"{k:,}={y:.2f}" for k, y in zip(ks, ys)))
    print(f"  vertice de la parabola ponderada: K = {vertice:,.0f}")
    print(f"  (el argmax de los puntos medidos es {ks[np.argmax(ys)]:,})")

    print("\n=== D) recomendacion ===\n")
    sesgos = [res[v]["sesgo"] for v in res]
    print(f"  sesgo del argmax publico en los folds: {sesgos[0]:+,.0f} y {sesgos[1]:+,.0f}"
          f"  (promedio {np.mean(sesgos):+,.0f})")
    print(f"  NO hay direccion consistente: el argmax sale largo en el mes flaco y corto en el gordo.")
    print(f"  Y la asimetria a favor del lado alto solo aparece a +-2.000 o mas; cerca del")
    print(f"  optimo, en el fold flaco, pasarse cuesta MAS que quedarse corto.")
    print(f"\n  => el argmax publico de agosto es 9.000; el vertice ajustado es {vertice:,.0f}.")
    print(f"     Con la asimetria a favor del lado alto, el candidato es "
          f"{int(round(max(9000, vertice)/250)*250):,}.")
    print(f"\nlisto en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
