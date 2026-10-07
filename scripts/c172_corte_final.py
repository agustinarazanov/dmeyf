"""EL CORTE FINAL, decidido con los folds y no con el argmax publico.

Por que cambia: el argmax publico (9.000) tiene z=+0,72 contra 11.000, o sea no
distingue. El publico tiene ~275 positivos; los folds tienen ~1.000. Pero el publico
SI midio bien una cosa: la FORMA de la curva, que cae 11 M entre 9.000 y 15.000 y
descarta la banda alta.

Metodo: la curva de CAPTURA de cada fold no depende de la prevalencia. Se la combina
con cada prevalencia posible de agosto (870 a 1.139, el rango observado) para armar la
curva de ganancia, y se elige el K que minimiza el peor arrepentimiento.
"""
import time

import numpy as np

import competencia as c

PARAMS = {"objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
          "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
          "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
          "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1}
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
PREVALENCIAS = [870, 960, 1000, 1098, 1139]        # el rango observado en los meses etiquetados
KS = np.arange(7000, 14001, 250)
N_AGOSTO = 164_647


def curva_de_captura(data, meses, mes_val, semillas):
    """fraccion de los que se van que cae en el top K. No depende de la prevalencia."""
    pred = c.columnas_predictoras(data)
    X, y, w = c.preparar(data, meses, "pesos", 0.25)
    v = data[data[c.fe.MES] == mes_val]
    es = (v["clase_ternaria"].to_numpy() == "BAJA+2")
    caps = []
    for s in semillas:
        cl = c.clave(PARAMS, meses, "pesos", 0.25, "base150", 250, s)
        m = c.entrenar_o_cargar(PARAMS, X[pred], y, w, 250, s,
                                c.EXPERIMENTOS / "c137_combo", f"base_{mes_val}_{s}_{cl}")
        o = np.argsort(-m.predict(v[pred]))
        caps.append(np.array([es[o[:K]].sum() / es.sum() for K in KS]))
    return np.mean(caps, axis=0)


def main():
    t0 = time.time()
    data = c.cargar("competencia_01.parquet")
    sem = c.SEMILLAS + [112909, 314159, 562991, 733517, 951413]

    curvas = {}
    for meses, val in FOLDS:
        curvas[val] = curva_de_captura(data, meses, val, sem)
        print(f"  curva de captura del fold {val} lista   [{time.time()-t0:.0f}s]")

    print(f"\n  captura por corte (promedio de los dos folds):")
    prom = np.mean(list(curvas.values()), axis=0)
    for K in (8000, 9000, 10000, 11000, 12000, 13000):
        print(f"    K={K:6,}  {100*prom[list(KS).index(K)]:5.1f}%")

    # escenarios: cada (fold, prevalencia) da una curva de ganancia y un optimo
    print(f"\n  optimo por escenario:")
    escenarios = {}
    for val, cap in curvas.items():
        for N in PREVALENCIAS:
            aciertos = cap * N
            g = aciertos * c.GANANCIA_ACIERTO - (KS - aciertos) * c.COSTO_ESTIMULO
            escenarios[(val, N)] = g
            print(f"    fold {val}, prevalencia {N:5,}  ->  K* = {KS[np.argmax(g)]:6,}"
                  f"   (max {g.max()/1e6:6.1f} M)")

    print(f"\n  arrepentimiento relativo por K (peor caso sobre los {len(escenarios)} escenarios):")
    regret = {}
    for i, K in enumerate(KS):
        r = [(g.max() - g[i]) / g.max() for g in escenarios.values()]
        regret[K] = (max(r), float(np.mean(r)))
    for K in (8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000):
        peor, medio = regret[K]
        print(f"    K={K:6,}  peor {100*peor:5.2f}%   medio {100*medio:5.2f}%")

    k_mm = min(regret, key=lambda K: regret[K][0])
    k_me = min(regret, key=lambda K: regret[K][1])
    print(f"\n  MINIMAX (minimiza el peor arrepentimiento):  K = {k_mm:,}  "
          f"(peor {100*regret[k_mm][0]:.2f}%)")
    print(f"  MINIMO ESPERADO:                             K = {k_me:,}  "
          f"(medio {100*regret[k_me][1]:.2f}%)")
    print(f"\n  el publico (que no distingue dentro de la banda, pero SI descarta la banda alta)")
    print(f"  mide un maximo en 9.000 y una caida de 11 M hasta 15.000.")
    print(f"\nlisto en {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
