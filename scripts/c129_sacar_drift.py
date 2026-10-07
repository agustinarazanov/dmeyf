"""Sacar las variables con concept drift: ¿mejora o empeora?

c128 barrio las 150 variables midiendo el AUC univariado mes a mes. Las que
tienen PSI bajo (la distribucion no se mueve) pero AUC que SI se mueve son las
peligrosas: lo que significan cambia sin que ningun control lo detecte.

Las dos de prestamo encabezan la lista y cruzan el 0,50 en junio: de proteger
pasan a no decir nada. Pero sacar una variable con drift no es gratis — tambien
se va su señal. Hay que medirlo.

Se mide lo mismo que en c127: ganancia en los dos folds Y cuanta captura se
pierde al alejarse un mes (distancia 2 vs 3, mismo mes de train).
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

KS = (9_000, 11_000)
NBR, N_SEMILLAS = 250, 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c129_sacar_drift"
CARPETA.mkdir(parents=True, exist_ok=True)

PRESTAMOS = ["mprestamos_personales", "cprestamos_personales"]
VISA_DERIVA = ["Visa_msaldopesos", "Visa_msaldototal", "Visa_mpagominimo",
               "Visa_mlimitecompra", "Visa_mfinanciacion_limite", "Visa_fechaalta"]


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    sem = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    d = c.cargar("competencia_01.parquet")
    base = c.columnas_predictoras(d)
    drift = pd.read_parquet(c.EXPERIMENTOS / "c128_drift.parquet")
    top15 = drift[(drift.psi_mar_jun < 0.1) & (drift.senal_max > 0.02)] \
        .nlargest(15, "rango_auc")["var"].tolist()

    sets = {
        "base":            base,
        "sin prestamos":   [x for x in base if x not in PRESTAMOS],
        "sin prest+visa":  [x for x in base if x not in PRESTAMOS + VISA_DERIVA],
        "sin top15 drift": [x for x in base if x not in top15],
    }
    for k, v in sets.items():
        print(f"  {k:17} {len(v)} columnas (saca {len(base)-len(v)})")

    # --- ganancia en los dos folds ---
    FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
    gan = {k: {x: [] for x in KS} for k in sets}
    cap = {}
    print(f"\n{'set':17} " + " ".join(f"gan@{k//1000}k".rjust(10) for k in KS))
    for etq, pred in sets.items():
        for meses, mv in FOLDS:
            X, y, w = c.preparar(d, meses, "pesos", 0.25)
            X = X[pred]
            val = d[d[c.fe.MES] == mv]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            for s in sem:
                cl = c.clave(PARAMS, meses, "pesos", 0.25, f"{etq}{len(pred)}", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                        f"{etq.replace(' ','')}_{mv}_{s}_{cl}")
                acum = c.ganancia_acumulada(m.predict(val[pred]), es)
                for k in KS:
                    gan[etq][k].append(acum[k - 1])
        print(f"  {etq:15} " + " ".join(f"{np.mean(gan[etq][k])/1e6:>9,.1f}M" for k in KS)
              + f"   [{time.time()-t0:.0f}s]")

    print(f"\n=== pareado contra base, por corte ===")
    for etq in sets:
        if etq == "base":
            continue
        out = []
        for k in KS:
            dd = np.array(gan[etq][k]) - np.array(gan["base"][k])
            p = wilcoxon(dd).pvalue if not np.allclose(dd, 0) else 1.0
            out.append(f"{dd.mean()/1e6:+6.1f}M {(dd>0).sum():>2}/{len(dd)} p={p:.2f}")
        print(f"  {etq:17} " + " | ".join(out))

    # --- envejecimiento: misma receta que c127 ---
    print(f"\n=== captura perdida al pasar de distancia 2 a 3 ===")
    for etq, pred in sets.items():
        caps = {}
        for mv in (202105, 202106):
            X, y, w = c.preparar(d, [202103], "pesos", 0.25)
            X = X[pred]
            val = d[d[c.fe.MES] == mv]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            cc = []
            for s in sem:
                cl = c.clave(PARAMS, [202103], "pesos", 0.25, f"{etq}{len(pred)}", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                        f"{etq.replace(' ','')}_{mv}_{s}_{cl}")
                orden = np.argsort(m.predict(val[pred]))[::-1]
                acum = np.cumsum(es[orden])
                cc.append([acum[k - 1] / es.sum() for k in KS])
            caps[mv] = np.mean(cc, axis=0)
        p = caps[202105] - caps[202106]
        cap[etq] = p
        print(f"  {etq:17} " + " ".join(f"{x:+7.1%}" for x in p) + f"   media {np.mean(p):+.2%}")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
