"""Por que el feature engineering perdia: 202103 no tiene pasado.

En 202103 las 102 columnas de lag/delta/slope son 100% nulas —no existe mes
anterior— y en el resto de los meses son ~1,4%. El patron de nulos es entonces
un identificador perfecto de marzo, o sea vuelve a meter por la ventana el
foto_mes que sacamos por la puerta.

Hipotesis: entrenando SOLO en 202104, donde la historia ya existe, el FE deja
de perder. Todo se valida en 202106 para que el unico cambio sea el dataset.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

MESES_TRAIN = [202104]
MES_VAL = 202106
NUM_BOOST_ROUND = 250
CARPETA = c.EXPERIMENTOS / "c104_diagnostico_fe"
CARPETA.mkdir(parents=True, exist_ok=True)

PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
HISTORIA = ("__lag1", "__delta1", "__slope")


def main() -> None:
    t0 = time.time()
    base = c.cargar("competencia_01.parquet")
    fe = c.cargar("competencia_01_fe.parquet")
    pred_base = c.columnas_predictoras(base)
    pred_fe = c.columnas_predictoras(fe)
    pred_rank = [x for x in pred_fe if not x.endswith(HISTORIA)]

    configs = [
        ("base",     base, pred_base, "baja2",  0.0),
        ("base_p25", base, pred_base, "pesos", 0.25),
        ("fe",       fe,   pred_fe,   "baja2",  0.0),
        ("fe_p25",   fe,   pred_fe,   "pesos", 0.25),
        ("ferank",   fe,   pred_rank, "baja2",  0.0),
        ("ferank_p25", fe, pred_rank, "pesos", 0.25),
    ]
    print(f"train {MESES_TRAIN} -> {MES_VAL} | base {len(pred_base)} cols | "
          f"fe {len(pred_fe)} | fe solo ranks {len(pred_rank)}\n")

    gan = {}
    for nombre, data, pred, target, peso in configs:
        X, y, w = c.preparar(data, MESES_TRAIN, target, peso)
        X = X[pred]
        val = data[data[c.fe.MES] == MES_VAL]
        es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        acums = []
        for semilla in c.SEMILLAS:
            cl = c.clave({**PARAMS, "_cols": len(pred)}, MESES_TRAIN, target, peso,
                         nombre, NUM_BOOST_ROUND, semilla)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NUM_BOOST_ROUND, semilla,
                                    CARPETA, f"{nombre}_{semilla}_{cl}")
            acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
        a = np.mean(acums, axis=0)
        gan[nombre] = {"curva": a, "obs": np.array([x[10_999] for x in acums]),
                       "kopt": int(np.argmax(a)) + 1, "max": a.max()}
        print(f"  {nombre:12} @11.000={a[10_999]/1e6:>7,.1f}M  K*={gan[nombre]['kopt']:>6,}  "
              f"max={a.max()/1e6:>7,.1f}M   [{time.time()-t0:.0f}s]")

    print("\n=== pareado por semilla, a 11.000 envios, contra base ===")
    ref = gan["base"]["obs"]
    filas = []
    for k, v in gan.items():
        d = v["obs"] - ref
        filas.append({"config": k, "gan_M": v["obs"].mean() / 1e6,
                      "delta_M": d.mean() / 1e6, "gana_en": f"{(d > 0).sum()}/{len(d)}",
                      "p": np.nan if np.allclose(d, 0) else wilcoxon(d).pvalue})
    t = pd.DataFrame(filas).sort_values("delta_M", ascending=False)
    print(t.round(3).to_string(index=False))
    t.to_parquet(CARPETA / "resumen.parquet", index=False)

    print("\ncontraste con el fold B de c102 (train 202103+202104, mismo 202106):")
    print("  base_baja2 daba 396,4 M y fe_baja2 349,5 M  -> el FE perdia 46,9 M")
    print(f"  entrenando solo en 202104: base {gan['base']['obs'].mean()/1e6:,.1f} M, "
          f"fe {gan['fe']['obs'].mean()/1e6:,.1f} M  -> "
          f"el FE {'gana' if gan['fe']['obs'].mean() > ref.mean() else 'pierde'} "
          f"{abs(gan['fe']['obs'].mean()-ref.mean())/1e6:,.1f} M")
    print(f"\nlisto en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
