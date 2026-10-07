"""¿Las 14 features de v101 le agregan algo al modelo?

Lift alto no alcanza: el arbol puede llegar solo a 'rojo y sin acuerdo' con dos
cortes. Lo que no puede inventar es lo ACUMULADO (meses_en_rojo, meses_sin_trx),
porque no esta en la fila. Esto separa las dos cosas:

  base          150 columnas
  v101          base + las 14
  v101_hist     base + solo las acumuladas (lo que el arbol no puede derivar)
  v101_fila     base + solo las de la misma fila (lo que si podria derivar)
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
TARGET, PESO, NBR = "pesos", 0.25, 250
N_SEMILLAS = 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
HIST = ("v_meses_en_rojo", "v_meses_sin_trx", "v_ctrx_minimo", "v_ctrx_delta")
CARPETA = c.EXPERIMENTOS / "c112_v101"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    data = c.cargar("competencia_01_v101.parquet")
    todas = c.columnas_predictoras(data)
    nuevas = [x for x in todas if x.startswith("v_")]
    base = [x for x in todas if not x.startswith("v_")]
    sets = {
        "base": base,
        "v101": todas,
        "v101_hist": base + [x for x in nuevas if x in HIST],
        "v101_fila": base + [x for x in nuevas if x not in HIST],
    }
    print(f"{len(base)} columnas base, {len(nuevas)} nuevas "
          f"({len([x for x in nuevas if x in HIST])} acumuladas)\n")

    res = {}
    for etq, pred in sets.items():
        obs, escen = [], {}
        for meses, mes_val in FOLDS:
            X, y, w = c.preparar(data, meses, TARGET, PESO)
            X = X[pred]
            val = data[data[c.fe.MES] == mes_val]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            acums = []
            for s in semillas:
                cl = c.clave(PARAMS, meses, TARGET, PESO, f"{etq}{len(pred)}", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                        f"{etq}_{mes_val}_{s}_{cl}")
                acums.append(c.ganancia_acumulada(m.predict(val[pred]), es))
            escen[str(mes_val)] = np.mean(acums, axis=0)
            obs.append(acums)
        corte, tabla = c.corte_minimax(escen)
        plano = np.array([a[corte - 1] for acs in obs for a in acs])
        res[etq] = {"obs": plano, "corte": corte,
                    "m05": escen["202105"][corte - 1], "m06": escen["202106"][corte - 1],
                    "ncols": len(pred)}
        print(f"  {etq:11} {len(pred):>3} cols  corte={corte:>6,}  "
              f"202105={escen['202105'][corte-1]/1e6:>6,.1f}M  "
              f"202106={escen['202106'][corte-1]/1e6:>6,.1f}M  "
              f"media={plano.mean()/1e6:>6,.1f}M  [{time.time()-t0:.0f}s]")

    print("\n=== pareado por (fold, semilla) contra base ===")
    ref = res["base"]["obs"]
    for etq, v in res.items():
        d = v["obs"] - ref
        if np.allclose(d, 0):
            continue
        print(f"  {etq:11} {d.mean()/1e6:+7.1f} M   gana en {(d>0).sum()}/{len(d)}   "
              f"p={wilcoxon(d).pvalue:.4f}")
    pd.DataFrame({k: v["obs"] for k, v in res.items()}).to_parquet(
        CARPETA / "por_semilla.parquet", index=False)

    import lightgbm as lgb
    mod = sorted(CARPETA.glob("v101_202106_261431_*.txt"))
    if mod:
        m = lgb.Booster(model_file=str(mod[0]))
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        imp = (imp / imp.sum() * 100).sort_values(ascending=False)
        print("\ndonde quedaron las nuevas en importancia (de "
              f"{len(imp)} columnas):")
        for n in nuevas:
            if n in imp.index:
                print(f"  {n:22} {imp[n]:>5.2f}%   puesto {list(imp.index).index(n)+1}")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
