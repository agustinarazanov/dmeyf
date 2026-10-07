"""Cuatro variantes para sondear agosto, con lectura local y publica.

Los folds validan en JUNIO; el leaderboard mide AGOSTO. Hoy quedo demostrado
que el mes manda (202104 le saca +34 M a 202103 con los mismos positivos), asi
que el publico no es redundante con lo local: mide otro mes. En particular, las
features acumuladas de v101 son planas en los folds porque 202103 no tiene
historia — en agosto tendrian seis meses de recorrido.

  v1_recencia   base + decaimiento lambda=0,15 por antiguedad de mes
  v2_v101       base + las 14 de v101 (incluidas las acumuladas)
  v3_combo      las dos cosas
  v4_reciente   base, entrenando SOLO en 202105+202106 (corte duro)
"""

import json
import time

import numpy as np
import pandas as pd

import competencia as c
import registro as r

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
MESES_TODOS = [202103, 202104, 202105, 202106]
TARGET, PESO, NBR, CORTE = "pesos", 0.25, 250, 11_000
LAMBDA = 0.15
N_FOLD, N_FINAL = 5, 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c113_sonda"
CARPETA.mkdir(parents=True, exist_ok=True)


def edad_meses(serie, ref):
    return ((ref // 100 * 12 + ref % 100) - (serie // 100 * 12 + serie % 100)).to_numpy()


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS + np.random.choice(1_000_000, size=N_FINAL - 5, replace=False).tolist()
    base_df = c.cargar("competencia_01.parquet")
    v101_df = c.cargar("competencia_01_v101.parquet")
    pred_base = c.columnas_predictoras(base_df)
    pred_v101 = c.columnas_predictoras(v101_df)

    VAR = {
        "v1_recencia": dict(data=base_df, pred=pred_base, meses=MESES_TODOS, lam=LAMBDA),
        "v2_v101":     dict(data=v101_df, pred=pred_v101, meses=MESES_TODOS, lam=1.0),
        "v3_combo":    dict(data=v101_df, pred=pred_v101, meses=MESES_TODOS, lam=LAMBDA),
        "v4_reciente": dict(data=base_df, pred=pred_base, meses=[202105, 202106], lam=1.0),
    }
    print(f"corte fijo {CORTE:,} | lambda {LAMBDA} | {N_FOLD} semillas en folds, "
          f"{N_FINAL} en la entrega\n")

    resumen = {}
    for etq, cfg in VAR.items():
        data, pred, lam = cfg["data"], cfg["pred"], cfg["lam"]
        # --- lectura local en los folds (los meses de train se recortan al fold) ---
        locales = []
        for meses_f, mes_val in FOLDS:
            meses = [m for m in cfg["meses"] if m in meses_f] or meses_f
            X, y, w = c.preparar(data, meses, TARGET, PESO)
            X = X[pred]
            sub = data[data[c.fe.MES].isin(meses)]
            w = w * (lam ** edad_meses(sub[c.fe.MES], max(meses)))
            val = data[data[c.fe.MES] == mes_val]
            es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
            g = []
            for s in semillas[:N_FOLD]:
                cl = c.clave(PARAMS, meses, TARGET, PESO, f"{etq}{lam}{len(pred)}", NBR, s)
                m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA,
                                        f"{etq}_f{mes_val}_{s}_{cl}")
                g.append(c.ganancia_acumulada(m.predict(val[pred]), es)[CORTE - 1])
            locales.append(np.mean(g))

        # --- modelo final sobre sus meses -> 202108 ---
        X, y, w = c.preparar(data, cfg["meses"], TARGET, PESO)
        X = X[pred]
        sub = data[data[c.fe.MES].isin(cfg["meses"])]
        w = w * (lam ** edad_meses(sub[c.fe.MES], max(cfg["meses"])))
        fut = data[data[c.fe.MES] == c.MES_COMPETENCIA]
        ids = fut[c.fe.ID].to_numpy()
        validos = set(ids.tolist())
        archivos = []
        for s in semillas:
            cl = c.clave(PARAMS, cfg["meses"], TARGET, PESO, f"{etq}{lam}{len(pred)}", NBR, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"{etq}_fin_{s}_{cl}")
            nom = f"{etq}_s{s}.csv"
            n = c.escribir_envios(c.top_k(m.predict(fut[pred]), ids, CORTE),
                                  CARPETA / "envios" / nom, validos)
            archivos.append({"nombre": nom, "ruta": CARPETA / "envios" / nom,
                             "semilla": int(s), "n_envios": n})
        resumen[etq] = {"local_202105": locales[0], "local_202106": locales[1],
                        "local_media": float(np.mean(locales)), "archivos": archivos,
                        "ncols": len(pred), "meses": cfg["meses"], "lam": lam,
                        "filas": len(X)}
        print(f"  {etq:13} {len(pred):>3} cols  {len(X):>7,} filas  "
              f"local 202105={locales[0]/1e6:>6,.1f}M  202106={locales[1]/1e6:>6,.1f}M  "
              f"media={np.mean(locales)/1e6:>6,.1f}M  [{time.time()-t0:.0f}s]")

    with r.conectar() as con:
        for etq, v in resumen.items():
            eid = r.alta_experimento(
                con, f"c113_{etq}", notebook="c113_sonda.py",
                descripcion=f"sonda de agosto: {etq}. Los folds miden junio, el leaderboard "
                            "mide agosto; con el mes mandando tanto, el publico no es redundante.",
                dataset="competencia_01_v101.parquet" if v["ncols"] > 160 else "competencia_01.parquet",
                n_columnas=v["ncols"], meses_train=v["meses"], mes_validacion=202106,
                target=TARGET, peso_baja1=PESO, params={**PARAMS, "lambda_recencia": v["lam"]},
                num_boost_round=NBR, n_semillas=N_FINAL, corte_envios=CORTE,
                ganancia_val_media=v["local_media"], ruta=str(CARPETA))
            con.execute("update experimento set justificacion_corte=%s where id=%s",
                        (f"Corte {CORTE} heredado de c107 (minimax con 20 semillas, peor caso "
                         "-3,47%). Se fija igual en las cuatro variantes para que la comparacion "
                         "en el leaderboard sea del modelo y no del corte.", eid))
    json.dump({k: {x: y for x, y in v.items() if x != "archivos"} for k, v in resumen.items()},
              open(CARPETA / "resumen.json", "w"), indent=2, default=str)
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
