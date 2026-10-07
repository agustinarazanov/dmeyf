"""Fase 1: base vs FE y los cuatro esquemas de target, en los mismos folds.

Cada configuracion se evalua EN SU PROPIO CORTE optimo (elegido por minimax
regret sobre los dos folds), nunca a 0,025 fijo: fusionar clases cambia la
escala de los scores, y comparar a umbral fijo esta amañado.

La comparacion es PAREADA por (fold, semilla) y se cierra con Wilcoxon, porque
la ganancia es una suma de saltos discretos y no hay razon para creerle
normalidad.
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import competencia as c
import registro as r

FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
NUM_BOOST_ROUND = 250
DATASETS = {"base": "competencia_01.parquet", "fe": "competencia_01_fe.parquet"}
ESQUEMAS = [("baja2", 0.0), ("pesos", 0.25), ("pesos", 0.50), ("baja12", 1.0)]

PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}

CARPETA = c.EXPERIMENTOS / "c102_comparacion"
CARPETA.mkdir(parents=True, exist_ok=True)


def evaluar(data, pred, nombre, dataset, target, peso) -> dict:
    curvas_por_fold, observaciones = {}, []
    for meses, mes_val in FOLDS:
        X, y, w = c.preparar(data, meses, target, peso)
        val = data[data[c.fe.MES] == mes_val]
        es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        Xval = val[pred]
        acums = []
        for semilla in c.SEMILLAS:
            cl = c.clave(PARAMS, meses, target, peso, dataset, NUM_BOOST_ROUND, semilla)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NUM_BOOST_ROUND, semilla,
                                    CARPETA, f"{nombre}_{mes_val}_{semilla}_{cl}")
            acum = c.ganancia_acumulada(m.predict(Xval), es)
            acums.append(acum)
            observaciones.append({"fold": mes_val, "semilla": semilla, "acum": acum})
        curvas_por_fold[f"{mes_val} ({es.sum()} B+2)"] = np.mean(acums, axis=0)

    corte, tabla = c.corte_minimax(curvas_por_fold)
    for o in observaciones:
        o["ganancia"] = o["acum"][corte - 1]
    return {
        "nombre": nombre, "dataset": dataset, "target": target, "peso": peso,
        "corte": corte,
        "peor_caso": float(tabla.loc[corte, "peor_caso"]),
        "obs": pd.DataFrame([{k: v for k, v in o.items() if k != "acum"}
                             for o in observaciones]),
        "curvas": curvas_por_fold,
    }


def main() -> None:
    t0 = time.time()
    resultados = []
    for clave_ds, archivo in DATASETS.items():
        data = c.cargar(archivo)
        pred = c.columnas_predictoras(data)
        print(f"\n=== {clave_ds}: {len(pred)} predictoras ===")
        for target, peso in ESQUEMAS:
            etq = f"{clave_ds}_{target}" + (f"{peso:g}".replace("0.", "") if target == "pesos" else "")
            res = evaluar(data, pred, etq, archivo, target, peso)
            g = res["obs"].groupby("fold")["ganancia"].mean()
            resultados.append(res)
            print(f"  {etq:16} corte={res['corte']:>6,}  "
                  + "  ".join(f"{k}={v/1e6:,.1f}M" for k, v in g.items())
                  + f"  peor_regret={res['peor_caso']:+.2%}  [{time.time()-t0:.0f}s]")

    # ---- tabla resumen, normalizada por fold para poder promediar ----
    filas = []
    for res in resultados:
        o = res["obs"]
        filas.append({
            "config": res["nombre"], "corte": res["corte"],
            "g_202105": o[o.fold == 202105].ganancia.mean(),
            "g_202106": o[o.fold == 202106].ganancia.mean(),
            "regret_peor": res["peor_caso"],
        })
    resumen = pd.DataFrame(filas)
    base = {r_["nombre"]: r_ for r_ in resultados}["base_baja2"]
    for r_ in resultados:
        d = r_["obs"].ganancia.to_numpy() - base["obs"].ganancia.to_numpy()
        p = np.nan if np.allclose(d, 0) else wilcoxon(d).pvalue
        resumen.loc[resumen.config == r_["nombre"], "delta_vs_base"] = d.mean()
        resumen.loc[resumen.config == r_["nombre"], "gana_en"] = f"{(d > 0).sum()}/{len(d)}"
        resumen.loc[resumen.config == r_["nombre"], "p_wilcoxon"] = p
    resumen = resumen.sort_values("delta_vs_base", ascending=False)
    resumen.to_parquet(CARPETA / "resumen.parquet", index=False)

    print("\n=== resumen (pareado por fold x semilla contra base_baja2) ===")
    m = resumen.copy()
    for col in ("g_202105", "g_202106", "delta_vs_base"):
        m[col] = (m[col] / 1e6).round(1)
    m["regret_peor"] = (m["regret_peor"] * 100).round(2)
    m["p_wilcoxon"] = m["p_wilcoxon"].round(4)
    print(m.to_string(index=False))

    ganador = resumen.iloc[0]
    print(f"\nmejor: {ganador.config}  (+{ganador.delta_vs_base/1e6:,.1f} M sobre base_baja2, "
          f"gana en {ganador.gana_en}, p={ganador.p_wilcoxon:.4f})")
    (CARPETA / "params.json").write_text(json.dumps(
        {"params": PARAMS, "folds": [[m_, v] for m_, v in FOLDS],
         "num_boost_round": NUM_BOOST_ROUND}, indent=2))

    with r.conectar() as con:
        for res in resultados:
            o = res["obs"]
            r.alta_experimento(
                con, res["nombre"], notebook="c102_comparacion.py",
                descripcion="fase 1: base vs FE x cuatro esquemas de target",
                dataset=res["dataset"], n_columnas=None,
                meses_train=[202103, 202104], mes_validacion=202106,
                target=res["target"], peso_baja1=res["peso"], params=PARAMS,
                num_boost_round=NUM_BOOST_ROUND, n_semillas=len(c.SEMILLAS),
                corte_envios=int(res["corte"]),
                ganancia_val_media=float(o.ganancia.mean()),
                ganancia_val_sd=float(o.ganancia.std()), ruta=str(CARPETA))
    print(f"\nlisto en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
