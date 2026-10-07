"""Fase 3: modelo final, 20 semillas, y el ensamble por rank.

Gana base_pesos25: los hiperparametros que z701 ya tenia afinados, con el
target fusionado y BAJA+1 pesando 0,25. Los 31 trials de Optuna no le ganaron
—quedaron ~7,5 M abajo, 5/10, o sea empate— porque su objetivo era CV DENTRO de
los meses de train con el corte elegido post-hoc, y ese objetivo premia fusionar
las clases del todo. Optimizo bien hacia el blanco equivocado.

Antes de escribir nada se mide el ensamble por rank contra las semillas sueltas
en los folds, porque el submit final es de UN archivo y hay que saber cual.
"""

import json
import time

import numpy as np
import pandas as pd

import competencia as c
import registro as r

NOMBRE = "c107_pesos25"
DATASET = "competencia_01.parquet"
MESES_TRAIN = [202103, 202104, 202105, 202106]
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
TARGET, PESO = "pesos", 0.25
NUM_BOOST_ROUND = 250
N_SEMILLAS = 20

PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / NOMBRE


def main() -> None:
    t0 = time.time()
    np.random.seed(c.SEMILLAS[0])
    semillas = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - len(c.SEMILLAS),
                                             replace=False).tolist()
    data = c.cargar(DATASET)
    pred = c.columnas_predictoras(data)
    print(f"{len(pred)} predictoras | {N_SEMILLAS} semillas | target {TARGET} w={PESO}")

    # ---- corte y chequeo del ensamble, en los folds ----
    escenarios, ens_vs_solo = {}, []
    for meses, mes_val in FOLDS:
        X, y, w = c.preparar(data, meses, TARGET, PESO)
        X = X[pred]
        val = data[data[c.fe.MES] == mes_val]
        es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        scores = {}
        for s in semillas:
            cl = c.clave(PARAMS, meses, TARGET, PESO, DATASET, NUM_BOOST_ROUND, s)
            m = c.entrenar_o_cargar(PARAMS, X, y, w, NUM_BOOST_ROUND, s, CARPETA,
                                    f"val{mes_val}_{s}_{cl}")
            scores[s] = m.predict(val[pred])
        acums = [c.ganancia_acumulada(v, es) for v in scores.values()]
        escenarios[str(mes_val)] = np.mean(acums, axis=0)
        ens_vs_solo.append((mes_val, c.ganancia_acumulada(c.ensamble_por_rank(scores), es),
                            acums))
        print(f"  fold -> {mes_val}: {len(scores)} semillas  [{time.time()-t0:.0f}s]")

    corte, tabla = c.corte_minimax(escenarios)
    tabla.to_parquet(CARPETA / "regret.parquet")
    print(f"\ncorte minimax: {corte:,} (peor caso {tabla.loc[corte,'peor_caso']:+.2%})")

    print("\nensamble por rank vs semillas sueltas, en el corte elegido:")
    for mes_val, ens, acums in ens_vs_solo:
        solas = np.array([a[corte - 1] for a in acums])
        e = ens[corte - 1]
        pct = (solas < e).mean()
        print(f"  {mes_val}: ensamble {e/1e6:,.1f}M vs semillas {solas.mean()/1e6:,.1f}M "
              f"(mejor {solas.max()/1e6:,.1f}, peor {solas.min()/1e6:,.1f}) "
              f"-> le gana al {pct:.0%} de las semillas")

    ganancia_media = float(np.mean([e[corte - 1] for _, e, _ in ens_vs_solo]))

    # ---- modelo final sobre los cuatro meses -> 202108 ----
    X, y, w = c.preparar(data, MESES_TRAIN, TARGET, PESO)
    X = X[pred]
    fut = data[data[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    validos = set(ids.tolist())
    print(f"\nfinal: train {MESES_TRAIN} ({len(X):,}) -> {c.MES_COMPETENCIA} ({len(fut):,})")

    scores, archivos = {}, []
    for s in semillas:
        cl = c.clave(PARAMS, MESES_TRAIN, TARGET, PESO, DATASET, NUM_BOOST_ROUND, s)
        m = c.entrenar_o_cargar(PARAMS, X, y, w, NUM_BOOST_ROUND, s, CARPETA, f"fin_{s}_{cl}")
        scores[s] = m.predict(fut[pred])
        nombre = f"{NOMBRE}_s{s}.csv"
        n = c.escribir_envios(c.top_k(scores[s], ids, corte), CARPETA / "envios" / nombre, validos)
        archivos.append({"nombre": nombre, "ruta": CARPETA / "envios" / nombre,
                         "semilla": int(s), "n_envios": n})
    print(f"  {len(archivos)} archivos de {corte:,} envios  [{time.time()-t0:.0f}s]")

    ens = c.ensamble_por_rank(scores)
    nombre_ens = f"{NOMBRE}_ensamble.csv"
    n = c.escribir_envios(c.top_k(ens, ids, corte), CARPETA / "envios" / nombre_ens, validos)
    print(f"  + {nombre_ens} ({n:,} envios)")

    pd.DataFrame({"numero_de_cliente": ids, "ensamble": ens,
                  **{f"s{k}": v for k, v in scores.items()}}
                 ).to_parquet(CARPETA / f"scores_{c.MES_COMPETENCIA}.parquet", index=False)
    (CARPETA / "params.json").write_text(json.dumps(
        {"params": PARAMS, "target": TARGET, "peso_baja1": PESO, "meses_train": MESES_TRAIN,
         "num_boost_round": NUM_BOOST_ROUND, "corte": int(corte),
         "semillas": [int(s) for s in semillas]}, indent=2))

    sol = len(set(c.top_k(scores[semillas[0]], ids, corte)) &
              set(c.top_k(ens, ids, corte))) / corte
    print(f"\nsolapamiento semilla-1 vs ensamble: {sol:.1%}")

    with r.conectar() as con:
        exp = r.alta_experimento(
            con, NOMBRE, notebook="c107_final.py",
            descripcion="fase 3: base_pesos25 (params z701 + peso BAJA+1 0,25), 20 semillas "
                        "+ ensamble por rank. Gano a base_baja2 por +8,6 M (8/10, p=0,0078) y "
                        "los 31 trials de Optuna no le ganaron (-7,5 M, 5/10).",
            dataset=DATASET, n_columnas=len(pred), meses_train=MESES_TRAIN,
            mes_validacion=202106, target=TARGET, peso_baja1=PESO, params=PARAMS,
            num_boost_round=NUM_BOOST_ROUND, n_semillas=N_SEMILLAS, corte_envios=int(corte),
            ganancia_val_media=ganancia_media, ruta=str(CARPETA))
        con.execute("update experimento set justificacion_corte=%s where id=%s",
                    (f"Corte {corte} por minimax regret sobre los dos folds con 20 semillas. "
                     f"Peor caso {tabla.loc[corte,'peor_caso']:+.2%}. Tabla en regret.parquet.",
                     exp))
    print(f"\nlisto en {time.time()-t0:.0f} s. Archivos en {CARPETA/'envios'}")


if __name__ == "__main__":
    main()
