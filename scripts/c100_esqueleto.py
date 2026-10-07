"""Fase 0: el camino completo de punta a punta, con parametros conocidos.

No busca nada. Entrena los hiperparametros que z701 ya tenia afinados, scorea
202108 y escribe los CSV de entrega. Lo que valida es el CAMINO, que es lo unico
que la validacion local no puede chequear: si el bot acepta el archivo.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import competencia as c
import registro as r

NOMBRE = "c100_esqueleto"
DATASET = "competencia_01.parquet"
MESES_TRAIN = [202103, 202104, 202105, 202106]
MES_VALIDACION = 202106
# Los dos folds walk-forward posibles con max(train) <= validacion - 2.
# Hacen falta LOS DOS: el K optimo lo manda la prevalencia del mes de
# validacion (202105 tiene 870 BAJA+2 y quiere 8.092 envios; 202106 tiene
# 1.098 y quiere 14.788, con el MISMO modelo), y la de 202108 no se conoce.
FOLDS = [([202103], 202105), ([202103, 202104], 202106)]
TARGET = "baja2"
NUM_BOOST_ROUND = 250
N_SEMILLAS_ENVIO = 3

PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "first_metric_only": True,
    "boost_from_average": True,
    "feature_pre_filter": False,
    "max_bin": 31,
    "num_leaves": 45,
    "learning_rate": 0.0077,
    "min_data_in_leaf": 174,
    "feature_fraction": 0.277,
    "bagging_fraction": 0.918,
    "bagging_freq": 1,
}

CARPETA = c.EXPERIMENTOS / NOMBRE
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    data = c.cargar(DATASET)
    predictoras = c.columnas_predictoras(data)
    print(f"{len(data):,} filas, {len(predictoras)} columnas predictoras")
    for prohibida in ("es_baja", "numero_de_cliente", "foto_mes", "Visa_Finiciomora"):
        assert prohibida not in predictoras, prohibida

    # ---- validacion out of time: los dos folds ----
    escenarios, scores_val = {}, {}
    for meses, mes_val in FOLDS:
        Xv, yv, wv = c.preparar(data, meses, TARGET)
        val = data[data[c.fe.MES] == mes_val]
        es_baja2 = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        etq = f"f{mes_val}"
        acums = []
        for semilla in c.SEMILLAS:
            cl = c.clave(PARAMS, meses, TARGET, 0.0, DATASET, NUM_BOOST_ROUND, semilla)
            m = c.entrenar_o_cargar(PARAMS, Xv, yv, wv, NUM_BOOST_ROUND, semilla,
                                    CARPETA, f"{etq}_{semilla}_{cl}")
            s = m.predict(val[predictoras])
            acums.append(c.ganancia_acumulada(s, es_baja2))
            scores_val.setdefault(mes_val, {})[semilla] = (s, es_baja2)
        curva = np.mean(acums, axis=0)
        clave_esc = f"{mes_val} ({es_baja2.sum()} B+2)"
        escenarios[clave_esc] = curva
        print(f"fold train {meses} -> {mes_val}: {es_baja2.sum()} BAJA+2, "
              f"K*={int(np.argmax(curva)) + 1:,}, max={curva.max():,.0f}")

    corte, tabla = c.corte_minimax(escenarios)
    tabla.to_parquet(CARPETA / "regret.parquet")
    print("\nregret (%) por cortar en un K fijo:")
    print((tabla * 100).round(2).to_string())
    print(f"\ncorte elegido por minimax regret: {corte:,} "
          f"(peor caso {tabla.loc[corte, 'peor_caso']:.2%}, "
          f"esperado {tabla.loc[corte, 'esperado']:.2%})")
    print(f"  el optimo de 202106 solo seria {int(np.argmax(escenarios[list(escenarios)[1]])) + 1:,}, "
          f"con peor caso {tabla['peor_caso'].min():.2%}: perseguir el pico de un mes es la trampa")

    # ---- cuanto ruido mete el sorteo publico/privado del 25% ----
    s0, es0 = scores_val[MES_VALIDACION][c.SEMILLAS[0]]
    pub = c.simular_publico_privado(s0, es0, corte)
    print(f"\npublico 25% sobre {MES_VALIDACION}: media {pub.mean() / 1e6:,.1f} M, "
          f"sd {pub.std() / 1e6:,.1f} M ({pub.std() / pub.mean():.1%})")
    print("  -> diferencias menores a eso entre dos submits son el sorteo, no el modelo")
    pub.to_frame().to_parquet(CARPETA / "publico_privado.parquet", index=False)

    ganancias_corte = [curva[corte - 1] for curva in escenarios.values()]
    ganancia_media = float(np.mean(ganancias_corte))
    ganancia_sd = float(np.std(ganancias_corte))

    # ---- modelo final: todos los meses etiquetados -> 202108 ----
    Xt, yt, wt = c.preparar(data, MESES_TRAIN, TARGET)
    fut = data[data[c.fe.MES] == c.MES_COMPETENCIA]
    Xfut, ids = fut[predictoras], fut[c.fe.ID].to_numpy()
    validos = set(ids.tolist())
    print(f"\nfinal: train {MESES_TRAIN} ({len(Xt):,}) -> {c.MES_COMPETENCIA} ({len(Xfut):,})")

    scores_fut, archivos = {}, []
    for semilla in c.SEMILLAS[:N_SEMILLAS_ENVIO]:
        clave = c.clave(PARAMS, MESES_TRAIN, TARGET, 0.0, DATASET, NUM_BOOST_ROUND, semilla)
        m = c.entrenar_o_cargar(PARAMS, Xt, yt, wt, NUM_BOOST_ROUND, semilla,
                                CARPETA, f"fin_{semilla}_{clave}")
        s = m.predict(Xfut)
        scores_fut[semilla] = s
        nombre = f"{NOMBRE}_s{semilla}.csv"
        ruta = CARPETA / "envios" / nombre
        n = c.escribir_envios(c.top_k(s, ids, corte), ruta, validos)
        archivos.append({"nombre": nombre, "ruta": ruta, "semilla": semilla, "n_envios": n})
        print(f"  semilla {semilla}: {nombre}, {n:,} envios")

    pd.DataFrame({"numero_de_cliente": ids, **{f"s{k}": v for k, v in scores_fut.items()}}) \
        .to_parquet(CARPETA / f"scores_{c.MES_COMPETENCIA}.parquet", index=False)
    (CARPETA / "params.json").write_text(json.dumps(
        {"params": PARAMS, "meses_train": MESES_TRAIN, "target": TARGET,
         "num_boost_round": NUM_BOOST_ROUND, "corte_envios": corte,
         "dataset": DATASET}, indent=2))

    coincidencia = len(set(c.top_k(scores_fut[c.SEMILLAS[0]], ids, corte)) &
                       set(c.top_k(scores_fut[c.SEMILLAS[1]], ids, corte))) / corte
    print(f"\nsolapamiento entre dos semillas: {coincidencia:.1%}")

    # ---- registro ----
    with r.conectar() as con:
        r.crear_esquema(con)
        exp_id = r.alta_experimento(
            con, NOMBRE, notebook="c100_esqueleto.py",
            descripcion="esqueleto fase 0: params de z701, sin busqueda",
            dataset=DATASET, n_columnas=len(predictoras),
            meses_train=MESES_TRAIN, mes_validacion=MES_VALIDACION,
            target=TARGET, params=PARAMS, num_boost_round=NUM_BOOST_ROUND,
            n_semillas=N_SEMILLAS_ENVIO, corte_envios=corte,
            ganancia_val_media=ganancia_media, ganancia_val_sd=ganancia_sd,
            ruta=str(CARPETA))
        existe = con.execute("select id from submit where nombre=%s",
                             (NOMBRE,)).fetchone()
        if existe is None:
            r.alta_submit(
                con, NOMBRE, exp_id, archivos,
                hipotesis="validar que el bot acepta el formato del archivo y "
                          "capturar el texto literal de su respuesta",
                delta_contra=None,
                delta_descripcion="primer submit: linea de base con los "
                                  "hiperparametros de z701, sin busqueda")
        print(r.listar(con).to_string(index=False))

    print(f"\nlisto en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
