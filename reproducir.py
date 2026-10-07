"""Reproduce la entrega final desde el dataset original.

La materia lo exige: «scripts/notebooks que partiendo del dataset original
permitan a los profesores replicar exactamente sus entregas finales, esto
incluye a las semillas aleatorias».

    python reproducir.py --salida /tmp/entrega.csv

Parte de `data/competencia_01_crudo.csv` (el CSV tal cual se baja de la catedra,
SIN clase_ternaria) y termina en el CSV que se le manda al bot. No depende de
nada que haya quedado en disco de corridas anteriores: si no encuentra el panel
con la etiqueta, lo reconstruye.

Todo lo que decide el resultado esta aca arriba, explicito.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe  # noqa: E402

CRUDO = RAIZ / "data" / "competencia_01_crudo.csv"
PANEL = RAIZ / "data" / "competencia_01.parquet"

# ---------------------------------------------------------------- la receta
MESES_TRAIN = [202103, 202104, 202105, 202106]
MES_PREDECIR = 202108
TARGET = "pesos"          # positivo = BAJA+1 o BAJA+2
PESO_BAJA1 = 0.25         # BAJA+2 pesa 1,0; BAJA+1 pesa esto
NUM_BOOST_ROUND = 250
CORTE = 10_000            # ver la justificacion abajo
N_SEMILLAS = 20

PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}

# columnas que NO entran al modelo, y por que
FUERA = (
    fe.COLUMNAS_DE_BAJA          # miran el futuro: es_baja ES la respuesta
    + [fe.ID,                    # asa de memorizacion pura
       fe.MES,                   # 202108 es un valor que el modelo nunca vio
       fe.CLASE, "clase_binaria"]
    + ["Visa_Finiciomora",       # ceros inyectados en 202105 y 202108 (el mes a predecir)
       "Master_Finiciomora"]
)

SEMILLAS_PROPIAS = [261431, 269281, 429899, 560771, 749401]


def semillas() -> list[int]:
    np.random.seed(SEMILLAS_PROPIAS[0])      # sin esto cambian en cada corrida
    extra = np.random.choice(1_000_000, size=N_SEMILLAS - len(SEMILLAS_PROPIAS),
                             replace=False).tolist()
    return SEMILLAS_PROPIAS + [int(x) for x in extra]


def panel() -> pd.DataFrame:
    if not PANEL.exists():
        print(f"  {PANEL.name} no existe: reconstruyendo desde {CRUDO.name}")
        con = fe.conectar()
        fe.construir_base(con, str(CRUDO), str(PANEL))
        con.close()
    d = pd.read_parquet(PANEL)
    d[fe.ID] = d[fe.ID].astype("int64")      # viene DOUBLE: 48000000 -> 4.8e+07 rompe el submit
    # DuckDB no garantiza el orden de las filas al reconstruir desde el CSV, y LightGBM
    # depende del orden: sin esto las mismas semillas dan OTRO csv en cada reconstruccion,
    # que es justo lo que este script promete que no pasa.
    return d.sort_values([fe.ID, fe.MES], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=Path("entrega_final.csv"))
    args = ap.parse_args()
    t0 = time.time()

    d = panel()
    pred = [c for c in d.columns if c not in FUERA]
    print(f"  {len(d):,} filas | {len(pred)} columnas predictoras")

    tr = d[d[fe.MES].isin(MESES_TRAIN)]
    clase = tr[fe.CLASE].to_numpy()
    y = np.isin(clase, ["BAJA+1", "BAJA+2"]).astype("int8")
    w = np.where(clase == "BAJA+1", PESO_BAJA1, 1.0)
    X = tr[pred]
    fut = d[d[fe.MES] == MES_PREDECIR]
    ids = fut[fe.ID].to_numpy()
    print(f"  train {MESES_TRAIN} ({len(X):,} filas, {y.sum():,} positivos) "
          f"-> {MES_PREDECIR} ({len(fut):,})")

    rangos = []
    for s in semillas():
        m = lgb.train({**PARAMS, "seed": s, "verbose": -1},
                      lgb.Dataset(X, label=y, weight=w), num_boost_round=NUM_BOOST_ROUND)
        rangos.append(pd.Series(m.predict(fut[pred])).rank(pct=True).to_numpy())
        print(f"    semilla {s} ({len(rangos)}/{N_SEMILLAS})", end="\r", flush=True)
    ens = np.mean(rangos, axis=0)            # ensamble por RANGO, no por score
    print(f"\n  {N_SEMILLAS} modelos entrenados [{time.time()-t0:.0f}s]")

    elegidos = np.sort(ids[np.argsort(ens)[::-1][:CORTE]]).astype("int64")
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.salida, elegidos, fmt="%d")

    texto = args.salida.read_text()
    assert "e+" not in texto and "." not in texto, "notacion cientifica en el CSV"
    assert len(np.loadtxt(args.salida, dtype="int64")) == CORTE
    assert set(elegidos) <= set(ids.tolist())

    # Pertenecer al mes es MEMBRESIA, no CORRESPONDENCIA: si los scores se indexan contra
    # los ids de otra tabla (DuckDB reordena filas al escribir un parquet) salen ids
    # perfectamente validos que el modelo nunca eligio. Eso ya costo un submit con
    # ganancia -49,86. Este chequeo exige que los elegidos esten ENRIQUECIDOS en
    # marcadores de riesgo; una seleccion desalineada no lo esta.
    fut = d[d[fe.MES] == MES_PREDECIR]
    for nombre, cond in {
        "ctrx_quarter == 0": fut["ctrx_quarter"].fillna(0) == 0,
        "mcuentas_saldo < 0": fut["mcuentas_saldo"].fillna(0) < 0,
    }.items():
        marcado = fut[fe.ID].isin(set(elegidos.tolist()))
        p_sel, p_resto = cond[marcado].mean(), cond[~marcado].mean()
        lift = p_sel / p_resto if p_resto else float("inf")
        assert np.isfinite(lift), f"lift no finito en «{nombre}»: los ids no son de {MES_PREDECIR}"
        print(f"  {nombre:22s} {p_sel:6.1%} vs {p_resto:6.1%}   x{lift:.2f}")
    assert lift > 1.5, "la seleccion no se distingue de la poblacion: scores e ids desalineados"

    print(f"\n  {args.salida}: {CORTE:,} envios")
    print(f"  sha256: {hashlib.sha256(args.salida.read_bytes()).hexdigest()}")
    print(f"  listo en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# POR QUE CORTE = 10.000
#
# Se eligio por MINIMAX entre dos escenarios que los datos no permiten separar, no por
# el argmax de ninguna curva.
#
# LO QUE MIDE EL PUBLICO. Seis cortes del MISMO ranking, 20 archivos de una semilla cada
# uno (mismas 20 semillas), sobre la misma particion publica fija:
#
#     corte      8.000   9.000   10.000   11.000   13.000   15.000
#     publico    90,23   94,22    93,24    91,36    87,15    83,18
#
# Convertido a tasa de acierto MARGINAL por banda (un acierto publico vale +1,1 M; hay
# 250 clientes publicos por cada 1.000 de corte; el equilibrio es 2,5%):
#
#     8.000 -> 9.000      ~10,0 aciertos de 250     4,00%   muy por encima del equilibrio
#     9.000 -> 10.000      ~5,4 de 250              2,14%   debajo
#     10.000 -> 11.000     ~4,5 de 250              1,82%   debajo
#
# La tasa cruza el 2,5% al principio de la banda 9-10k: el optimo PUBLICO esta en ~9.200.
# Pero con ~5 aciertos por banda, el 2,14% viene con +-0,9% y el equilibrio cae adentro
# del intervalo. El publico NO distingue 9.000 de 10.000.
#
# LO QUE NO SE PUEDE MEDIR, y es la razon de fondo. La particion publica es FIJA: la
# cantidad de aciertos publicos en una banda del ranking es un numero deterministico, no
# una muestra. Mandar mas archivos o mas puntos de corte devuelve ese mismo numero partido
# en pedazos mas chicos. El +-0,9% no es ruido de medicion reducible: es cuanto representa
# esa banda publica a la privada, y eso no lo baja ningun submit.
#
# EL MINIMAX. Dos escenarios compatibles con lo medido:
#   (a) la cola de agosto es la que mide el publico (~1,85% de 9k a 15k, tipo 202105):
#       cortar en 10.000 en vez de 9.000 cuesta ~5 M al mes.
#   (b) agosto tiene la prevalencia que sugiere el NIVEL publico (142 aciertos en el top
#       9.000 -> ~1.090 BAJA+2, tipo 202106): cortar en 9.000 cuesta ~16 M.
# 10.000 minimiza la peor perdida. 11.000 ya no: la banda 10-11k esta medida bajo el
# equilibrio. 9.000 tampoco: su perdida en el escenario (b) es tres veces mayor.
#
# QUE SE RETRACTA DE LAS VERSIONES ANTERIORES DE ESTE BLOQUE
#
# 1. "La curva es plana, cualquier K entre 8.000 y 11.000 da igual". Medido con 20
#    archivos, la banda 8-9k rinde 4,0% y la 10-11k 1,82%. No es plana.
# 2. "El corte 9.000 vale +2,87 M con 5 sigma". Ese sigma era el error de la media ENTRE
#    SEMILLAS sobre una particion fija. El ruido que corresponde -el sorteo del 25%- es
#    +-3,98 M entre cortes, medido simulando 3.000 particiones. El z real es +0,72.
# 3. "La prevalencia de agosto se estimo en ~1.050-1.100 por dos metodos independientes".
#    El barrido de cortes la contradice. Sigue en disputa y por eso el corte es minimax.
# 4. "No se elige K mirando el publico". Sigue siendo cierto para el ARGMAX de una curva
#    ruidosa. No para la tasa marginal por banda, que es una medicion directa sobre
#    agosto -el unico mes que importa- y que los folds no pueden dar porque miden mayo
#    y junio, cuyas prevalencias son justamente lo que no sabemos.
