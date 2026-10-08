"""El corte calculado, no medido a submits: prevalencia estimada + captura en el fold C.

Dos instrumentos locales para el corte del modelo nuevo (dataset lags12 + receta Denicolay):

1. PREVALENCIA DE AGOSTO. Un modelo con target BAJA+2 puro (objective binary, sin pesos)
   da probabilidades; la suma de p sobre 202108 estima cuantos BAJA+2 hay en agosto.
   Control de calibracion: la misma suma sobre 202106 con un modelo entrenado en [03,04]
   (fold B), contra los 1.098 reales; y sobre 202107 con [03,04,05] contra los 1.103 BAJA+1.
2. CURVA DE CAPTURA EN EL FOLD C ([03,04,05] -> 202107, target BAJA+1), el unico fold donde
   la historia esta poblada en el entrenamiento. Captura(K) = fraccion de positivos dentro
   del top K. Es independiente de la prevalencia; combinada con los escenarios de prevalencia
   (870 / 960 / 1.098 / 1.139 observados + la estimada en 1) da la ganancia esperada por K y el
   arrepentimiento de cada corte (metodo de c172).

Salida: experimentos/c203_corte_calculado/{prevalencia.json, captura_foldC.parquet, regret.csv}
y la tabla impresa. Semillas: las primeras N de c107. Modelos cacheados por semilla.

    python scripts/c203_corte_calculado.py --semillas 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))
import competencia as c  # noqa: E402
from c201_receta_lags import DATASET, NBR, params_receta, semillas_c107  # noqa: E402

NOMBRE = "c203_corte_calculado"
CARPETA = c.EXPERIMENTOS / NOMBRE
KS = list(range(8_000, 16_001, 500))
PREVALENCIAS_OBSERVADAS = [870, 960, 1_098, 1_139]


def entrenar(d, meses, target, s, etiqueta):
    X, y, w = c.preparar(d, meses, target, 0.25 if target == "pesos" else 0.0)
    pred = c.columnas_predictoras(d)
    P = params_receta(len(X))
    return c.entrenar_o_cargar(P, X[pred], y, w, NBR, s, CARPETA, etiqueta), pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)
    d = c.cargar(DATASET)
    sem = semillas_c107(args.semillas)

    # ---- 1. prevalencia estimada: modelos BAJA+2 puro ---------------------------------
    prev = {}
    for nombre, meses, mes_val, real in [
        ("fold_B_202106", [202103, 202104], 202106, 1_098),
        ("fold_C_202107", [202103, 202104, 202105], 202107, None),   # solo BAJA+1 conocidos
        ("agosto", [202103, 202104, 202105, 202106], 202108, None),
    ]:
        sumas = []
        for s in sem:
            m, pred = entrenar(d, meses, "baja2", s, f"baja2_{nombre}_{s}")
            fut = d[d[c.fe.MES] == mes_val]
            sumas.append(float(m.predict(fut[pred]).sum()))
        prev[nombre] = {"suma_p_media": float(np.mean(sumas)), "suma_p_sd": float(np.std(sumas)),
                        "real": real}
        print(f"  {nombre:16s} suma p = {np.mean(sumas):8.1f} ± {np.std(sumas):5.1f}   real {real}   [{time.time()-t0:.0f}s]")
    (CARPETA / "prevalencia.json").write_text(json.dumps(prev, indent=2))

    # ---- 2. captura en el fold C con la configuracion de la entrega (pesos 0,25) --------
    meses, mes_val = [202103, 202104, 202105], 202107
    val = d[d[c.fe.MES] == mes_val]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+1")       # en 202107 el positivo visible es BAJA+1
    scores = {}
    for s in sem:
        m, pred = entrenar(d, meses, "pesos", s, f"pesos_foldC_{s}")
        scores[s] = m.predict(val[pred])
    ens = c.ensamble_por_rank(scores)
    orden = np.argsort(ens)[::-1]
    acum = np.cumsum(es[orden])
    captura = pd.DataFrame({"K": KS, "captura": [acum[k - 1] / es.sum() for k in KS]})
    captura.to_parquet(CARPETA / "captura_foldC.parquet")
    print("\n  captura fold C (ensamble):")
    print(captura.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 3. ganancia esperada por K bajo cada prevalencia, y arrepentimiento ------------
    escenarios = PREVALENCIAS_OBSERVADAS + [round(prev["agosto"]["suma_p_media"])]
    filas = []
    for n_pos in escenarios:
        g = {k: cap * n_pos * c.GANANCIA_ACIERTO - (k - cap * n_pos) * c.COSTO_ESTIMULO
             for k, cap in zip(captura.K, captura.captura)}
        mejor = max(g.values())
        for k, v in g.items():
            filas.append({"prevalencia": n_pos, "K": k, "ganancia_M": v / 1e6,
                          "regret_pct": 100 * (mejor - v) / mejor})
    reg = pd.DataFrame(filas)
    tabla = reg.pivot(index="K", columns="prevalencia", values="regret_pct")
    tabla["peor"] = tabla.max(axis=1)
    tabla["medio"] = tabla.drop(columns="peor").mean(axis=1)
    tabla.to_csv(CARPETA / "regret.csv")
    print("\n  arrepentimiento % por K y prevalencia (columnas), peor caso y medio:")
    print(tabla.round(2).to_string())
    print(f"\n  minimax: K = {tabla['peor'].idxmin():,}   minimo esperado: K = {tabla['medio'].idxmin():,}")
    print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
