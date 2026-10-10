"""Positive-unlabeled sobre 202107: un mes mas cerca de agosto sin envenenar los negativos (A6 del backlog).

c168 agrego 202107 etiquetando BAJA+1 como positivo y TODO el resto como negativo, y perdio 33,7 M:
entre esos "negativos" hay ~870 BAJA+2 escondidos (los que se van en septiembre), que son justo los
clientes que mas se parecen a un positivo. Aca:

- 202103-202106 como siempre (target pesos: BAJA+2 peso 1, BAJA+1 peso 0,25).
- 202107: los 1.103 BAJA+1 entran como positivos (peso --peso-b1-07, por defecto 0,25 como en los
  otros meses); de los demas, se EXCLUYEN los --podar mejor rankeados por el ensamble de c201 (son los
  candidatos a BAJA+2 escondido) y el resto entra como negativo.
- El ranking de 202107 sale de los primeros 5 modelos cacheados de c201 (misma receta y dataset).

Sin validacion local limpia a horizonte 2 que incluya 202107 (no existe); lo mide el publico.

    python scripts/c240_pu_202107.py --podar 2500 --semillas 5
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent / "src"))
import competencia as c  # noqa: E402
from c201_receta_lags import DATASET, NBR, params_receta, semillas_c107, CARPETA as C201  # noqa: E402

NOMBRE = "c240_pu"
CARPETA = c.EXPERIMENTOS / NOMBRE
MESES = [202103, 202104, 202105, 202106]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--podar", type=int, default=2_500)
    ap.add_argument("--peso-b1-07", type=float, default=0.25)
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--cortes", type=int, nargs="+", default=[14_000, 10_000])
    args = ap.parse_args()
    t0 = time.time()
    CARPETA.mkdir(parents=True, exist_ok=True)

    d = c.cargar(DATASET)
    pred = c.columnas_predictoras(d)
    X4, y4, w4 = c.preparar(d, MESES, "pesos", 0.25)
    X4 = X4[pred]

    # --- 202107: ranking con los modelos de c201 y poda de los sospechosos de BAJA+2 ---
    j = d[d[c.fe.MES] == 202107]
    modelos = sorted(glob.glob(str(C201 / "modelo_receta_*.txt")))[:5]
    rk = c.ensamble_por_rank({i: lgb.Booster(model_file=m).predict(j[pred]) for i, m in enumerate(modelos)})
    es_b1 = (j[c.fe.CLASE].to_numpy() == "BAJA+1")
    orden = np.argsort(rk)[::-1]
    sospechoso = np.zeros(len(j), dtype=bool)
    sospechoso[orden[:args.podar]] = True
    podados = sospechoso & ~es_b1
    keep = ~podados
    print(f"  202107: {len(j):,} filas | {int(es_b1.sum()):,} BAJA+1 (positivos, peso {args.peso_b1_07}) | "
          f"top {args.podar:,} del ranking: {int((sospechoso & es_b1).sum()):,} son BAJA+1, {int(podados.sum()):,} se podan | "
          f"{int(keep.sum() - es_b1.sum()):,} negativos quedan [{time.time()-t0:.0f}s]", flush=True)
    X7 = j[pred][keep]
    y7 = es_b1[keep].astype("int8")
    w7 = np.where(es_b1[keep], args.peso_b1_07, 1.0)

    X = pd.concat([X4, X7], ignore_index=True)
    y = np.concatenate([y4, y7])
    w = np.concatenate([w4, w7])
    fut = d[d[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    Xfut = fut[pred]
    del d, X4, X7, j
    P = params_receta(len(X))
    dataset_tag = f"{DATASET}+07pu{args.podar}w{args.peso_b1_07}"
    print(f"  train {len(X):,} filas, {int(y.sum()):,} positivos [{time.time()-t0:.0f}s]", flush=True)

    scores = {}
    for i, s in enumerate(semillas_c107(args.semillas), 1):
        ruta = CARPETA / f"scores_202108_pu{args.podar}_s{s}.parquet"
        if ruta.exists():
            t = pd.read_parquet(ruta)
            assert (t[c.fe.ID].to_numpy() == ids).all()
            scores[s] = t["score"].to_numpy()
        else:
            etq = f"pu{args.podar}_{s}_{c.clave(P, MESES + [202107], 'pesos', 0.25, dataset_tag, NBR, s)}"
            m = c.entrenar_o_cargar(P, X, y, w, NBR, s, CARPETA, etq)
            scores[s] = m.predict(Xfut)
            pd.DataFrame({c.fe.ID: ids, "score": scores[s]}).to_parquet(ruta)
        print(f"  semilla {s} ({i}/{args.semillas}) [{time.time()-t0:.0f}s]", flush=True)

    ens = c.ensamble_por_rank(scores)
    ref = c.cargar("competencia_01.parquet")
    validos = set(ids.tolist())
    for K in args.cortes:
        E = CARPETA / f"envios_{K}"
        E.mkdir(exist_ok=True)
        for k, s in enumerate(scores):
            sel = c.top_k(scores[s], ids, K)
            if k == 0:
                c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{NOMBRE}{args.podar}_{K}_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens, ids, K), E / f"{NOMBRE}{args.podar}_{K}_ens{len(scores)}.csv", validos, referencia=ref)
        print(f"  corte {K:6,}: {len(scores)} archivos + ensamble", flush=True)
    print(f"  listo [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
