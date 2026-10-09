"""Una variante de la entrega, parametrizada: dataset x hiperparametros x target x meses.

Reemplaza escribir un script por experimento cuando lo que cambia es una celda de la grilla.
Cada corrida cachea modelos y scores de 202108 por semilla en experimentos/<nombre>/ y escribe
envios por corte (archivos de semilla + ensamble). Las semillas son las de c107, en orden.

    python scripts/c210_variante.py --nombre c211_lags_z701   --dataset competencia_01_lags12.parquet --params z701
    python scripts/c210_variante.py --nombre c212_base_receta --dataset competencia_01.parquet        --params receta
    python scripts/c210_variante.py --nombre c213_lags_baja2  --dataset competencia_01_lags12.parquet --params receta --target baja2
    python scripts/c210_variante.py --nombre c214_lags_sin03  --dataset competencia_01_lags12.parquet --params receta --meses 202104 202105 202106

--params z701   : los de la entrega vieja (45 hojas, lr 0,0077, mdl 174, ff 0,277, bagging 0,918; 250 rondas)
--params receta : la de Denicolay (mdl 0, lr 0,005, ff 0,5, bynode 0,2, 83 hojas; 1.000 rondas)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))
import competencia as c  # noqa: E402
from c201_receta_lags import params_receta, semillas_c107  # noqa: E402

PARAMS_Z701 = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
    "num_threads": 8, "force_col_wise": True, "deterministic": True,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nombre", required=True)
    ap.add_argument("--dataset", default="competencia_01_lags12.parquet")
    ap.add_argument("--params", choices=["z701", "receta"], default="receta")
    ap.add_argument("--target", choices=["baja2", "baja12", "pesos"], default="pesos")
    ap.add_argument("--peso-baja1", type=float, default=0.25)
    ap.add_argument("--meses", type=int, nargs="+", default=[202103, 202104, 202105, 202106])
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--cortes", type=int, nargs="+", default=[10_000, 14_000])
    ap.add_argument("--rondas", type=int, default=None, help="num_boost_round; por defecto 1.000 (receta) o 250 (z701)")
    ap.add_argument("--hojas", type=int, default=None, help="num_leaves (receta: 83)")
    ap.add_argument("--columnas", type=Path, default=None, help="archivo con una columna por linea: se entrena SOLO con esas predictoras (reduccion de dimensionalidad por importancia)")
    ap.add_argument("--ff", type=float, default=None, help="feature_fraction (receta: 0,5)")
    ap.add_argument("--hessian", type=float, default=None, help="factor sobre min_sum_hessian_in_leaf de la receta (1,0 = 12,79 x filas / 326.184)")
    args = ap.parse_args()
    t0 = time.time()
    carpeta = c.EXPERIMENTOS / args.nombre
    carpeta.mkdir(parents=True, exist_ok=True)

    d = c.cargar(args.dataset)
    pred = c.columnas_predictoras(d)
    if args.columnas:
        quedan = set(args.columnas.read_text().split())
        faltan = quedan - set(pred)
        assert not faltan, f"{len(faltan)} columnas de {args.columnas} no estan en el dataset: {sorted(faltan)[:5]}"
        pred = [x for x in pred if x in quedan]
    X, y, w = c.preparar(d, args.meses, args.target, args.peso_baja1)
    X = X[pred]
    fut = d[d[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    Xfut = fut[pred]
    del d
    if args.params == "receta":
        P, nbr = params_receta(len(X)), 1_000
    else:
        P, nbr = PARAMS_Z701, 250
    if args.rondas:
        nbr = args.rondas
    if args.hojas:
        P = {**P, "num_leaves": args.hojas}
    if args.ff:
        P = {**P, "feature_fraction": args.ff}
    if args.hessian:
        P = {**P, "min_sum_hessian_in_leaf": P["min_sum_hessian_in_leaf"] * args.hessian}
    print(f"  {args.nombre}: {args.dataset} | {len(pred)} predictoras | {args.params} | "
          f"{args.target} {args.peso_baja1 if args.target == 'pesos' else ''} | meses {args.meses} | "
          f"{len(X):,} filas, {int(y.sum()):,} positivos | {nbr} rondas, {P['num_leaves']} hojas, hessian {P['min_sum_hessian_in_leaf']:.2f} [{time.time()-t0:.0f}s]", flush=True)

    scores = {}
    for i, s in enumerate(semillas_c107(args.semillas), 1):
        ruta = carpeta / f"scores_202108_s{s}.parquet"
        if ruta.exists():
            t = pd.read_parquet(ruta)
            assert (t[c.fe.ID].to_numpy() == ids).all(), f"{ruta.name}: ids distintos del dataset actual"
            scores[s] = t["score"].to_numpy()
        else:
            etq = f"{args.params}_{s}_{c.clave(P, args.meses, args.target, args.peso_baja1, args.dataset + (f"|{len(pred)}cols" if args.columnas else ""), nbr, s)}"
            m = c.entrenar_o_cargar(P, X, y, w, nbr, s, carpeta, etq)
            scores[s] = m.predict(Xfut)
            pd.DataFrame({c.fe.ID: ids, "score": scores[s]}).to_parquet(ruta)
        print(f"  semilla {s} ({i}/{args.semillas}) [{time.time()-t0:.0f}s]", flush=True)

    ens = c.ensamble_por_rank(scores)
    pd.DataFrame({c.fe.ID: ids, "ensamble": ens}).to_parquet(carpeta / f"scores_202108_ens{len(scores)}.parquet")
    ref = c.referencia_202108()
    validos = set(ids.tolist())
    for K in args.cortes:
        E = carpeta / f"envios_{K}"
        E.mkdir(exist_ok=True)
        for j, s in enumerate(scores):
            sel = c.top_k(scores[s], ids, K)
            if j == 0:
                c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{args.nombre}_{K}_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens, ids, K), E / f"{args.nombre}_{K}_ens{len(scores)}.csv", validos, referencia=ref)
        print(f"  corte {K:6,}: {len(scores)} archivos + ensamble", flush=True)
    print(f"  listo [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
