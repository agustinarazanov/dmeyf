"""Verifica que el pipeline de reproducir.py reproduce EXACTAMENTE un modelo registrado, semilla por semilla.

Entrena la semilla pedida con el codigo de reproducir.py (datos(), params(), Dataset nuevo) y compara las
predicciones sobre 202108 con experimentos/c241_rondas2000/scores_202108_s<semilla>.parquet. 10-oct: la corrida
de anoche difirio en 157 ids; la hipotesis es el Dataset compartido entre semillas (bins muestreados con la semilla).

    python scripts/c280_verificar_semilla.py --semilla 269281 --rondas 2000
"""
import argparse, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import reproducir as r  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--semilla", type=int, default=269281)
ap.add_argument("--rondas", type=int, default=2000)
ap.add_argument("--experimento", default="c241_rondas2000")
a = ap.parse_args(); t0 = time.time()
d = r.datos(); pred = [x for x in d.columns if x not in r.FUERA]
tr = d[d[r.fe.MES].isin(r.MESES_TRAIN)]; clase = tr[r.fe.CLASE].to_numpy()
y = np.isin(clase, ["BAJA+1", "BAJA+2"]).astype("int8"); w = np.where(clase == "BAJA+1", r.PESO_BAJA1, 1.0)
fut = d[d[r.fe.MES] == r.MES_PREDECIR]
m = lgb.train({**r.params(len(tr)), "seed": a.semilla}, lgb.Dataset(tr[pred], label=y, weight=w), num_boost_round=a.rondas)
p = m.predict(fut[pred])
reg = pd.read_parquet(RAIZ / "experimentos" / a.experimento / f"scores_202108_s{a.semilla}.parquet")
assert (reg[r.fe.ID].to_numpy() == fut[r.fe.ID].to_numpy()).all()
q = reg["score"].to_numpy()
top = lambda v: set(fut[r.fe.ID].to_numpy()[np.argsort(-v, kind="stable")[:r.CORTE]])
print(f"  semilla {a.semilla}, {a.rondas} rondas [{time.time()-t0:.0f}s]")
print(f"  max |p - registrado| = {np.abs(p - q).max():.3e}   identicos: {np.array_equal(p, q)}")
print(f"  top {r.CORTE:,}: {len(top(p) ^ top(q)) // 2} ids distintos")
