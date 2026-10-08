"""Captura a HORIZONTE 2 del modelo nuevo: fold B ([03,04] -> 202106, BAJA+2).

c203 midio la captura en el fold C, que valida BAJA+1 a un mes: tarea mas facil, curva mas
concentrada, K* mas chico (9.500-10.500). El publico, que mide agosto a dos meses, dice
13.500-14.500. Este script mide la curva a dos meses con el unico fold que lo permite; tiene
lag1 en abril y nada en marzo, asi que subestima al modelo, pero la FORMA de la curva en la cola
es lo que decide si el K* a horizonte 2 esta a la derecha.

    python scripts/c216_captura_foldB.py --semillas 3
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ.parent))
import competencia as c
from c201_receta_lags import DATASET, NBR, params_receta, semillas_c107
NOMBRE = "c216_captura_foldB"; CARPETA = c.EXPERIMENTOS / NOMBRE
KS = list(range(8_000, 16_001, 500)); PREV = [870, 960, 1_098, 1_139]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--semillas", type=int, default=3); a = ap.parse_args()
    t0 = time.time(); CARPETA.mkdir(parents=True, exist_ok=True)
    d = c.cargar(DATASET); pred = c.columnas_predictoras(d)
    meses, mes_val = [202103, 202104], 202106
    X, y, w = c.preparar(d, meses, "pesos", 0.25); X = X[pred]; P = params_receta(len(X))
    val = d[d[c.fe.MES] == mes_val]; es = (val[c.fe.CLASE].to_numpy() == "BAJA+2"); Xv = val[pred]; del d
    scores = {}
    for s in semillas_c107(a.semillas):
        etq = f"pesos_foldB_{s}_{c.clave(P, meses, 'pesos', 0.25, DATASET, NBR, s)}"
        scores[s] = c.entrenar_o_cargar(P, X, y, w, NBR, s, CARPETA, etq).predict(Xv)
        print(f"  semilla {s} [{time.time()-t0:.0f}s]", flush=True)
    ens = c.ensamble_por_rank(scores); orden = np.argsort(ens)[::-1]; acum = np.cumsum(es[orden])
    cap = pd.DataFrame({"K": KS, "captura": [acum[k-1]/es.sum() for k in KS]})
    cap.to_parquet(CARPETA / "captura_foldB.parquet")
    g_real = c.ganancia_acumulada(ens, es)
    cap["ganancia_real_M"] = [g_real[k-1]/1e6 for k in KS]
    print("\n  captura fold B (horizonte 2, 1.098 positivos) y ganancia real en 202106:")
    print(cap.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    filas = []
    for n_pos in PREV:
        g = {k: cp*n_pos*c.GANANCIA_ACIERTO - (k-cp*n_pos)*c.COSTO_ESTIMULO for k, cp in zip(cap.K, cap.captura)}
        mejor = max(g.values())
        filas += [{"prevalencia": n_pos, "K": k, "regret_pct": 100*(mejor-v)/mejor} for k, v in g.items()]
    t = pd.DataFrame(filas).pivot(index="K", columns="prevalencia", values="regret_pct")
    t["peor"] = t.max(axis=1); t["medio"] = t.drop(columns="peor").mean(axis=1); t.to_csv(CARPETA / "regret.csv")
    print("\n  arrepentimiento % por K:"); print(t.round(2).to_string())
    print(f"\n  minimax: K = {t['peor'].idxmin():,}   minimo esperado: K = {t['medio'].idxmin():,}   [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
