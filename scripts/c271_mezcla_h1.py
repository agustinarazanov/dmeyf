"""Mezcla por rango, semilla a semilla, del modelo vigente (c241, pesos 0,25, 03-06) con el de horizonte 1
(c270, target BAJA+1, 03-07). Review monday-45, idea A.

Por que semilla a semilla: 5 archivos contra los 5 de c241 (105,51) con la MISMA suerte de semillas, en vez de
un archivo de ensamble contra otro (ruido +-1,5). Peso 0,3 para el miembro de horizonte 1: BAJA+1 y BAJA+2 son
casi indistinguibles por features (AUC 0,62), asi que su ranking es una senal de "se va pronto" con el mes
mas fresco (julio) y 5.103 positivos exactos, sin BAJA+2 escondidos (lo que hundio a c240).

    python scripts/c271_mezcla_h1.py [--peso 0.3] [--cortes 14000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import competencia as c  # noqa: E402
sys.path.insert(0, str(RAIZ / "scripts"))
from c201_receta_lags import semillas_c107  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--peso", type=float, default=0.3)
ap.add_argument("--cortes", type=int, nargs="+", default=[14_000])
a = ap.parse_args()
nombre = f"c271_mezcla_h1_p{int(a.peso * 100):02d}"
carpeta = c.EXPERIMENTOS / nombre; carpeta.mkdir(exist_ok=True)
ref = c.referencia_202108()
ids = None; mezclas = {}
for s in semillas_c107(5):
    a1 = pd.read_parquet(c.EXPERIMENTOS / "c241_rondas2000" / f"scores_202108_s{s}.parquet")
    h1 = pd.read_parquet(c.EXPERIMENTOS / "c270_h1_0307" / f"scores_202108_s{s}.parquet")
    assert (a1[c.fe.ID].to_numpy() == h1[c.fe.ID].to_numpy()).all()
    ids = a1[c.fe.ID].to_numpy()
    r1 = pd.Series(a1["score"].to_numpy()).rank(pct=True).to_numpy()
    r2 = pd.Series(h1["score"].to_numpy()).rank(pct=True).to_numpy()
    mezclas[s] = (1 - a.peso) * r1 + a.peso * r2
    print(f"  semilla {s}: correlacion de rangos c241 vs h1 = {np.corrcoef(r1, r2)[0, 1]:.3f}")
    pd.DataFrame({c.fe.ID: ids, "score": mezclas[s]}).to_parquet(carpeta / f"scores_202108_s{s}.parquet")
validos = set(ids.tolist())
for K in a.cortes:
    E = carpeta / f"envios_{K}"; E.mkdir(exist_ok=True)
    for j, s in enumerate(mezclas):
        sel = c.top_k(mezclas[s], ids, K)
        if j == 0:
            c.verificar_seleccion(sel, ref)
            base = set(c.top_k(pd.read_parquet(c.EXPERIMENTOS / "c241_rondas2000" / f"scores_202108_s{s}.parquet")["score"].to_numpy(), ids, K).tolist())
            print(f"  corte {K:,}: la mezcla cambia {K - len(base & set(sel.tolist())):,} ids respecto de c241 (semilla {s})")
        c.escribir_envios(sel, E / f"{nombre}_{K}_s{s}.csv", validos)
print("  listo")
