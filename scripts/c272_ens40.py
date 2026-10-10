"""Candidato de entrega: ensamble por rango de 40 modelos = 20 semillas a 1.000 rondas (c201) + 20 a 2.000 (c241).

Por que: la evidencia entre 1.000 y 2.000 rondas es +0,85 +- 0,6 (20 archivos contra 20) y -0,4 +- 1,5
(ensamble contra ensamble): indistinguibles. Promediar los dos es la cobertura, y es literalmente lo que pide
la catedra ("10+ modelos con hiperparametros distintos, promediando"). No se decide por el publico de un archivo.

Lee los scores cacheados (no entrena), escribe el envio a 14.000 y su SHA-256.
"""
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import competencia as c  # noqa: E402

CORTE = 14_000
carpeta = c.EXPERIMENTOS / "c272_ens40"; carpeta.mkdir(exist_ok=True)
scores, ids = {}, None
for exp in ("c201_receta_lags", "c241_rondas2000"):
    for f in sorted(glob.glob(str(c.EXPERIMENTOS / exp / "scores_202108_s*.parquet"))):
        t = pd.read_parquet(f)
        if ids is None:
            ids = t[c.fe.ID].to_numpy()
        assert (t[c.fe.ID].to_numpy() == ids).all()
        scores[f"{exp}:{Path(f).stem.rsplit('_s', 1)[1]}"] = t["score"].to_numpy()
assert len(scores) == 40, len(scores)
ens = c.ensamble_por_rank(scores)
pd.DataFrame({c.fe.ID: ids, "ensamble": ens}).to_parquet(carpeta / "scores_202108_ens40.parquet")
E = carpeta / f"envios_{CORTE}"; E.mkdir(exist_ok=True)
ruta = E / f"c272_ens40_{CORTE}.csv"
sel = c.top_k(ens, ids, CORTE)
c.escribir_envios(sel, ruta, set(ids.tolist()), referencia=c.referencia_202108())
sha = hashlib.sha256(ruta.read_bytes()).hexdigest()
# solapamiento con los dos ensambles de 20
for exp, nombre in (("c201_receta_lags", "c201_receta_lags_14000_ens.csv"), ("c241_rondas2000", "c241_rondas2000_14000_ens20.csv")):
    otro = set(np.loadtxt(c.EXPERIMENTOS / exp / "envios_14000" / nombre, dtype="int64").tolist())
    print(f"  solapamiento con {exp} ens20: {len(otro & set(sel.tolist())):,} de {CORTE:,}")
print(f"  {ruta}\n  sha256: {sha}")
