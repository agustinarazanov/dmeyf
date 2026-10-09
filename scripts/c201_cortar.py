"""Corta envios desde los scores cacheados de c201, sin reentrenar ni cargar el dataset grande.

    python c201_cortar.py --cortes 10000 11500 12500 [--semillas 5]

Lee experimentos/c201_receta_lags/scores_202108_s<semilla>.parquet (uno por
semilla ya entrenada), arma el ensamble por rank y escribe, por corte, un CSV por
semilla mas uno del ensamble. Sirve para mandar antes de que terminen las 20.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ.parent))   # la raiz del repo: competencia.py y fe_panel.py
import competencia as c  # noqa: E402
import c201_receta_lags as base  # noqa: E402
from c201_receta_lags import semillas_c107  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortes", type=int, nargs="+", default=[10_000, 11_500])
    ap.add_argument("--semillas", type=int, default=20)
    ap.add_argument("--experimento", default=None, help="carpeta en experimentos/ (default: c201_receta_lags)")
    args = ap.parse_args()
    CARPETA, NOMBRE = (c.EXPERIMENTOS / args.experimento, args.experimento) if args.experimento else (base.CARPETA, base.NOMBRE)

    scores, ids = {}, None
    for s in semillas_c107(args.semillas):
        ruta = CARPETA / f"scores_202108_s{s}.parquet"
        if not ruta.exists():
            continue
        t = pd.read_parquet(ruta)
        if ids is None:
            ids = t[c.fe.ID].to_numpy()
        assert (t[c.fe.ID].to_numpy() == ids).all(), "los ids no coinciden entre semillas"
        scores[s] = t["score"].to_numpy()
    if not scores:
        sys.exit("no hay scores cacheados todavia")
    print(f"  {len(scores)} semillas disponibles: {list(scores)}")

    ens = c.ensamble_por_rank(scores)
    pd.DataFrame({c.fe.ID: ids, "ensamble": ens}).to_parquet(
        CARPETA / f"scores_202108_ens{len(scores)}.parquet")
    ref = c.cargar("competencia_01.parquet")
    validos = set(ids.tolist())
    for K in args.cortes:
        E = CARPETA / f"envios_{K}"
        E.mkdir(exist_ok=True)
        for j, s in enumerate(scores):
            sel = c.top_k(scores[s], ids, K)
            if j == 0:
                c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{NOMBRE}_{K}_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens, ids, K), E / f"{NOMBRE}_{K}_ens{len(scores)}.csv",
                          validos, referencia=ref)
        print(f"  corte {K:6,}: {len(scores)} archivos de semilla + ensamble de {len(scores)}")


if __name__ == "__main__":
    main()
