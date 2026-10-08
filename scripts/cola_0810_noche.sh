#!/bin/zsh
# despues del PU: la receta con 2.000 rondas (los best_iter de la busqueda escalaban con los meses)
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -qE "listo|Traceback" experimentos/c240_pu.log; do sleep 60; done
echo "== $(date '+%H:%M') c241: receta con 2.000 rondas"
python - <<'PY'
import sys; sys.argv = ["c210", "--nombre", "c241_rondas2000", "--dataset", "competencia_01_lags12.parquet", "--params", "receta", "--semillas", "5", "--cortes", "14000", "10000"]
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import c210_variante as v
# misma receta, 2.000 rondas en vez de 1.000
orig = v.main
import competencia as c
from c201_receta_lags import params_receta
def main2():
    import argparse, time, numpy as np, pandas as pd
    args = argparse.Namespace(nombre="c241_rondas2000", dataset="competencia_01_lags12.parquet", params="receta",
                              target="pesos", peso_baja1=0.25, meses=[202103,202104,202105,202106], semillas=5, cortes=[14000,10000])
    t0 = time.time(); carpeta = c.EXPERIMENTOS / args.nombre; carpeta.mkdir(parents=True, exist_ok=True)
    d = c.cargar(args.dataset); pred = c.columnas_predictoras(d)
    X, y, w = c.preparar(d, args.meses, args.target, args.peso_baja1); X = X[pred]
    fut = d[d[c.fe.MES] == c.MES_COMPETENCIA]; ids = fut[c.fe.ID].to_numpy(); Xfut = fut[pred]; del d
    P, nbr = params_receta(len(X)), 2000
    scores = {}
    for i, s in enumerate(v.semillas_c107(args.semillas), 1):
        ruta = carpeta / f"scores_202108_s{s}.parquet"
        if ruta.exists():
            t = pd.read_parquet(ruta); assert (t[c.fe.ID].to_numpy() == ids).all(); scores[s] = t["score"].to_numpy()
        else:
            etq = f"receta2000_{s}_{c.clave(P, args.meses, args.target, args.peso_baja1, args.dataset, nbr, s)}"
            m = c.entrenar_o_cargar(P, X, y, w, nbr, s, carpeta, etq); scores[s] = m.predict(Xfut)
            pd.DataFrame({c.fe.ID: ids, "score": scores[s]}).to_parquet(ruta)
        print(f"  semilla {s} ({i}/{args.semillas}) [{time.time()-t0:.0f}s]", flush=True)
    ens = c.ensamble_por_rank(scores); ref = c.cargar("competencia_01.parquet"); validos = set(ids.tolist())
    for K in args.cortes:
        E = carpeta / f"envios_{K}"; E.mkdir(exist_ok=True)
        for j, s in enumerate(scores):
            sel = c.top_k(scores[s], ids, K)
            if j == 0: c.verificar_seleccion(sel, ref)
            c.escribir_envios(sel, E / f"{args.nombre}_{K}_s{s}.csv", validos)
        c.escribir_envios(c.top_k(ens, ids, K), E / f"{args.nombre}_{K}_ens{len(scores)}.csv", validos, referencia=ref)
        print(f"  corte {K:6,}: {len(scores)} archivos + ensamble", flush=True)
    print(f"  listo [{time.time()-t0:.0f}s]", flush=True)
main2()
PY
echo "== $(date '+%H:%M') fin"
