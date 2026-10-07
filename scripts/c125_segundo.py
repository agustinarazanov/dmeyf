"""Segundo ensamble INDEPENDIENTE, para ver si el pico en 9.000 es real.

El leaderboard dice que 9.000 envios le gana a 11.000 por 4,57 M. La proyeccion
con la prevalencia estimada de agosto dice lo contrario, y por un margen
parecido. Los dos no pueden tener razon.

Volver a mandar el MISMO archivo no sirve: la particion publica es fija y
devuelve el mismo numero. Lo unico que trae informacion nueva es un RANKING
distinto. Si un ensamble armado con semillas que no se usaron nunca tambien
prefiere 9.000, el pico es real; si prefiere 11.000, era un grupo afortunado de
aciertos en las primeras posiciones de aquel ranking.
"""

import time

import numpy as np
import pandas as pd

import competencia as c

MESES = [202103, 202104, 202105, 202106]
TARGET, PESO, NBR = "pesos", 0.25, 250
CORTES = [9_000, 11_000]
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c125_segundo"
CARPETA.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    # las 20 del ensamble actual, para EXCLUIRLAS
    np.random.seed(c.SEMILLAS[0])
    usadas = set(c.SEMILLAS + np.random.choice(1_000_000, size=15, replace=False).tolist())
    # 20 semillas nuevas, de un sorteo distinto y sin interseccion
    rng = np.random.default_rng(c.SEMILLAS[4])
    nuevas = []
    while len(nuevas) < 20:
        x = int(rng.integers(1, 1_000_000))
        if x not in usadas and x not in nuevas:
            nuevas.append(x)
    print(f"semillas nuevas: {len(nuevas)}, interseccion con las viejas: "
          f"{len(set(nuevas) & usadas)}")

    data = c.cargar("competencia_01.parquet")
    pred = c.columnas_predictoras(data)
    X, y, w = c.preparar(data, MESES, TARGET, PESO)
    X = X[pred]
    fut = data[data[c.fe.MES] == c.MES_COMPETENCIA]
    ids = fut[c.fe.ID].to_numpy()
    validos = set(ids.tolist())

    rangos = []
    for s in nuevas:
        cl = c.clave(PARAMS, MESES, TARGET, PESO, "c125", NBR, s)
        m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"fin_{s}_{cl}")
        rangos.append(pd.Series(m.predict(fut[pred])).rank(pct=True).to_numpy())
        print(f"  {len(rangos)}/20 [{time.time()-t0:.0f}s]", end="\r", flush=True)
    ens = np.mean(rangos, axis=0)
    print()

    viejo = pd.read_parquet(c.EXPERIMENTOS / "c107_pesos25" / "scores_202108.parquet")
    v = viejo.set_index("numero_de_cliente").ensamble
    n = pd.Series(ens, index=ids.astype("int64"))
    comun = v.index.intersection(n.index)
    print(f"Spearman contra el ensamble viejo: {v[comun].corr(n[comun], method='spearman'):.4f}")

    for k in CORTES:
        sel = c.top_k(ens, ids, k)
        c.verificar_seleccion(sel, data)
        ruta = CARPETA / "envios" / f"c125_seg_{k}.csv"
        nn = c.escribir_envios(sel, ruta, validos)
        sol = len(set(sel) & set(c.top_k(v[ids.astype('int64')].to_numpy(), ids, k))) / k
        print(f"  {ruta.name}: {nn:,} envios, solapamiento con el viejo {sol:.1%}")

    pd.DataFrame({"numero_de_cliente": ids, "ensamble": ens}).to_parquet(
        CARPETA / "scores_202108.parquet", index=False)
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
