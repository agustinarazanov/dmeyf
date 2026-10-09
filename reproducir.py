"""Reproduce la entrega final desde el dataset original.

La materia lo exige: «scripts/notebooks que partiendo del dataset original permitan a los
profesores replicar exactamente sus entregas finales, esto incluye a las semillas aleatorias».

    python reproducir.py --rondas 2000 --salida entrega_final.csv        # ensamble de 20 a 2.000 rondas
    python reproducir.py --rondas 1000 2000 --salida entrega_final.csv   # ensamble de 40 (20 a 1.000 + 20 a 2.000)

Parte de `data/competencia_01_crudo.csv` (el CSV tal cual se baja de la catedra, SIN clase_ternaria),
construye la clase, arma el dataset con historia, entrena las 20 semillas y escribe el CSV que se le
manda al bot. Imprime el SHA-256 para compararlo con el entregado. Tarda ~2 horas en 8 cores.

LOS TRES CANDIDATOS (9-oct-2026), misma receta, mismos datos, mismas 20 semillas; cambia solo num_boost_round:
  --rondas 1000       -> `c201_lags_ens20_14000`      104,17 publico   sha 5157ac24...
  --rondas 2000       -> `c241_rondas2000_ens20_14000` 103,79 publico   sha 657d747a...
  --rondas 1000 2000  -> `c272_ens40_14000`            103,02 publico   sha 7ace9c16...
El SHA esperado se elige segun --rondas. LA ENTREGA FINAL se fija en SHA_FINAL cuando se decida.

LA RECETA COMUN:
  - variables: las 150 crudas + lag 1, delta 1, lag 2 y delta 2 de cada una (750 predictoras),
    con `ccajas_depositos` en NULL en 202105 (es todo cero ese mes)
  - fuera: id, foto_mes, clase, las columnas de baja, Visa_Finiciomora y Master_Finiciomora
  - clase: positivo = BAJA+1 o BAJA+2, con BAJA+1 pesando 0,25
  - entrena 202103..202106, 20 semillas por cada valor de --rondas, receta de hiperparametros de Denicolay
  - ensamble por RANGO de todas las predicciones sobre 202108; se envian los 14.000 mejores

Todo lo que decide el resultado esta aca arriba, explicito. Nada depende de modelos guardados.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe  # noqa: E402
from competencia import verificar_seleccion  # noqa: E402

CRUDO = RAIZ / "data" / "competencia_01_crudo.csv"
PANEL = RAIZ / "data" / "competencia_01.parquet"
LAGS = RAIZ / "data" / "competencia_01_lags12.parquet"
SHA_POR_RONDAS = {
    (1000,): "5157ac244673de5f50a4531373b4164fd58d5d9905edc5fd7f99e0bfe64b4ae9",       # c201_lags_ens20_14000
    (2000,): "657d747addc61efad64145ec9cef550c29c50c5ce283745f176d243270bb9d2e",       # c241_rondas2000_ens20_14000
    (1000, 2000): "7ace9c160cbc47ab047af41da18ee1774c2d3c71626cc3ba599c140d37e1643d",  # c272_ens40_14000
}
SHA_FINAL = None   # se fija al decidir la entrega

# ---------------------------------------------------------------- la receta
MESES_TRAIN = [202103, 202104, 202105, 202106]
MES_PREDECIR = 202108
PESO_BAJA1 = 0.25         # BAJA+2 pesa 1,0; BAJA+1 pesa esto
RONDAS_DEFAULT = [2000]       # --rondas lo cambia; ver SHA_POR_RONDAS
CORTE = 14_000            # publico: meseta 13.500-14.500; fold B a horizonte 2: minimax 13.500, esperado 14.000
N_SEMILLAS = 20
SEMILLAS_PROPIAS = [261431, 269281, 429899, 560771, 749401]

FUERA = (
    fe.COLUMNAS_DE_BAJA          # miran el futuro: es_baja ES la respuesta
    + [fe.ID, fe.MES, fe.CLASE, "clase_binaria"]
    + ["Visa_Finiciomora", "Master_Finiciomora"]   # ceros inyectados en 202105 y 202108
)


def params(n_filas: int) -> dict:
    """Receta de Denicolay (Zulip, J-Clase 08 > En Limpio, 2026-10-06)."""
    return {
        "objective": "binary", "boosting_type": "gbdt", "metric": "auc",
        "boost_from_average": True, "feature_pre_filter": False,
        "max_bin": 31, "min_data_in_leaf": 0,
        "feature_fraction": 0.5, "feature_fraction_bynode": 0.2,
        "learning_rate": 0.005, "num_leaves": 83,
        "min_sum_hessian_in_leaf": 12.791817 * n_filas / 326_184,
        "bagging_fraction": 1.0, "bagging_freq": 0,
        "num_threads": 8, "force_col_wise": True, "deterministic": True, "verbose": -1,
    }


def semillas() -> list[int]:
    np.random.seed(SEMILLAS_PROPIAS[0])      # sin esto cambian en cada corrida
    extra = np.random.choice(1_000_000, size=15, replace=False).tolist()
    return (SEMILLAS_PROPIAS + [int(x) for x in extra])[:N_SEMILLAS]


def construir_lags(con) -> None:
    """Las 150 crudas + lag1/delta1/lag2/delta2 de cada una, en FLOAT. Igual que scripts/c200 del 7-oct.

    OJO: la version que entreno la entrega anula ccajas_depositos en 202105 SOLO en la columna del mes;
    sus lags van sobre los ceros crudos. Se reproduce tal cual (scripts/c200 corrige esto para
    datasets futuros, pero la entrega es esta).
    """
    con.execute(f"create or replace view base as select * from read_parquet('{PANEL}')")
    campos = [x for x in fe.columnas_numericas(con, "base", excluir=("Visa_Finiciomora", "Master_Finiciomora"))
              if x not in FUERA]
    nuevas = []
    for x in campos:
        nuevas.append(f"lag({x}, 1) over historia :: FLOAT as {x}__lag1")
        nuevas.append(f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1")
        nuevas.append(f"lag({x}, 2) over historia :: FLOAT as {x}__lag2")
        nuevas.append(f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2")
    sql = (
        "select * exclude (ccajas_depositos)"
        f"\n  , case when {fe.MES} = 202105 then null else ccajas_depositos end as ccajas_depositos"
        + fe._fragmento(nuevas) + "\nfrom base" + fe.clausula_ventana("historia")
    )
    con.execute(f"copy ({sql}) to '{LAGS}' (format PARQUET, COMPRESSION ZSTD)")


def datos() -> pd.DataFrame:
    con = fe.conectar()
    if not PANEL.exists():
        print(f"  {PANEL.name} no existe: reconstruyendo desde {CRUDO.name}")
        fe.construir_base(con, str(CRUDO), str(PANEL))
    if not LAGS.exists():
        print(f"  {LAGS.name} no existe: construyendo la historia")
        construir_lags(con)
    con.close()
    d = pd.read_parquet(LAGS)
    d = d.drop(columns=fe.COLUMNAS_DE_BAJA, errors="ignore")
    d[fe.ID] = d[fe.ID].astype("int64")      # viene DOUBLE: 48000000 -> 4.8e+07 rompe el submit
    # DuckDB no garantiza el orden de las filas y LightGBM depende de el.
    return d.sort_values([fe.ID, fe.MES], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, default=Path("entrega_final.csv"))
    ap.add_argument("--semillas", type=int, default=N_SEMILLAS, help="menos semillas = prueba rapida (no reproduce el SHA)")
    ap.add_argument("--rondas", type=int, nargs="+", default=RONDAS_DEFAULT, help="num_boost_round de cada bloque de 20 semillas; varios valores = ensamble de todos")
    args = ap.parse_args()
    t0 = time.time()

    d = datos()
    pred = [x for x in d.columns if x not in FUERA]
    assert len(pred) == 750, f"{len(pred)} predictoras, se esperaban 750"
    assert len(d) == 983_061, f"{len(d):,} filas, se esperaban 983.061"
    tr = d[d[fe.MES].isin(MESES_TRAIN)]
    clase = tr[fe.CLASE].to_numpy()
    y = np.isin(clase, ["BAJA+1", "BAJA+2"]).astype("int8")
    w = np.where(clase == "BAJA+1", PESO_BAJA1, 1.0)
    X = tr[pred]
    fut = d[d[fe.MES] == MES_PREDECIR]
    ids = fut[fe.ID].to_numpy()
    Xfut = fut[pred]
    print(f"  train {MESES_TRAIN} ({len(X):,} filas, {int(y.sum()):,} positivos) -> {MES_PREDECIR} ({len(fut):,}) "
          f"| {len(pred)} predictoras [{time.time()-t0:.0f}s]", flush=True)

    P = params(len(X))
    rangos = []
    for nbr in args.rondas:
        ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
        for i, s in enumerate(semillas()[:args.semillas], 1):
            m = lgb.train({**P, "seed": s}, ds, num_boost_round=nbr)
            rangos.append(pd.Series(m.predict(Xfut)).rank(pct=True).to_numpy())
            print(f"    {nbr} rondas, semilla {s} ({i}/{args.semillas}) [{time.time()-t0:.0f}s]", flush=True)
    ens = np.mean(rangos, axis=0)            # ensamble por RANGO, no por score, de todos los modelos

    elegidos = np.sort(ids[np.argsort(-ens, kind="stable")[:CORTE]]).astype("int64")   # empates: por orden de id
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.salida, elegidos, fmt="%d")
    texto = args.salida.read_text()
    assert "e+" not in texto and "." not in texto, "notacion cientifica en el CSV"
    assert len(np.loadtxt(args.salida, dtype="int64")) == CORTE
    for nombre, (p_sel, p_resto, lift) in verificar_seleccion(elegidos, d).items():
        print(f"  {nombre:22s} {p_sel:6.1%} vs {p_resto:6.1%}   x{lift:.2f}")
    sha = hashlib.sha256(args.salida.read_bytes()).hexdigest()
    esperado = SHA_POR_RONDAS.get(tuple(args.rondas))
    print(f"\n  {args.salida}: {CORTE:,} envios, rondas {args.rondas}, {len(rangos)} modelos  [{time.time()-t0:.0f}s]")
    print(f"  sha256: {sha}")
    if args.semillas < N_SEMILLAS or esperado is None:
        print("  (sin SHA de referencia para esta combinacion)")
    else:
        print("  " + ("COINCIDE con el submit registrado" if sha == esperado else
                      f"NO coincide con el submit registrado ({esperado[:12]}...): revisar version de LightGBM, hilos y orden de filas"))
        if SHA_FINAL:
            print("  " + ("ES LA ENTREGA FINAL" if sha == SHA_FINAL else "no es la entrega final"))


if __name__ == "__main__":
    main()
