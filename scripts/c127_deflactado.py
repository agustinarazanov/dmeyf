"""¿Que features envejecen menos? Deflactar montos vs historicas vs crudas.

Hallazgo que motiva esto: el modelo pierde ~3 puntos de captura por cada mes de
distancia entre los datos de entrenamiento y el mes a predecir, y eso vale ~50 M
(12x todo lo que logro el feature engineering). Los tres remedios que probamos
—decaimiento, sacar meses, solo el reciente— fallaron porque atacaban EL MES y
el problema es LA DISTANCIA.

Hipotesis nueva: hay features que envejecen menos.
  - Un nivel en pesos (mcuentas_saldo) se corre con la inflacion (+14,6% entre
    marzo y agosto) y con los cambios de precio del banco.
  - Un monto DEFLACTADO por la mediana de su propio mes conserva la forma de la
    distribucion pero queda comparable entre meses. Es el punto medio entre el
    crudo y el percent_rank, que destruye la escala del todo.
  - Una feature HISTORICA (meses en rojo, pendiente) es relativa al propio
    cliente, asi que no deberia correrse con el calendario.

EL TEST no es «¿mejora la ganancia?» sino «¿pierde menos captura al alejarse?».
Se mide el MISMO mes de train a dos distancias:
    202103 -> 202105  (distancia 2)
    202103 -> 202106  (distancia 3)
La captura no depende de la prevalencia del mes, asi que se puede comparar.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fe_panel as fe
import competencia as c

DESTINO = "data/competencia_01_defl.parquet"
KS = (5_000, 9_000, 11_000, 15_000)
NBR, N_SEMILLAS = 250, 10
PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
}
CARPETA = c.EXPERIMENTOS / "c127_deflactado"
CARPETA.mkdir(parents=True, exist_ok=True)


def construir(raiz: Path) -> list[str]:
    con = fe.conectar()
    con.execute(f"""create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / 'data/competencia_01.parquet'}')""")
    cols = [r[0] for r in con.sql("describe crudo").fetchall()]
    montos = [x for x in cols if x.startswith("m") and x not in ("mes",)]
    # deflactor POR COLUMNA: la mediana de los valores positivos de ese mes.
    # Por columna y no global porque no se mueven parejas: mpayroll salta x1,6 en
    # junio por el aguinaldo y mcomisiones_mantenimiento hace un escalon porque el
    # banco cambio el precio. Nada de eso es inflacion.
    defl = ", ".join(
        f'median(case when {m} > 0 then {m} end) over (partition by foto_mes) as "{m}__d"'
        for m in montos)
    con.execute(f"create or replace table med as select *, {defl} from crudo")
    ratio = ", ".join(f'ratio_seguro({m}, "{m}__d") as "{m}__defl"' for m in montos)
    con.execute(f"""create or replace table defl as
        select * exclude ({", ".join(f'"{m}__d"' for m in montos)}), {ratio} from med""")
    con.execute(f"copy defl to '{raiz / DESTINO}' (format parquet, compression zstd)")
    n = con.sql("select count(*) from (describe defl)").fetchone()[0]
    print(f"{len(cols)} -> {n} columnas ({len(montos)} montos deflactados)")
    con.close()
    return montos


def captura(data, pred, meses, mes_val, etq, semillas):
    X, y, w = c.preparar(data, meses, "pesos", 0.25)
    X = X[pred]
    val = data[data[c.fe.MES] == mes_val]
    es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
    caps = []
    for s in semillas:
        cl = c.clave(PARAMS, meses, "pesos", 0.25, f"{etq}{len(pred)}", NBR, s)
        m = c.entrenar_o_cargar(PARAMS, X, y, w, NBR, s, CARPETA, f"{etq}_{mes_val}_{s}_{cl}")
        orden = np.argsort(m.predict(val[pred]))[::-1]
        acum = np.cumsum(es[orden])
        caps.append([acum[k - 1] / es.sum() for k in KS])
    return np.mean(caps, axis=0)


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    montos = construir(raiz)
    d = c.cargar("competencia_01_defl.parquet")
    todas = c.columnas_predictoras(d)
    defl = [x for x in todas if x.endswith("__defl")]
    crudas = [x for x in todas if not x.endswith("__defl")]
    sets = {
        "crudas":          crudas,
        "deflactadas":     [x for x in crudas if x not in montos] + defl,   # reemplaza montos
        "crudas+deflact":  todas,
    }
    np.random.seed(c.SEMILLAS[0])
    sem = c.SEMILLAS + np.random.choice(1_000_000, size=N_SEMILLAS - 5, replace=False).tolist()
    print(f"\nmismo mes de train (202103) a dos distancias, {N_SEMILLAS} semillas\n")
    print(f"{'set':18} {'cols':>5} {'dist':>5} " + " ".join(f"{k//1000}k".rjust(8) for k in KS))
    filas = []
    for etq, pred in sets.items():
        for mes_val, dist in ((202105, 2), (202106, 3)):
            cap = captura(d, pred, [202103], mes_val, etq, sem)
            filas.append({"set": etq, "dist": dist, **{f"k{k}": v for k, v in zip(KS, cap)}})
            print(f"  {etq:16} {len(pred):>5} {dist:>5} " + " ".join(f"{x:8.1%}" for x in cap)
                  + f"   [{time.time()-t0:.0f}s]")
    t = pd.DataFrame(filas)
    t.to_parquet(CARPETA / "captura.parquet", index=False)
    print("\n=== cuanta captura se PIERDE al pasar de distancia 2 a 3 ===")
    for etq in sets:
        a = t[(t.set == etq) & (t.dist == 2)].iloc[0]
        b = t[(t.set == etq) & (t.dist == 3)].iloc[0]
        p = [a[f"k{k}"] - b[f"k{k}"] for k in KS]
        print(f"  {etq:16} " + " ".join(f"{x:+8.1%}" for x in p) + f"   media {np.mean(p):+.2%}")
    print("\n  (menos perdida = envejece menos)")
    print(f"\nlisto en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
