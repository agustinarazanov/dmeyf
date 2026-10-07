"""Cuanto ruido tiene COMPARAR dos submits en el leaderboard publico.

El sd que devuelve el bot mide la semilla sobre una particion fija; no sirve
como umbral de decision. Lo que importa es otra cosa: si el publico dice que A
le gana a B por X, ¿cuanto de eso es señal?

La particion publica es FIJA, asi que la comparacion entre dos submits esta
pareada sobre ella y el ruido relevante es el de la DIFERENCIA, no el de cada
lectura por separado. Esto lo mide sobre 202106, donde sí tenemos la verdad.
"""

import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import competencia as c

PUBLICO = 0.25
REPETICIONES = 2_000
CARPETA = c.EXPERIMENTOS / "c102_comparacion"


def aporte_en_corte(score: np.ndarray, es: np.ndarray, corte: int) -> np.ndarray:
    v = np.zeros(len(score))
    elegidos = np.argsort(score)[::-1][:corte]
    v[elegidos] = c.aporte(es[elegidos])
    return v


def main() -> None:
    resumen = pd.read_parquet(CARPETA / "resumen.parquet")
    data_por_archivo = {}
    aportes, totales = {}, {}

    for _, fila in resumen.iterrows():
        archivo = "competencia_01_fe.parquet" if fila.config.startswith("fe") \
            else "competencia_01.parquet"
        if archivo not in data_por_archivo:
            d = c.cargar(archivo)
            data_por_archivo[archivo] = (d, c.columnas_predictoras(d))
        data, pred = data_por_archivo[archivo]
        val = data[data[c.fe.MES] == 202106]
        es = (val[c.fe.CLASE].to_numpy() == "BAJA+2")
        modelos = sorted(CARPETA.glob(f"modelo_{fila.config}_202106_*.txt"))
        if not modelos:
            continue
        s = np.mean([lgb.Booster(model_file=str(m)).predict(val[pred]) for m in modelos], axis=0)
        v = aporte_en_corte(s, es, int(fila.corte))
        aportes[fila.config] = v
        totales[fila.config] = v.sum()

    n = len(next(iter(aportes.values())))
    n_pub = int(round(n * PUBLICO))
    rng = np.random.default_rng(c.SEMILLAS[3])
    cortes = [rng.permutation(n)[:n_pub] for _ in range(REPETICIONES)]
    pub = {k: np.array([v[idx].sum() for idx in cortes]) for k, v in aportes.items()}

    print(f"{len(aportes)} configuraciones, {REPETICIONES} particiones de {PUBLICO:.0%} sobre 202106\n")
    print("lectura publica de cada una (media +- sd por el sorteo de la particion):")
    for k, v in pub.items():
        print(f"  {k:18} {v.mean()/1e6:>7,.1f} M  +- {v.std()/1e6:>5,.1f} M   "
              f"(total mes {totales[k]/1e6:>7,.1f} M)")

    filas = []
    for a, b in itertools.combinations(aportes, 2):
        verdad = totales[a] - totales[b]
        d = pub[a] - pub[b]
        estimador = d / PUBLICO
        filas.append({
            "A": a, "B": b,
            "dif_real_M": verdad / 1e6,
            "dif_publica_media_M": d.mean() / 1e6,
            "sd_dif_publica_M": d.std() / 1e6,
            "sd_estimador_M": estimador.std() / 1e6,
            "acierta_el_orden": float(np.mean(np.sign(d) == np.sign(verdad))),
        })
    comp = pd.DataFrame(filas).sort_values("dif_real_M", key=abs, ascending=False)
    comp.to_parquet(CARPETA / "ruido_publico.parquet", index=False)
    print("\ncomparaciones pareadas:")
    print(comp.round(2).to_string(index=False))

    sd_tipico = comp.sd_dif_publica_M.median()
    print(f"\nsd tipico de la DIFERENCIA publica entre dos configs: {sd_tipico:,.2f} M")
    print(f"  -> una brecha publica menor a ~{2*sd_tipico:,.1f} M (2 sd) no distingue nada")
    print(f"  -> en el mes entero eso son ~{2*sd_tipico/PUBLICO:,.1f} M")
    chicas = comp[comp.dif_real_M.abs() < 10]
    if len(chicas):
        print(f"\ncuando la diferencia real es < 10 M, el publico acierta el orden "
              f"{chicas.acierta_el_orden.mean():.1%} de las veces")
    grandes = comp[comp.dif_real_M.abs() >= 10]
    if len(grandes):
        print(f"cuando es >= 10 M, acierta {grandes.acierta_el_orden.mean():.1%}")


if __name__ == "__main__":
    main()
