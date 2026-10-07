"""Canaritos asesinos: poda de features comparando contra ruido puro.

Portado de `labo-imp/dmeyf2024`, src/wf-etapas/z1601_CN_canaritos_asesinos.r.

La idea: se inyectan variables aleatorias ("canaritos") proporcionales al ancho del
dataset, se entrena, y se elimina toda variable REAL que rankee por importancia por
debajo de los canaritos. Una variable que no le gana al ruido no aporta.

    umbral      = mediana(posicion de los canaritos) + desvios * sd(posicion)
    sobreviven  = reales con posicion < umbral

Por que nos importa: agregamos 73 columnas (223 totales) con feature_fraction=0,277,
asi que cada arbol ve ~62 de 223 y las buenas quedan DILUIDAS entre ruido. El pipeline
de la catedra poda ANTES de entrenar; nosotros nunca lo hicimos, y es la explicacion
mas probable de por que el FE grande perdio 5,2 sigma contra el publico.
"""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import competencia as c

PARAMS = {
    "objective": "binary", "boosting_type": "gbdt", "first_metric_only": True,
    "boost_from_average": True, "feature_pre_filter": False, "max_bin": 31,
    "num_leaves": 45, "learning_rate": 0.0077, "min_data_in_leaf": 174,
    "feature_fraction": 0.277, "bagging_fraction": 0.918, "bagging_freq": 1,
    "verbose": -1,
}
MESES = [202103, 202104, 202105, 202106]
NBR = 250


def sobrevivientes(data, predictoras, semilla, ratio=0.2, desvios=4.0,
                   meses=MESES, params=PARAMS, num_boost_round=NBR):
    """Devuelve (lista de sobrevivientes, diagnostico) para UNA semilla."""
    X, y, w = c.preparar(data, meses, "pesos", 0.25)
    X = X[predictoras].copy()
    rng = np.random.default_rng(semilla)
    n_can = max(1, int(len(predictoras) * ratio))
    canaritos = [f"canarito{i:04d}" for i in range(n_can)]
    # de a bloques: concat una sola vez, no 44 inserts que fragmentan el DataFrame
    ruido = pd.DataFrame(rng.uniform(size=(len(X), n_can)), columns=canaritos, index=X.index)
    X = pd.concat([X, ruido], axis=1)

    modelo = lgb.train({**params, "seed": semilla},
                       lgb.Dataset(X, label=y, weight=w), num_boost_round=num_boost_round)
    imp = pd.DataFrame({"feature": modelo.feature_name(),
                        "gain": modelo.feature_importance("gain")})
    # posicion 1 = la mas importante. Desempate estable por nombre.
    imp = imp.sort_values(["gain", "feature"], ascending=[False, True]).reset_index(drop=True)
    imp["pos"] = np.arange(1, len(imp) + 1)
    pos_can = imp.loc[imp.feature.isin(canaritos), "pos"]
    umbral = pos_can.median() + desvios * pos_can.std()
    vivos = imp[(~imp.feature.isin(canaritos)) & (imp.pos < umbral)]
    diag = {"semilla": int(semilla), "n_canaritos": n_can, "umbral": float(umbral),
            "canarito_mejor": int(pos_can.min()), "canarito_mediana": float(pos_can.median()),
            "n_vivos": int(len(vivos)), "n_gain_cero": int((imp.gain == 0).sum())}
    return vivos.feature.tolist(), diag


def podar(data, predictoras, semillas, ratio=0.2, desvios=4.0, minimo_votos=None):
    """Corre los canaritos con varias semillas y se queda con las que sobreviven
    en al menos `minimo_votos` de ellas. Una sola semilla elige ruido."""
    minimo_votos = minimo_votos or (len(semillas) // 2 + 1)
    votos, diags = {}, []
    for s in semillas:
        vivos, d = sobrevivientes(data, predictoras, s, ratio, desvios)
        for f in vivos:
            votos[f] = votos.get(f, 0) + 1
        diags.append(d)
        print(f"    semilla {s:<7} umbral {d['umbral']:7.1f}  sobreviven {d['n_vivos']:4d}"
              f"  (mejor canarito en la posicion {d['canarito_mejor']})", flush=True)
    finales = sorted([f for f, v in votos.items() if v >= minimo_votos],
                     key=lambda f: -votos[f])
    return finales, votos, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="competencia_01_fe.parquet")
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--desvios", type=float, default=4.0)
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--salida", type=Path, default=Path("canaritos_sobrevivientes.json"))
    args = ap.parse_args()

    t0 = time.time()
    data = c.cargar(args.dataset)
    pred = c.columnas_predictoras(data)
    sem = c.SEMILLAS[: args.semillas]
    print(f"{args.dataset}: {len(pred)} predictoras | ratio {args.ratio} | "
          f"desvios {args.desvios} | {len(sem)} semillas\n")

    finales, votos, diags = podar(data, pred, sem, args.ratio, args.desvios)
    print(f"\n  sobreviven {len(finales)} de {len(pred)} "
          f"({len(finales)/len(pred):.0%}) por mayoria de {len(sem)} semillas")

    estab = [d["n_vivos"] for d in diags]
    print(f"  estabilidad entre semillas: {min(estab)}-{max(estab)} vivos "
          f"(sd {np.std(estab):.1f}, {100*np.std(estab)/np.mean(estab):.1f}%)")
    cero = [d["n_gain_cero"] for d in diags]
    print(f"  variables con gain=0: {min(cero)}-{max(cero)} — ninguna deberia sobrevivir")

    base = set(c.columnas_predictoras(c.cargar("competencia_01.parquet")))
    de_base = [f for f in finales if f in base]
    print(f"\n  de las {len(base)} columnas base sobreviven {len(de_base)} "
          f"({len(de_base)/len(base):.0%})")
    print(f"  de las {len(pred)-len(base)} nuevas del FE sobreviven "
          f"{len(finales)-len(de_base)} ({(len(finales)-len(de_base))/max(1,len(pred)-len(base)):.0%})")

    print("\n  top 15 sobrevivientes:")
    for f in finales[:15]:
        print(f"    {f}")

    args.salida.write_text(json.dumps(
        {"dataset": args.dataset, "ratio": args.ratio, "desvios": args.desvios,
         "semillas": [int(s) for s in sem], "n_predictoras": len(pred),
         "sobrevivientes": finales, "votos": votos, "diagnostico": diags}, indent=1))
    print(f"\n  -> {args.salida}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
