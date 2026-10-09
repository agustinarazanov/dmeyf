"""Piezas compartidas de la Primera Competencia: datos, ganancia, corte y envios.

Convenciones que valen para todo el directorio:

- Una sola convencion de ganancia: +GANANCIA_ACIERTO si el estimulado era
  BAJA+2, -COSTO_ESTIMULO si no. Es la consistente con el umbral
  0,025 = 27.500 / (1.072.500 + 27.500). z601 mezcla dos y no las distingue.
- El corte se elige en CANTIDAD DE ENVIOS, no en probabilidad: la escala de los
  scores cambia con el target y con la semilla, la cantidad de envios no.
- Fuera de X van siempre: las columnas de baja (es_baja ES la respuesta),
  numero_de_cliente (asa de memorizacion), foto_mes (202108 es un valor que el
  modelo nunca vio) y las dos F*iniciomora (rotas justo en 202108).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fe_panel as fe  # noqa: E402

RAIZ = Path(__file__).resolve().parent
EXPERIMENTOS = RAIZ / "experimentos"
DATOS = RAIZ / "data"          # symlink a ../dmeyf2026/monday/data

GANANCIA_ACIERTO = fe.GANANCIA_ACIERTO      # 1_072_500
COSTO_ESTIMULO = fe.COSTO_ESTIMULO          # 27_500
UMBRAL = COSTO_ESTIMULO / (GANANCIA_ACIERTO + COSTO_ESTIMULO)   # 0.025
SEMILLAS = fe.SEMILLAS

MES_COMPETENCIA = 202108
MESES_ETIQUETADOS = [202103, 202104, 202105, 202106]

MIN_ENVIOS = 8_000
MAX_ENVIOS = 15_000

ROTAS_EN_202108 = ["Visa_Finiciomora", "Master_Finiciomora"]
NO_PREDICTORAS = [fe.ID, fe.MES, fe.CLASE, "clase_binaria", "peso"]
FUERA_DE_X = fe.COLUMNAS_DE_BAJA + NO_PREDICTORAS + ROTAS_EN_202108


def cargar(dataset: str = "competencia_01.parquet") -> pd.DataFrame:
    data = pd.read_parquet(DATOS / dataset)
    data = data.drop(columns=fe.COLUMNAS_DE_BAJA, errors="ignore")
    data[fe.ID] = data[fe.ID].astype("int64")
    # Orden canonico. DuckDB no garantiza el orden de las filas al escribir un parquet
    # y LightGBM depende de el, asi que sin esto el pipeline y reproducir.py entrenan
    # sobre ordenes distintos y entregan CSV distintos con las mismas semillas.
    # OJO: la llave de cache NO incluye el orden, asi que los modelos entrenados antes
    # de este cambio quedan obsoletos aunque entrenar_o_cargar los acepte.
    return data.sort_values([fe.ID, fe.MES], kind="mergesort").reset_index(drop=True)


def columnas_predictoras(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in FUERA_DE_X]


def preparar(data: pd.DataFrame, meses: list[int], target: str = "baja2",
             peso_baja1: float = 0.0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """X, y, peso para un conjunto de meses.

    target 'baja2'  : positivo = BAJA+2
    target 'baja12' : positivo = BAJA+1 o BAJA+2
    target 'baja1'  : positivo = BAJA+1 solo (horizonte 1: permite entrenar con 202107, que tiene BAJA+1 completo)
    target 'pesos'  : positivo = BAJA+1 o BAJA+2, con peso 1 para BAJA+2 y
                      peso_baja1 para BAJA+1. Interpola entre los dos anteriores.
    """
    d = data[data[fe.MES].isin(meses)]
    clase = d[fe.CLASE].to_numpy()
    if target == "baja1":
        y = (d[fe.CLASE] == "BAJA+1").astype("int8").to_numpy(); peso = np.ones(len(d))
    elif target == "baja2":
        y = (clase == "BAJA+2").astype("int8")
        peso = np.ones(len(d))
    elif target in ("baja12", "pesos"):
        y = np.isin(clase, ["BAJA+1", "BAJA+2"]).astype("int8")
        peso = np.ones(len(d)) if target == "baja12" else np.where(
            clase == "BAJA+1", peso_baja1, 1.0)
    else:
        raise ValueError(f"target desconocido: {target!r}")
    return d[columnas_predictoras(data)], y, peso


def aporte(es_baja2: np.ndarray) -> np.ndarray:
    # np.where() acepta cualquier cosa truthy: pasarle el array de clase_ternaria
    # (strings) da TODOS aciertos sin tirar error, y la ganancia sale 40x de mas.
    es_baja2 = np.asarray(es_baja2)
    if es_baja2.dtype.kind not in "bi" or (es_baja2.dtype.kind == "i" and es_baja2.max(initial=0) > 1):
        raise TypeError(f"aporte() espera booleanos o 0/1, recibio dtype={es_baja2.dtype}. "
                        f"Probablemente falte el '== \"BAJA+2\"'.")
    return np.where(es_baja2, GANANCIA_ACIERTO, -COSTO_ESTIMULO).astype("float64")


def ganancia_acumulada(scores: np.ndarray, es_baja2: np.ndarray) -> np.ndarray:
    orden = np.argsort(-np.asarray(scores), kind="stable")   # empates: orden de entrada, no el del quicksort
    return np.cumsum(aporte(es_baja2)[orden])


def ganancia(scores: np.ndarray, es_baja2: np.ndarray, envios: int) -> float:
    return float(ganancia_acumulada(scores, es_baja2)[envios - 1])


def curva_de_corte(scores, es_baja2, desde=6_000, hasta=16_000, paso=250) -> pd.DataFrame:
    acum = ganancia_acumulada(scores, es_baja2)
    cortes = np.arange(desde, min(hasta, len(acum)) + 1, paso)
    return pd.DataFrame({"envios": cortes, "ganancia": acum[cortes - 1]})


def meseta(curvas: pd.DataFrame, columna: str = "ganancia",
           tolerancia: float = 0.995) -> tuple[int, int, int]:
    """El tramo donde la ganancia media se mantiene sobre `tolerancia` del maximo.

    Devuelve (desde, centro, hasta). Se elige el centro, no el pico: cada semilla
    tiene su maximo en otro lado y el pico exacto es ruido.
    """
    medias = curvas.groupby("envios")[columna].mean()
    alto = medias[medias >= medias.max() * tolerancia]
    desde, hasta = int(alto.index.min()), int(alto.index.max())
    centro = int(np.clip(round((desde + hasta) / 2 / 250) * 250, MIN_ENVIOS, MAX_ENVIOS))
    return desde, centro, hasta


def simular_publico_privado(scores, es_baja2, envios: int, publico: float = 0.25,
                            repeticiones: int = 500, semilla: int = SEMILLAS[3]):
    """Reparte la ganancia del mes en publico/privado muchas veces.

    Da la resolucion del instrumento: cuanto se mueve la lectura publica por el
    solo sorteo de la particion, sin que el modelo cambie en nada.
    """
    scores, es_baja2 = np.asarray(scores), np.asarray(es_baja2)   # una Series con indice de etiquetas se indexa mal
    n = len(scores)
    elegidos = np.argsort(scores)[::-1][:envios]
    v = np.zeros(n)
    v[elegidos] = aporte(es_baja2[elegidos])
    rng = np.random.default_rng(semilla)
    n_pub = int(round(n * publico))
    pub = np.empty(repeticiones)
    for i in range(repeticiones):
        pub[i] = v[rng.permutation(n)[:n_pub]].sum()
    return pd.Series(pub, name="publico")


def clave(params: dict, meses: list[int], target: str, peso_baja1: float,
          dataset: str, num_boost_round: int, semilla: int) -> str:
    firma = json.dumps(
        {"params": {k: v for k, v in sorted(params.items()) if k != "seed"},
         "meses": sorted(meses), "target": target, "peso_baja1": peso_baja1,
         "dataset": dataset, "nbr": num_boost_round, "semilla": semilla},
        sort_keys=True, default=str)
    return hashlib.sha256(firma.encode()).hexdigest()[:16]


def entrenar_o_cargar(params: dict, X, y, peso, num_boost_round: int, semilla: int,
                      carpeta: Path, etiqueta: str) -> lgb.Booster:
    """Entrena, o carga del disco si ya se entreno lo mismo. El patron de z701."""
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"modelo_{etiqueta}.txt"
    if destino.exists():
        guardado = lgb.Booster(model_file=str(destino))
        # La llave de cache incluye el NOMBRE del dataset, no su contenido: si un
        # parquet se regenera con otras columnas, el nombre no cambia y se cargaria
        # un modelo ajeno (con la misma cantidad de columnas ni siquiera tira error,
        # devuelve scores mal). Chequear los nombres de las features lo detecta.
        if list(guardado.feature_name()) == list(X.columns):
            return guardado
        print(f"    cache invalido en {destino.name}: el modelo guardado tiene "
              f"{guardado.num_feature()} columnas distintas de las pedidas, reentreno")
        destino.unlink()
    p = {**params, "seed": semilla, "verbose": -1}
    modelo = lgb.train(p, lgb.Dataset(X, label=y, weight=peso),
                       num_boost_round=num_boost_round)
    modelo.save_model(str(destino))
    return modelo


def escribir_envios(ids, ruta: Path, validos: set[int] | None = None,
                    referencia: "pd.DataFrame | None" = None) -> int:
    """Escribe el CSV de entrega y lo vuelve a leer para verificarlo.

    El bot pide: sin encabezado, una columna, 8.000-15.000 filas, enteros y
    NUNCA notacion cientifica. numero_de_cliente viene como DOUBLE en el parquet
    y 48000000 / 31100000 estan los dos en 202108, asi que el 4.8e+07 no es
    hipotetico: es lo que pasa si uno no castea.
    """
    ids = np.asarray(ids).astype("int64")
    if not MIN_ENVIOS <= len(ids) <= MAX_ENVIOS:
        raise ValueError(f"{len(ids)} envios, fuera de [{MIN_ENVIOS}, {MAX_ENVIOS}]")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("hay numero_de_cliente repetidos")
    if validos is not None and not set(ids.tolist()) <= validos:
        faltan = set(ids.tolist()) - validos
        raise ValueError(f"{len(faltan)} ids no estan en el mes de la competencia")

    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Orden canonico por id. Al bot le da igual el orden (pide una columna de ids sin
    # encabezado), pero asi el archivo es identico byte a byte venga del pipeline o de
    # reproducir.py, y verificar la reproducibilidad es un diff en vez de un set().
    np.savetxt(ruta, np.sort(ids), fmt="%d")

    texto = ruta.read_text()
    if "e+" in texto or "E+" in texto or "." in texto:
        raise AssertionError(f"{ruta}: hay notacion cientifica o decimales")
    releido = np.loadtxt(ruta, dtype="int64", ndmin=1)
    if not np.array_equal(np.sort(releido), np.sort(ids)):
        raise AssertionError(f"{ruta}: lo releido no coincide con lo escrito")
    # el chequeo de `validos` es tautologico cuando sale del mismo array que `ids`.
    # Lo que importa es la CORRESPONDENCIA: que los elegidos parezcan elegidos por un
    # modelo y no indexados contra otra tabla. Ver verificar_seleccion().
    if referencia is not None:
        verificar_seleccion(ids, referencia)
    return len(ids)


def top_k(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    # estable: clientes con score identico (perfil todo-cero) entran en orden de id, no segun el quicksort
    return np.asarray(ids)[np.argsort(-np.asarray(scores), kind="stable")[:k]]


def ensamble_por_rank(scores_por_semilla: dict[int, np.ndarray]) -> np.ndarray:
    """Promedia posiciones, no probabilidades: las escalas no son comparables."""
    rangos = [pd.Series(s).rank(pct=True).to_numpy() for s in scores_por_semilla.values()]
    return np.mean(rangos, axis=0)


def regret(escenarios: dict[str, np.ndarray], desde: int = MIN_ENVIOS,
           hasta: int = MAX_ENVIOS, paso: int = 500) -> pd.DataFrame:
    """Cuanto se pierde, en cada escenario, por cortar en un K fijo.

    Un escenario es un mes de validacion con su curva de ganancia acumulada.
    Hace falta porque el K optimo lo manda la PREVALENCIA del mes —202105 tiene
    870 BAJA+2 y quiere 8.092 envios, 202106 tiene 1.098 y quiere 14.788— y la
    prevalencia de 202108 no se conoce. Elegir el K optimo de un mes solo es la
    misma trampa que elegir el pico de una semilla sola.
    """
    ks = np.arange(desde, hasta + 1, paso)
    tabla = pd.DataFrame(
        {nombre: [curva[k - 1] / curva.max() - 1 for k in ks]
         for nombre, curva in escenarios.items()}, index=ks)
    tabla.index.name = "envios"
    tabla["peor_caso"] = tabla.min(axis=1)
    tabla["esperado"] = tabla.drop(columns="peor_caso").mean(axis=1)
    return tabla


def corte_minimax(escenarios: dict[str, np.ndarray], **kw) -> tuple[int, pd.DataFrame]:
    tabla = regret(escenarios, **kw)
    return int(tabla["peor_caso"].idxmax()), tabla


def referencia_202108() -> pd.DataFrame:
    """Solo las columnas y el mes que usa verificar_seleccion: evita cargar y ordenar 121 MB por corrida."""
    return pd.read_parquet(DATOS / "competencia_01.parquet",
                           columns=[fe.ID, fe.MES, "ctrx_quarter", "mcuentas_saldo", "cpayroll_trx"],
                           filters=[(fe.MES, "==", MES_COMPETENCIA)])


def verificar_seleccion(elegidos, data, mes: int = MES_COMPETENCIA,
                        minimo_lift: float = 2.0) -> dict:
    """¿La seleccion parece hecha por un modelo, o parece al azar?

    El guard de escribir_envios chequea que los ids EXISTAN en el mes; eso es
    membresia, no correspondencia. Si uno indexa scores de una tabla contra los
    ids de otra —y DuckDB reordena las filas al escribir el parquet— salen ids
    perfectamente validos que no son los que el modelo eligio. Pasa el guard y
    el bot devuelve ganancia NEGATIVA.

    Esto lo caza antes de enviar: los elegidos tienen que estar enriquecidos en
    marcadores de riesgo conocidos. Si no hay separacion, la seleccion es ruido.
    """
    mes_df = data[data[fe.MES] == mes]
    elegidos = np.asarray(elegidos)
    sel = mes_df[fe.ID].isin(set(elegidos.tolist()))
    # FALLA CERRADO. Si los elegidos no son del mes, cond[sel] queda vacio, la media da
    # NaN, el lift da NaN y `NaN < minimo_lift` es False: el guard no levantaba nada
    # JUSTO en el caso de desalineacion para el que se escribio.
    cubiertos = int(sel.sum())
    if cubiertos < len(elegidos):
        raise AssertionError(
            f"solo {cubiertos:,} de {len(elegidos):,} elegidos estan en el mes {mes}. "
            f"Los ids no corresponden a esa foto.")
    marcadores = {
        "ctrx_quarter == 0": mes_df["ctrx_quarter"].fillna(0) == 0,
        "mcuentas_saldo < 0": mes_df["mcuentas_saldo"].fillna(0) < 0,
        "sin sueldo": mes_df["cpayroll_trx"].fillna(0) == 0,
    }
    reporte = {}
    for nombre, cond in marcadores.items():
        p_sel = cond[sel].mean()
        p_resto = cond[~sel].mean()
        reporte[nombre] = (p_sel, p_resto, p_sel / p_resto if p_resto else np.inf)
    lifts = [r[2] for r in reporte.values()]
    if not all(np.isfinite(l) for l in lifts):
        raise AssertionError(
            f"hay lifts no finitos {lifts}: algun marcador no se pudo evaluar. "
            f"Tipicamente los elegidos no corresponden al mes {mes}.")
    mejor = max(lifts)
    if mejor < minimo_lift:
        detalle = " | ".join(f"{k}: {v[0]:.1%} vs {v[1]:.1%} (x{v[2]:.2f})"
                             for k, v in reporte.items())
        raise AssertionError(
            f"la seleccion no se distingue de la poblacion (lift max {mejor:.2f} < "
            f"{minimo_lift}). Casi seguro los scores y los ids vienen de tablas "
            f"con distinto orden de filas. {detalle}")
    return reporte
