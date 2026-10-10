"""Feature engineering del panel cliente x mes, en SQL sobre DuckDB.

El panel de la competencia es una fila por (numero_de_cliente, foto_mes). Este
modulo concentra las dos cosas que todos los notebooks necesitan y que hoy
estan copiadas y pegadas entre ellos:

1. `construir_base`: arma el dataset base a partir del crudo, con la
   `clase_ternaria` de siempre mas cuatro columnas de baja a nivel cliente
   (`mes_ultimo`, `mes_baja`, `es_baja`, `k`).
2. Los generadores de SQL (`sql_rank`, `sql_lag`, `sql_delta`, `sql_slope`,
   `sql_ventana`): escriben a mano lo que seria insoportable tipear para 150
   campos, siguiendo los patrones de z402_Feature_Engineering_en_SQL.ipynb.

Uso tipico:

    import fe_panel as fe
    con = fe.conectar()
    fe.construir_base(con, "data/competencia_01_crudo.csv", "data/competencia_01.parquet")
"""

from __future__ import annotations

import duckdb

# --- constantes del curso -------------------------------------------------

#: Las primas que pidio la catedra al arrancar la cursada.
SEMILLAS: list[int] = [261431, 269281, 429899, 560771, 749401]
#: La de toda corrida unica (muestreo, KMeans, RandomForest).
SEMILLA: int = SEMILLAS[0]

#: Valor anual de un cliente premium, segun la minuta de la clase 1.
VALOR_CLIENTE_ANUAL: int = 2_200_000
#: Ganancia esperada de acertar un BAJA+2 (2.2M * 50% de aceptacion).
GANANCIA_ACIERTO: int = 1_072_500
#: Costo de estimular a un cliente.
COSTO_ESTIMULO: int = 27_500

#: Columnas que `construir_base` agrega mirando el futuro del cliente. Sirven para
#: analisis (quien se murio, cuando), pero meterlas en un modelo es leakage puro:
#: `es_baja` y `mes_baja` SON la respuesta. Hay que sacarlas antes de entrenar.
COLUMNAS_DE_BAJA = ["mes_ultimo", "mes_baja", "es_baja", "k", "meses_historia"]

ID = "numero_de_cliente"
MES = "foto_mes"
CLASE = "clase_ternaria"


# --- conexion y macros ----------------------------------------------------

_MACROS = """
-- foto_mes (YYYYMM) <-> indice de mes continuo, para poder sumar y restar meses
create or replace macro mes_indice(m) as ((m // 100) * 12 + (m % 100) - 1);
create or replace macro indice_mes(i) as ((i // 12) * 100 + (i % 12) + 1);
create or replace macro mes_mas(m, n) as indice_mes(mes_indice(m) + n);
create or replace macro meses_entre(a, b) as (mes_indice(b) - mes_indice(a));

-- el nulo de las c*/m* significa "no tiene el producto", o sea cero
create or replace macro suma_sin_null(a, b) as ifnull(a, 0) + ifnull(b, 0);
-- ratio que sobrevive al nulo y a la division por cero: devuelve NULL, que es honesto
create or replace macro ratio_seguro(a, b) as ifnull(a, 0) / nullif(ifnull(b, 0), 0);
"""


def conectar(base: str | None = None) -> duckdb.DuckDBPyConnection:
    """Abre una conexion DuckDB con las macros del proyecto ya registradas."""
    con = duckdb.connect(base) if base else duckdb.connect()
    con.execute(_MACROS)
    return con


# --- dataset base ---------------------------------------------------------

_SQL_BASE = """
with crudo as (
    select * from {origen}
),
periodos as (select distinct {mes} as {mes} from crudo),
clientes as (select distinct {id} as {id} from crudo),
-- el cliente puede faltar un mes y volver, asi que la grilla completa va primero
grilla as (select {id}, {mes} from clientes cross join periodos),
presencia as (
    select g.{id}, g.{mes},
           case when c.{id} is null then 0 else 1 end as mes_0
    from grilla g
    left join crudo c using ({id}, {mes})
),
horizonte as (
    select *,
           lead(mes_0, 1) over (partition by {id} order by {mes}) as mes_1,
           lead(mes_0, 2) over (partition by {id} order by {mes}) as mes_2
    from presencia
),
etiqueta as (
    select {id}, {mes}, mes_0,
           case when mes_1 = 1 and mes_2 = 0 then 'BAJA+2'
                when mes_1 = 0               then 'BAJA+1'
                when mes_1 = 1 and mes_2 = 1 then 'CONTINUA'
           end as {clase}
    from horizonte
),
-- lo mismo pero a nivel cliente: cuando se murio, si es que se murio
vida as (
    select {id},
           max({mes}) filter (where mes_0 = 1) as mes_ultimo,
           count(*)   filter (where mes_0 = 1) as meses_historia
    from etiqueta
    group by 1
),
panel as (select max({mes}) as mes_final from crudo)
select c.*,
       e.{clase},
       v.mes_ultimo,
       v.meses_historia,
       v.mes_ultimo < p.mes_final                       as es_baja,
       if(v.mes_ultimo < p.mes_final,
          mes_mas(v.mes_ultimo, 1), null)               as mes_baja,
       meses_entre(c.{mes}, v.mes_ultimo)               as k
from crudo c
join etiqueta e using ({id}, {mes})
join vida     v using ({id})
cross join panel p
"""


def construir_base(
    con: duckdb.DuckDBPyConnection,
    origen: str,
    destino: str | None = None,
    tabla: str = "competencia_01",
) -> duckdb.DuckDBPyRelation:
    """Construye el dataset base: crudo + clase_ternaria + columnas de baja.

    `clase_ternaria` conserva exactamente la semantica de z101_target_sql: es una
    etiqueta *de fila*, y por eso todo BAJA+2 del mes t vuelve a aparecer como
    BAJA+1 en t+1. Las columnas nuevas son *de cliente* y no tienen ese
    solapamiento:

    - `mes_ultimo`     ultimo foto_mes en que el cliente aparece
    - `meses_historia` cuantas fotos tiene en total
    - `es_baja`        True si desaparecio dentro de la ventana observable
    - `mes_baja`       el primer mes en que ya no esta (null si sobrevive)
    - `k`              meses entre esta foto y la ultima; 0 = su ultima foto

    La cohorte de bajas es entonces `select distinct numero_de_cliente where es_baja`,
    sin doble conteo, y `k` alinea a cada cliente en su propio reloj.
    """
    lector = (
        f"read_csv('{origen}', sample_size=-1)"
        if origen.endswith((".csv", ".csv.gz"))
        else f"read_parquet('{origen}')"
    )
    sql = _SQL_BASE.format(origen=lector, id=ID, mes=MES, clase=CLASE)
    con.execute(f"create or replace table {tabla} as {sql}")
    if destino:
        formato = "PARQUET, COMPRESSION ZSTD" if destino.endswith(".parquet") else "CSV, HEADER TRUE"
        con.execute(f"copy {tabla} to '{destino}' (format {formato})")
    return con.table(tabla)


# --- generadores de SQL ---------------------------------------------------
# Todos devuelven un fragmento de SELECT que empieza con coma, listo para pegar
# despues de la ultima columna de una consulta.


def _fragmento(lineas: list[str]) -> str:
    return "".join(f"\n  , {linea}" for linea in lineas)


def sql_rank(campos: list[str], particion: str = MES, funcion: str = "percent_rank") -> str:
    """Rankea cada campo dentro de su mes: el antidoto al data drifting.

    Con inflacion, "tiene 48.000 pesos" no es comparable entre marzo y agosto,
    pero "esta en el percentil 70 de su mes" si. `funcion` acepta cualquiera de
    percent_rank, cume_dist, ntile(4), ntile(10), rank, dense_rank.
    """
    llamada = funcion if "(" in funcion else f"{funcion}()"
    sufijo = funcion.replace("(", "_").replace(")", "").replace(" ", "")
    return _fragmento(
        [f"{llamada} over (partition by {particion} order by {c}) as {c}__{sufijo}" for c in campos]
    )


def sql_lag(campos: list[str], n: int = 1, ventana: str = "historia") -> str:
    """El valor de hace n meses del mismo cliente."""
    return _fragmento([f"lag({c}, {n}) over {ventana} as {c}__lag{n}" for c in campos])


def sql_delta(campos: list[str], n: int = 1, ventana: str = "historia") -> str:
    """Cuanto cambio el campo respecto de hace n meses."""
    return _fragmento(
        [f"{c} - lag({c}, {n}) over {ventana} as {c}__delta{n}" for c in campos]
    )


def sql_ventana(campos: list[str], aggs: tuple[str, ...] = ("avg", "min", "max"), ventana: str = "historia") -> str:
    """Agregados moviles sobre la ventana nombrada que se pase."""
    return _fragmento(
        [f"{a}({c}) over {ventana} as {c}__{a}" for c in campos for a in aggs]
    )


def sql_slope(campos: list[str], eje: str = "cliente_antiguedad", ventana: str = "historia") -> str:
    """La pendiente de minimos cuadrados del campo contra el eje temporal.

    Es la variable que encarna la hipotesis de negocio de la materia: un cliente
    no se va de golpe, se va apagando. Un ctrx_quarter de 40 no dice nada; un 40
    con pendiente -12 es un cliente en caida libre. Como `cliente_antiguedad`
    crece exactamente 1 por mes, la pendiente se lee "cuanto gana o pierde por mes".
    """
    return _fragmento(
        [f"regr_slope({c}, {eje}) over {ventana} as {c}__slope" for c in campos]
    )


def clausula_ventana(
    nombre: str = "historia",
    orden: str = MES,
    marco: str | None = None,
    particion: str = ID,
) -> str:
    """La clausula WINDOW que nombra la ventana una sola vez.

    `marco=None` usa toda la historia previa del cliente. Ojo con el clasico
    'rows between 3 preceding and current row': son CUATRO meses, no tres.
    """
    marco = f" {marco}" if marco else ""
    return f"\nwindow {nombre} as (partition by {particion} order by {orden}{marco})"


def columnas_numericas(con: duckdb.DuckDBPyConnection, tabla: str, excluir: tuple[str, ...] = ()) -> list[str]:
    """Las columnas numericas de una tabla, sin las llaves ni el target."""
    fijas = {ID, MES, CLASE, "mes_ultimo", "mes_baja", "es_baja", "k", "meses_historia", *excluir}
    filas = con.execute(
        f"select column_name, data_type from information_schema.columns where table_name = ?", [tabla]
    ).fetchall()
    numericos = ("BIGINT", "INTEGER", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT")
    return [c for c, t in filas if c not in fijas and any(t.startswith(n) for n in numericos)]
