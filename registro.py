"""Registro de submits de la Primera Competencia, en Postgres.

Una fila por experimento, una por submit, una por archivo adjunto y una por
respuesta del bot. La base guarda metadatos y punteros; los arrays (scores,
curvas de ganancia) viven en parquet al lado de los modelos.

La regla que la estructura impone: un submit nace en estado 'preparado' con
enviado_en NULL. Nada lo pasa a 'enviado' salvo marcar_enviado(), que se llama
despues de la aprobacion explicita.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

RAIZ = Path(__file__).resolve().parent
EXPERIMENTOS = RAIZ / "experimentos"

ESTADOS = ("preparado", "enviado", "respondido", "fallido", "descartado")

ESQUEMA = """
create table if not exists experimento (
    id                 serial primary key,
    nombre             text unique not null,
    creado_en          timestamptz not null default now(),
    notebook           text,
    git_sha            text,
    descripcion        text,
    dataset            text,
    n_columnas         integer,
    meses_train        integer[],
    mes_validacion     integer,
    target             text,
    peso_baja1         numeric,
    params             jsonb,
    num_boost_round    integer,
    n_semillas         integer,
    corte_envios       integer,
    ganancia_val_media numeric,
    ganancia_val_sd    numeric,
    justificacion_corte text,
    ruta               text
);

create table if not exists submit (
    id                serial primary key,
    nombre            text unique not null,
    experimento_id    integer references experimento(id),
    creado_en         timestamptz not null default now(),
    enviado_en        timestamptz,
    estado            text not null default 'preparado',
    es_final          boolean not null default false,
    hipotesis         text,
    delta_contra      text,
    delta_descripcion text,
    zulip_message_id  bigint,
    notas             text,
    constraint submit_estado_valido check (estado in
        ('preparado','enviado','respondido','fallido','descartado')),
    constraint submit_nombre_limpio check (nombre ~ '^[A-Za-z0-9_-]+$')
);

create table if not exists archivo (
    id        serial primary key,
    submit_id integer not null references submit(id) on delete cascade,
    nombre    text not null,
    ruta      text not null,
    semilla   bigint,
    n_envios  integer,
    sha256    text,
    unique (submit_id, nombre)
);

create table if not exists resultado (
    id               serial primary key,
    submit_id        integer not null references submit(id) on delete cascade,
    recibido_en      timestamptz not null default now(),
    public_gain_mean numeric,
    public_gain_std  numeric,
    n_archivos       integer,
    submits_usados   integer,
    respuesta_cruda  text,
    submission_id    text,
    unique (submit_id)
);

create table if not exists leaderboard (
    id               serial primary key,
    tomado_en        timestamptz,
    puesto           integer,
    alumno           text,
    ganancia         numeric,
    zulip_message_id bigint
);

create index if not exists idx_submit_experimento on submit(experimento_id);
create index if not exists idx_archivo_submit     on archivo(submit_id);
create index if not exists idx_resultado_submit   on resultado(submit_id);

create or replace view v_submits as
select s.nombre            as submit,
       s.estado,
       s.es_final,
       e.nombre            as experimento,
       e.target,
       e.corte_envios,
       e.ganancia_val_media,
       r.public_gain_mean,
       r.public_gain_std,
       count(a.id)         as n_archivos,
       s.delta_contra,
       s.delta_descripcion,
       s.hipotesis,
       s.creado_en,
       s.enviado_en
from submit s
left join experimento e on e.id = s.experimento_id
left join archivo    a on a.submit_id = s.id
left join resultado  r on r.submit_id = s.id
group by s.id, e.nombre, e.target, e.corte_envios, e.ganancia_val_media,
         r.public_gain_mean, r.public_gain_std;
"""


def cargar_env(ruta: Path | None = None) -> None:
    ruta = ruta or RAIZ / ".env"
    if not ruta.exists():
        return
    for linea in ruta.read_text().splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip().strip('"').strip("'"))


def dsn() -> str:
    cargar_env()
    if os.environ.get("COMPETENCIA_DSN"):
        return os.environ["COMPETENCIA_DSN"]
    return (
        f"dbname={os.environ.get('POSTGRES_DB', 'competencia')} "
        f"user={os.environ.get('POSTGRES_USER', 'postgres')} "
        f"password={os.environ.get('POSTGRES_PASSWORD', '')} "
        f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
        f"port={os.environ.get('POSTGRES_PORT', '5432')}"
    )


def dsn_url() -> str:
    """El mismo destino en formato URL, que es lo que pide Optuna."""
    cargar_env()
    return (
        f"postgresql+psycopg://{os.environ.get('POSTGRES_USER', 'postgres')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', '')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'competencia')}"
    )


def conectar() -> psycopg.Connection:
    return psycopg.connect(dsn(), row_factory=dict_row, autocommit=True)


def crear_esquema(con: psycopg.Connection | None = None) -> None:
    propia = con is None
    con = con or conectar()
    try:
        con.execute(ESQUEMA)
    finally:
        if propia:
            con.close()


def _sha256(ruta: str | Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def alta_experimento(con: psycopg.Connection, nombre: str, **campos) -> int:
    """Inserta (o actualiza) un experimento y devuelve su id."""
    if "params" in campos and campos["params"] is not None:
        campos["params"] = json.dumps(campos["params"])
    columnas = ["nombre", *campos.keys()]
    valores = [nombre, *campos.values()]
    marcas = ", ".join(["%s"] * len(columnas))
    actualiza = ", ".join(f"{c} = excluded.{c}" for c in campos) or "nombre = excluded.nombre"
    fila = con.execute(
        f"insert into experimento ({', '.join(columnas)}) values ({marcas}) "
        f"on conflict (nombre) do update set {actualiza} returning id",
        valores,
    ).fetchone()
    return fila["id"]


def alta_submit(
    con: psycopg.Connection,
    nombre: str,
    experimento_id: int,
    archivos: list[dict],
    hipotesis: str | None = None,
    delta_contra: str | None = None,
    delta_descripcion: str | None = None,
    es_final: bool = False,
    notas: str | None = None,
) -> int:
    """Registra un submit en estado 'preparado'. NO lo envia.

    archivos: [{"nombre":..., "ruta":..., "semilla":..., "n_envios":...}, ...]
    """
    # Todo o nada: si falla a mitad de los archivos queda un submit a medio registrar
    # y el reintento se estrella contra el UNIQUE del nombre.
    with con.transaction():
        fila = con.execute(
            "insert into submit (nombre, experimento_id, hipotesis, delta_contra, "
            "delta_descripcion, es_final, notas) values (%s,%s,%s,%s,%s,%s,%s) returning id",
            (nombre, experimento_id, hipotesis, delta_contra, delta_descripcion, es_final, notas),
        ).fetchone()
        submit_id = fila["id"]
        for a in archivos:
            con.execute(
                "insert into archivo (submit_id, nombre, ruta, semilla, n_envios, sha256) "
                "values (%s,%s,%s,%s,%s,%s)",
                (submit_id, a["nombre"], str(a["ruta"]), a.get("semilla"),
                 a.get("n_envios"), a.get("sha256") or _sha256(a["ruta"])),
            )
    return submit_id


def marcar_enviado(con, nombre: str, zulip_message_id: int | None = None) -> None:
    con.execute(
        "update submit set estado='enviado', enviado_en=%s, zulip_message_id=%s "
        "where nombre=%s",
        (datetime.now(timezone.utc), zulip_message_id, nombre),
    )


def marcar_estado(con, nombre: str, estado: str, notas: str | None = None) -> None:
    if estado not in ESTADOS:
        raise ValueError(f"estado invalido: {estado}")
    con.execute(
        "update submit set estado=%s, notas=coalesce(%s, notas) where nombre=%s",
        (estado, notas, nombre),
    )


def alta_resultado(con, nombre_submit: str, respuesta_cruda: str,
                   public_gain_mean=None, public_gain_std=None,
                   n_archivos=None, submits_usados=None) -> int:
    fila = con.execute("select id from submit where nombre=%s", (nombre_submit,)).fetchone()
    if fila is None:
        raise LookupError(f"no existe el submit {nombre_submit!r}")
    res = con.execute(
        "insert into resultado (submit_id, public_gain_mean, public_gain_std, "
        "n_archivos, submits_usados, respuesta_cruda) values (%s,%s,%s,%s,%s,%s) returning id",
        (fila["id"], public_gain_mean, public_gain_std, n_archivos,
         submits_usados, respuesta_cruda),
    ).fetchone()
    con.execute("update submit set estado='respondido' where nombre=%s", (nombre_submit,))
    return res["id"]


def listar(con, solo_enviados: bool = False):
    import pandas as pd
    sql = "select * from v_submits"
    if solo_enviados:
        sql += " where enviado_en is not null"
    sql += " order by public_gain_mean desc nulls last, creado_en desc"
    return pd.DataFrame(con.execute(sql).fetchall())


def purgar(con, umbral: float | None = None, borrar_modelos: bool = False) -> list[str]:
    """Borra los bytes de los submits descartados o de mal score.

    Conserva siempre la fila: se tira el archivo, se conserva el conocimiento
    para no repetir el experimento.
    """
    sql = ("select s.nombre, a.ruta from submit s join archivo a on a.submit_id=s.id "
           "left join resultado r on r.submit_id=s.id where s.es_final = false and (")
    cond = ["s.estado = 'descartado'"]
    params: list = []
    if umbral is not None:
        cond.append("r.public_gain_mean < %s")
        params.append(umbral)
    sql += " or ".join(cond) + ")"
    borrados = []
    for fila in con.execute(sql, params).fetchall():
        p = Path(fila["ruta"])
        if p.exists():
            p.unlink()
            borrados.append(str(p))
    if borrar_modelos:
        for fila in con.execute(
            "select e.ruta from experimento e join submit s on s.experimento_id=e.id "
            "where s.estado='descartado' and e.ruta is not null"
        ).fetchall():
            for m in Path(fila["ruta"]).glob("modelo_s*.txt"):
                m.unlink()
                borrados.append(str(m))
    return borrados


if __name__ == "__main__":
    with conectar() as con:
        crear_esquema(con)
        print("esquema creado en", os.environ.get("POSTGRES_DB"))
        for t in con.execute(
            "select table_name from information_schema.tables "
            "where table_schema='public' order by 1"
        ).fetchall():
            print("  ", t["table_name"])
