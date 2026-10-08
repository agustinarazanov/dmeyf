"""Reparaciones de calendario juntas: aguinaldo (c252) + cliente_vip de junio + Visa_delinquency "fresca".

DICCIONARIO (leido antes de tocar):
- cliente_vip: "indica si marketing considera a esa cliente un cliente vip al momento de obtencion de la foto".
  Es una marca de marketing, no una conducta: que se duplique solo en junio (921 contra ~430) y vuelva en
  julio es un cambio de criterio de un mes. Reemplazar junio por mayo es un juicio, no una correccion de error.
- Visa_delinquency: "{0,1} indica si el cliente no llego a completar el pago minimo y esta moroso".
- Visa_Finiciomora: "dias para el inicio de la mora (el dia siguiente al vencimiento), contados a la fecha
  de la foto". Finiciomora = 0 con delinquency = 1 es entonces "vencio ayer y todavia no pago": por
  definicion NO es un dato roto sino mora recien iniciada, que aparece en masa en los meses cuyo cierre
  de tarjeta cae justo antes de la foto (05 y 08). Se la pone en 0 porque en junio (train) su delta1 es
  negativo y en agosto (scoring) positivo, un patron de calendario que el modelo no puede haber aprendido;
  es una decision de modelado sobre un valor correcto, no una reparacion.

El escaneo de las 600 columnas derivadas (8-oct 19:40) encontro, ademas del aguinaldo, dos artefactos de
calendario que llegan a los lags de agosto:
- cliente_vip: 921 en 202106 contra ~430 los demas meses (se duplica solo en junio); en agosto
  delta2 = agosto - junio da negativo a ~490 clientes. Se reemplaza el valor de junio por el de mayo.
- Visa_delinquency: inyectada en 202105 y 202108 (4.344 / 4.163 contra ~950). En junio delta1 = junio - mayo
  es negativo para los inyectados; en agosto delta1 = agosto - julio es positivo: el patron de entrenamiento
  es el inverso del de scoring. Se pone 0 donde Visa_Finiciomora = 0 (la firma de la inyeccion) en 05 y 08.

Parte de competencia_01_aguinaldo.parquet y recalcula solo los lags de estas dos columnas.
Salida: data/competencia_01_calendario.parquet
"""
import argparse, sys, time
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import fe_panel as fe
import competencia as c
ORIGEN = c.DATOS / "competencia_01_aguinaldo.parquet"
DESTINO = c.DATOS / "competencia_01_calendario.parquet"
COLS = ["cliente_vip", "Visa_delinquency"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--forzar", action="store_true"); a = ap.parse_args()
    if DESTINO.exists() and not a.forzar: raise SystemExit(f"{DESTINO.name} ya existe; --forzar para pisarlo")
    t0 = time.time(); con = fe.conectar(); con.execute("set memory_limit = '4GB'; set threads = 2")
    derivadas = [f"{x}__{s}" for x in COLS for s in ("lag1", "delta1", "lag2", "delta2")]
    con.execute(f"create or replace view base as select * exclude ({', '.join(COLS + derivadas)}) from read_parquet('{ORIGEN}')")
    con.execute(f"create or replace view crudo as select {fe.ID}, {fe.MES}, cliente_vip, Visa_delinquency, Visa_Finiciomora from read_parquet('{c.DATOS / 'competencia_01.parquet'}')")
    corr = (f"case when {fe.MES} = 202106 then coalesce(lag(cliente_vip) over (partition by {fe.ID} order by {fe.MES}), cliente_vip) else cliente_vip end as cliente_vip, "
            f"case when {fe.MES} in (202105, 202108) and Visa_Finiciomora = 0 then 0 else Visa_delinquency end as Visa_delinquency")
    lags = []
    for x in COLS:
        lags += [f"lag({x}, 1) over historia :: FLOAT as {x}__lag1", f"({x} - lag({x}, 1) over historia) :: FLOAT as {x}__delta1",
                 f"lag({x}, 2) over historia :: FLOAT as {x}__lag2", f"({x} - lag({x}, 2) over historia) :: FLOAT as {x}__delta2"]
    sql = (f"with d as (select {fe.ID}, {fe.MES}, {corr} from crudo),"
           f"\nj as (select b.*, d.cliente_vip, d.Visa_delinquency from base b join d using ({fe.ID}, {fe.MES}))"
           "\nselect *" + fe._fragmento(lags) + "\nfrom j" + fe.clausula_ventana("historia"))
    con.execute(f"copy ({sql}) to '{DESTINO}' (format PARQUET, COMPRESSION ZSTD)")
    chk = con.execute(f"select {fe.MES}, sum(cliente_vip), sum(Visa_delinquency) from read_parquet('{DESTINO}') group by 1 order by 1").fetchall()
    n, k = con.execute(f"select count(*), (select count(*) from (describe select * from read_parquet('{DESTINO}'))) from read_parquet('{DESTINO}')").fetchone()
    print(f"  {DESTINO.name}: {n:,} filas x {k} columnas | vip y delinquency por mes: {chk} [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
