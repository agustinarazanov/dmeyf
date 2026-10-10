"""FE historico v3: lo que falto, con el foco en lo que NO envejece.

La reunion de hoy marco que el FE historico rinde, y nuestros propios folds lo
confirman una vez leidos BIEN (fold por fold, no promediados): v2 con ventanas
fijas da +17,1 M en 202106 (8/10, p=0,027) y productos +8,1 M. En 202105, nada.
Misma firma que sacar los prestamos y que deflactar: sirve donde el drift ocurrio.

Lo que faltaba construir:
  1. AGREGADOS MOVILES. `fe_panel.sql_ventana` existe y nunca se uso: solo usamos
     lag, delta, rank y slope.
  2. RATIO CONTRA LA PROPIA HISTORIA -- lo mas importante. x / avg(x en los 3
     meses previos). Es inmune a la inflacion (numerador y denominador se inflan
     juntos) e inmune a la escala del cliente (el que mueve un millon y el que
     mueve diez mil dan ~1,0 si estan estables). Es literalmente "¿este cliente
     transacciona menos que EL MISMO?", la hipotesis central de la materia.
  3. ACELERACION: la diferencia de la diferencia. Distingue "viene cayendo
     despacio" de "se derrumbo este mes".
  4. VOLATILIDAD: desvio sobre la ventana, normalizado por el nivel propio.

Todas las ventanas son FIJAS de 3 meses: con `unbounded` la feature no significa
lo mismo en 202103 que en 202108 (la cobertura se triplica sola).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import fe_panel as fe

DESTINO = "data/competencia_01_hist.parquet"

CAMPOS = ["ctrx_quarter", "mcuentas_saldo", "mcaja_ahorro", "mpayroll", "cpayroll_trx",
          "mtarjeta_visa_consumo", "ctarjeta_visa_transacciones", "mtarjeta_master_consumo",
          "chomebanking_transacciones", "cmobile_app_trx", "ccajas_transacciones",
          "catm_trx", "mcomisiones", "mcomisiones_mantenimiento", "mactivos_margen",
          "mpasivos_margen", "cproductos", "mtransferencias_emitidas",
          "mtransferencias_recibidas", "ctarjeta_debito_transacciones"]


def main() -> None:
    t0 = time.time()
    raiz = Path(__file__).resolve().parent.parent
    con = fe.conectar()
    con.execute(f"""create or replace view crudo as
        select * exclude ({", ".join(fe.COLUMNAS_DE_BAJA)})
        from read_parquet('{raiz / 'data/competencia_01.parquet'}')""")

    # paso 1: los agregados de los 3 meses PREVIOS (sin el actual) y el lag
    prev = ", ".join(
        f"avg({c}) over v3prev as {c}__avg3, "
        f"stddev({c}) over v3prev as {c}__sd3, "
        f"min({c}) over v3prev as {c}__min3, "
        f"max({c}) over v3prev as {c}__max3, "
        f"lag({c},1) over v as {c}__l1, lag({c},2) over v as {c}__l2"
        for c in CAMPOS)
    con.execute(f"""create or replace table paso1 as select *, {prev} from crudo
        window v as (partition by numero_de_cliente order by foto_mes),
          v3prev as (partition by numero_de_cliente order by foto_mes
                     rows between 3 preceding and 1 preceding)""")

    # paso 2: lo que de verdad importa -- cocientes contra la propia historia
    nuevas = ", ".join(
        # ¿esta por debajo de SI MISMO? inmune a inflacion y a escala del cliente
        f"ratio_seguro({c}, {c}__avg3) as {c}__vs_propio, "
        # aceleracion: distingue caida lenta de derrumbe
        f"({c} - {c}__l1) - ({c}__l1 - {c}__l2) as {c}__acel, "
        # volatilidad relativa al nivel propio
        f"ratio_seguro({c}__sd3, {c}__avg3) as {c}__volrel, "
        # cuanto cayo desde su propio maximo recordado
        f"ratio_seguro({c}, {c}__max3) as {c}__vs_max"
        for c in CAMPOS)
    aux = [f"{c}__{s}" for c in CAMPOS for s in ("avg3", "sd3", "min3", "max3", "l1", "l2")]
    con.execute(f"""create or replace table hist as
        select * exclude ({", ".join(aux)}), {nuevas} from paso1""")

    n0 = con.sql("select count(*) from (describe crudo)").fetchone()[0]
    n1 = con.sql("select count(*) from (describe hist)").fetchone()[0]
    print(f"{n0} -> {n1} columnas (+{n1-n0}) sobre {len(CAMPOS)} campos x 4 construcciones")

    base = con.sql("""select avg((clase_ternaria='BAJA+2')::int) from hist
                      where clase_ternaria is not null""").fetchone()[0]
    print(f"\n{'feature':34} {'cob.08':>7} {'lift':>6}")
    for col in ("ctrx_quarter__vs_propio < 0.5", "ctrx_quarter__vs_max < 0.4",
                "ctrx_quarter__acel < -20", "mcuentas_saldo__vs_propio < 0",
                "chomebanking_transacciones__vs_propio < 0.5", "cproductos__vs_max < 1",
                "mpayroll__vs_propio < 0.5"):
        r = con.sql(f"""select
            (select round(100.0*avg(({col})::int),1) from hist where foto_mes=202108) c,
            round(avg((clase_ternaria='BAJA+2')::int)/{base},2) l
            from hist where clase_ternaria is not null and {col}""").fetchone()
        print(f"  {col:32} {r[0]:>6}% {r[1]:>6}")
    con.execute(f"copy hist to '{raiz / DESTINO}' (format parquet, compression zstd)")
    con.close()
    print(f"\nescrito {DESTINO} en {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
