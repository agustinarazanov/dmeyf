"""Exporta el ledger de Postgres a ledger/*.csv para que la trazabilidad viaje con el repo.

    python ledger_export.py

Cuatro tablas (experimento, submit, archivo, resultado) y la vista v_submits, ordenadas por id para
que el diff de git sea legible. Se corre despues de cada submit o al cerrar el dia.
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import registro as r  # noqa: E402

DESTINO = RAIZ / "ledger"


def main() -> None:
    DESTINO.mkdir(exist_ok=True)
    with r.conectar() as con:
        for tabla in ("experimento", "submit", "archivo", "resultado"):
            df = pd.DataFrame(con.execute(f"select * from {tabla} order by id").fetchall())
            df.to_csv(DESTINO / f"{tabla}.csv", index=False)
            print(f"  {tabla:12s} {len(df):4d} filas")
        v = pd.DataFrame(con.execute(
            "select * from v_submits order by public_gain_mean desc nulls last, creado_en").fetchall())
        v.to_csv(DESTINO / "v_submits.csv", index=False)
        print(f"  v_submits    {len(v):4d} filas")


if __name__ == "__main__":
    main()
