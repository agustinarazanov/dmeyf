#!/bin/zsh
# Cola de la noche del 7 al 8 de octubre. Arranca cuando termina c203 (pid 96778).
# Cada corrida cachea por semilla, asi que si se corta se retoma con el mismo comando.
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
while kill -0 96778 2>/dev/null; do sleep 30; done
echo "== $(date '+%H:%M') arranca la cola"
python scripts/c210_variante.py --nombre c211_lags_z701   --dataset competencia_01_lags12.parquet --params z701   --semillas 5
python scripts/c210_variante.py --nombre c212_base_receta --dataset competencia_01.parquet        --params receta --semillas 5
python scripts/c210_variante.py --nombre c213_lags_baja2  --dataset competencia_01_lags12.parquet --params receta --target baja2 --semillas 5
python scripts/c210_variante.py --nombre c214_lags_sin03  --dataset competencia_01_lags12.parquet --params receta --meses 202104 202105 202106 --semillas 5
python scripts/c202_lags_rank.py
python scripts/c210_variante.py --nombre c215_lagsrank    --dataset competencia_01_lagsrank.parquet --params receta --semillas 5
echo "== $(date '+%H:%M') fin de la cola"
