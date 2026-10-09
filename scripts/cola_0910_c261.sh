#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "listo" experimentos/c260_props2000.log; do sleep 60; done
echo "== $(date '+%H:%M') c261 agregados, receta 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c261_agregados2000 --dataset competencia_01_agregados.parquet --params receta --rondas 2000 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') fin"
