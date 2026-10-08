#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "listo" experimentos/c221_ensamble.log; do sleep 60; done
echo "== $(date '+%H:%M') c231: dataset reparado + transiciones, receta, 5 semillas"
python scripts/c210_variante.py --nombre c231_rep --dataset competencia_01_rep.parquet --params receta --semillas 5 --cortes 14000 10000
echo "== $(date '+%H:%M') fin"
