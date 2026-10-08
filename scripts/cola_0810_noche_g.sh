#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche_f.log; do sleep 60; done
echo "== $(date '+%H:%M') c246: receta con 3.000 rondas DE VERDAD, 5 semillas"
python scripts/c210_variante.py --nombre c246_rondas3000 --dataset competencia_01_lags12.parquet --params receta --rondas 3000 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') fin"
