#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0910_tarde.log; do sleep 60; done
echo "== $(date '+%H:%M') c263: solo las 392 columnas con el 99% de la ganancia, 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c263_gain99 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --semillas 5 --cortes 14000 --columnas experimentos/c241_rondas2000/columnas_gain99_392.txt
echo "== $(date '+%H:%M') fin"
