#!/bin/zsh
# Reduccion de dimensionalidad por importancia (20 modelos de c241): sin las 52 nunca usadas; y solo las 393 que juntan el 99% de la ganancia.
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0910_c261.log; do sleep 60; done
echo "== $(date '+%H:%M') c262: lags12 sin las 52 columnas que ningun modelo uso (698), 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c262_sin_muertas --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --semillas 5 --cortes 14000 --columnas experimentos/c241_rondas2000/columnas_usadas_698.txt
echo "== $(date '+%H:%M') c263: solo las 392 columnas con el 99% de la ganancia, 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c263_gain99 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --semillas 5 --cortes 14000 --columnas experimentos/c241_rondas2000/columnas_gain99_392.txt
echo "== $(date '+%H:%M') fin"
