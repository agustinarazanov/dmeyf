#!/bin/zsh
# Esta noche: reproducir de punta a punta los 20 modelos de 2.000 rondas (candidato) y comparar el SHA con c241 ens20.
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "listo" experimentos/c270_h1_0307.log; do sleep 60; done
echo "== $(date '+%H:%M') reproducir --rondas 2000"
python reproducir.py --rondas 2000 --salida experimentos/reproduccion/entrega_2000.csv
echo "== $(date '+%H:%M') fin"
