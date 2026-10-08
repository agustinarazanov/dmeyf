#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche.log; do sleep 60; done
echo "== $(date '+%H:%M') c242: peso_baja1 0,5"
python scripts/c210_variante.py --nombre c242_peso05 --dataset competencia_01_lags12.parquet --params receta --target pesos --peso-baja1 0.5 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') c243: baja12 (peso 1,0)"
python scripts/c210_variante.py --nombre c243_baja12 --dataset competencia_01_lags12.parquet --params receta --target baja12 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') fin"
