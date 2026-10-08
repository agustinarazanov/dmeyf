#!/bin/zsh
# barrido chico del regularizador y las hojas a 2.000 rondas (code review de monday-3d, punto 1)
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche_g.log; do sleep 60; done
echo "== $(date '+%H:%M') c247: hessian x0,5, 2.000 rondas"
python scripts/c210_variante.py --nombre c247_hess05 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --hessian 0.5 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') c248: hessian x2, 2.000 rondas"
python scripts/c210_variante.py --nombre c248_hess2 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --hessian 2.0 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') c249: 127 hojas, 2.000 rondas"
python scripts/c210_variante.py --nombre c249_hojas127 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --hojas 127 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') fin"
