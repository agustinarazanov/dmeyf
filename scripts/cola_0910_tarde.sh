#!/bin/zsh
# Despues de c263: barrido de feature_fraction (nunca barrido; Abregu +6,8 con 0,40) y miembro de horizonte 1 (review monday-45).
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0910_c262.log; do sleep 60; done
echo "== $(date '+%H:%M') c264: ff 0,4, 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c264_ff04 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --ff 0.4 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') c265: ff 0,3"
python scripts/c210_variante.py --nombre c265_ff03 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --ff 0.3 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') c270: horizonte 1 (target BAJA+1, meses 03-07), 2.000 rondas, 5 semillas"
python scripts/c210_variante.py --nombre c270_h1_0307 --dataset competencia_01_lags12.parquet --params receta --rondas 2000 --target baja1 --meses 202103 202104 202105 202106 202107 --semillas 5 --cortes 14000
echo "== $(date '+%H:%M') fin"
