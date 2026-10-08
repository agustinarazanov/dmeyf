#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche_d.log; do sleep 60; done
echo "== $(date '+%H:%M') c253: aguinaldo corregido, receta 2.000 rondas, 5 semillas"
sed -e 's/c241_rondas2000/c253_aguinaldo2000/g' -e 's/competencia_01_lags12.parquet/competencia_01_aguinaldo.parquet/g' -e 's/receta2000_/aguinaldo2000_/' scripts/cola_0810_noche.sh | sed -n '/^python - <<.PY.$/,/^PY$/p' > /tmp/c253_run.py.sh
zsh /tmp/c253_run.py.sh
echo "== $(date '+%H:%M') fin"
