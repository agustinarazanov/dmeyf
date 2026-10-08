#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche_e.log; do sleep 60; done
echo "== $(date '+%H:%M') c255: calendario (aguinaldo + vip + delinquency), receta 2.000 rondas, 5 semillas"
sed -e 's/c241_rondas2000/c255_calendario2000/g' -e 's/competencia_01_lags12.parquet/competencia_01_calendario.parquet/g' -e 's/receta2000_/calendario2000_/' scripts/cola_0810_noche.sh | sed -n '/^python - <<.PY.$/,/^PY$/p' > /tmp/c255_run.py.sh
zsh /tmp/c255_run.py.sh
echo "== $(date '+%H:%M') fin"
