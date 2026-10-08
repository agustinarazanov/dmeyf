#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "== .* fin" experimentos/cola_0810_noche_b.log; do sleep 60; done
echo "== $(date '+%H:%M') c244: receta con 3.000 rondas, 5 semillas"
sed -e 's/c241_rondas2000/c244_rondas3000/g' -e 's/nbr = 2000/nbr = 3000/' -e 's/receta2000_/receta3000_/' scripts/cola_0810_noche.sh | sed -n '/^python - <<.PY.$/,/^PY$/p' > /tmp/c244_run.py.sh
zsh /tmp/c244_run.py.sh
echo "== $(date '+%H:%M') c241 extra: 20 semillas de 2.000 rondas"
sed -e 's/semillas=5/semillas=20/' scripts/cola_0810_noche.sh | sed -n '/^python - <<.PY.$/,/^PY$/p' > /tmp/c241_20.py.sh
zsh /tmp/c241_20.py.sh
echo "== $(date '+%H:%M') fin"
