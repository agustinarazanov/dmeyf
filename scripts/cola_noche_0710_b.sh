#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=facultad
until grep -q "fin de la cola" experimentos/cola_noche_0710.log; do sleep 60; done
echo "== $(date '+%H:%M') cola b: captura fold B"
python scripts/c216_captura_foldB.py --semillas 3
echo "== $(date '+%H:%M') fin cola b"
