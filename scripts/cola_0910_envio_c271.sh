#!/bin/zsh
cd ~/Documents/segundo-cuatrimestre/competencia-1
until grep -q "listo" experimentos/c270_h1_0307.log; do sleep 60; done
echo "== $(date '+%H:%M') c271 mezcla h1 peso 0,3"
PYENV_VERSION=facultad python scripts/c271_mezcla_h1.py --peso 0.3 --cortes 14000
PYENV_VERSION=zulip python enviar.py --experimento c271_mezcla_h1_p30 --submit c271_mezcla_h1_p30_14000 --corte 14000 --enviar \
  --archivos "experimentos/c271_mezcla_h1_p30/envios_14000/c271_mezcla_h1_p30_14000_s*.csv" \
  --hipotesis "miembro de horizonte 1 (target BAJA+1 solo, train 03-07: julio tiene BAJA+1 completo y 5.103 positivos exactos, sin BAJA+2 escondidos como en c240) mezclado por rango con c241 semilla a semilla, peso 0,3. BAJA+1 y BAJA+2 son casi indistinguibles por features (AUC 0,62): su ranking es senal de 'se va pronto' con el mes mas fresco. 5 archivos contra los 5 de c241 con la misma suerte de semillas" \
  --delta-contra c241_rondas2000_14000 --delta-desc "mismas 5 semillas: 0,7 x rank(c241) + 0,3 x rank(h1) vs c241 solo (105,51 / media 20: 103,85)" \
  --descripcion "rank blend por semilla de c241 (pesos 0,25, 03-06, 2.000 rondas) y c270 (baja1, 03-07, 2.000 rondas), peso 0,3; corte 14.000"
echo "== $(date '+%H:%M') fin"
