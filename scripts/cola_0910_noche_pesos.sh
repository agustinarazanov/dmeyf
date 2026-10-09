#!/bin/zsh
# Curva dosis-respuesta del miembro de horizonte 1, pareada por semilla (5 archivos vs los 5 de c241 = peso 0):
# peso 0 -> 105,51 | 0,15 | 0,30 -> 105,75 | 0,50 | 1,00 (h1 solo). Cuatro submits con lo que queda del cupo del 9-oct.
cd /Users/agustina/Documents/segundo-cuatrimestre/competencia-1
PE="env PYENV_VERSION=zulip python"; P="env PYENV_VERSION=facultad python"
H="miembro de horizonte 1 (c270: target BAJA+1 solo, train 03-07) mezclado por rango con c241 semilla a semilla. Curva de peso pareada por semilla: 0 -> 105,51; 0,30 -> 105,75. Si h1 aporta senal, la curva sube y despues cae; si no, es plana dentro del ruido"
echo "== $(date +%H:%M) inicio"
$=PE enviar.py --reenviar --experimento c270_h1_0307 --submit c270_h1_0307_14000 --archivos "experimentos/c270_h1_0307/envios_14000/c270_h1_0307_14000_s*.csv" \
  --hipotesis "peso 1,0 de la curva: el modelo de horizonte 1 SOLO (target BAJA+1, train 03-07, 5.103 positivos, 2.000 rondas). Mide cuanto vale por si mismo un modelo que nunca vio BAJA+2 como etiqueta; se espera por debajo de c241 (BAJA+1 y BAJA+2 comparten features pero el peso 0,25 ya los mezcla)" \
  --delta-contra c241_rondas2000_14000 --delta-desc "mismas 5 semillas: h1 solo vs c241 solo (105,51)" --corte 14000 --enviar
for p in 0.15 0.5 0.7; do
  tag=$(printf "p%02d" $(( ${p} * 100 )))
  $P scripts/c271_mezcla_h1.py --peso $p
  $=PE enviar.py --experimento c271_mezcla_h1_$tag --submit c271_mezcla_h1_${tag}_14000 --archivos "experimentos/c271_mezcla_h1_$tag/envios_14000/c271_mezcla_h1_${tag}_14000_s*.csv" \
    --hipotesis "peso $p de la curva: $H" --delta-contra c271_mezcla_h1_p30_14000 --delta-desc "mismas 5 semillas: peso $p vs peso 0,3 (105,75) y peso 0 (105,51)" --corte 14000 --enviar
done
echo "== $(date +%H:%M) fin"
