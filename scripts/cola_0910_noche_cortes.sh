#!/bin/zsh
# Curva fina de corte con 20 archivos (sd de la media ~0,45) para los dos candidatos de entrega, mismas 20 semillas.
# c241 (2.000 rondas) tiene 13.000 -> 103,30 | 14.000 -> 103,85 | 15.000 -> 102,81; se agregan 13.500, 14.500, 12.000, 16.000.
# c201 (1.000 rondas) tiene 10.000 -> 100,03 | 14.000 -> 103,00; se agregan 13.000, 13.500, 14.500, 15.000.
cd /Users/agustina/Documents/segundo-cuatrimestre/competencia-1
E() { echo "== $(date '+%H:%M') $1"; }
E "cortes c241"
env PYENV_VERSION=facultad python scripts/c201_cortar.py --experimento c241_rondas2000 --cortes 13500 14500 12000 16000
E "cortes c201"
env PYENV_VERSION=facultad python scripts/c201_cortar.py --cortes 13000 13500 14500 15000
for K in 13500 14500 12000 16000; do
  E "c241 @ $K"
  env PYENV_VERSION=zulip python enviar.py --experimento c241_rondas2000 --submit c241_rondas2000_20_$K --corte $K --enviar \
    --archivos "experimentos/c241_rondas2000/envios_$K/c241_rondas2000_${K}_s*.csv" \
    --hipotesis "curva fina de corte del candidato de 2.000 rondas, 20 archivos (sd de la media ~0,45): 13.000 -> 103,30, 14.000 -> 103,85, 15.000 -> 102,81. Corte $K para ubicar el pico entre 13.000 y 15.000 (13.500/14.500) y anclar la pendiente afuera (12.000/16.000)" \
    --delta-contra c241_rondas2000_20_14000 --delta-desc "20 vs 20, mismas semillas: corte $K vs 14.000 (103,85)"
done
for K in 13000 13500 14500 15000; do
  E "c201 @ $K"
  env PYENV_VERSION=zulip python enviar.py --experimento c201_receta_lags --submit c201_lags20_$K --corte $K --enviar \
    --archivos "experimentos/c201_receta_lags/envios_$K/c201_receta_lags_${K}_s*.csv" \
    --hipotesis "curva fina de corte del candidato de 1.000 rondas, 20 archivos: 10.000 -> 100,03, 14.000 -> 103,00. Corte $K: si el pico esta en 14.000 tambien para 1.000 rondas, el corte de la entrega no depende de cual de los dos candidatos se elija" \
    --delta-contra c201_lags20_14000 --delta-desc "20 vs 20, mismas semillas: corte $K vs 14.000 (103,00)"
done
E "fin"
