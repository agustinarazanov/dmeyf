#!/bin/zsh
# Cuando termina la cola c (20 semillas de 2.000 rondas), manda el ensamble de 20 a 14.000.
# Hipotesis planificada el 8-oct: "el ensamble de 20 semillas de la ganadora contra 104,17".
cd ~/Documents/segundo-cuatrimestre/competencia-1
until grep -q "== .* fin" experimentos/cola_0810_noche_c.log; do sleep 120; done
ARCH=experimentos/c241_rondas2000/envios_14000/c241_rondas2000_14000_ens20.csv
until [ -f "$ARCH" ]; do sleep 30; done
echo "== $(date '+%H:%M') envio c241 ens20 @14.000"
PYENV_VERSION=zulip python enviar.py --experimento c241_rondas2000 --submit c241_rondas2000_ens20_14000 \
  --archivos "$ARCH" --corte 14000 --enviar \
  --hipotesis "2.000 rondas con 20 semillas: el ensamble de 20 de la receta a 1.000 rondas dio 104,17 (un archivo); 5 semillas sueltas de 2.000 rondas dieron 105,51 en promedio. Si el ensamble de 20 a 2.000 rondas supera 104,17 por mas de ~1,5, las rondas se adoptan para la entrega" \
  --delta-contra c201_lags_ens20_14000 --delta-desc "un archivo contra un archivo: ens20 a 2.000 rondas vs ens20 a 1.000 rondas (104,17)" \
  --descripcion "receta Denicolay sobre lags12, pesos 0,25, 03-06, 2.000 rondas, ensamble por rank de 20 semillas, corte 14.000"
echo "== $(date '+%H:%M') fin envio"
