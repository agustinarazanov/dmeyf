#!/bin/zsh
# 9-oct manana: las 8 mediciones planificadas sobre lo entrenado de noche. Todas a 14.000.
# Referencias: c201 5 semillas 104,16 | c201 20 archivos 103,00 | c201 ens20 104,17 | c241 (2.000 rondas) 5 semillas 105,51 | c241 ens20 103,79
cd ~/Documents/segundo-cuatrimestre/competencia-1
export PYENV_VERSION=zulip
E() { echo "== $(date '+%H:%M') $1"; }
E "1/8 c241 20 archivos"
python enviar.py --experimento c241_rondas2000 --submit c241_rondas2000_20_14000 --corte 14000 --enviar \
  --archivos "experimentos/c241_rondas2000/envios_14000/c241_rondas2000_14000_s*.csv" \
  --hipotesis "las 5 primeras semillas de c201 estan +1,2 sobre su media de 20 (104,16 vs 103,00). Si en c241 pasa lo mismo, el +1,35 de 2.000 rondas es suerte de semillas: la media de 20 de c241 contra la media de 20 de c201 (103,00) es la comparacion justa, mismas 20 semillas" \
  --delta-contra c201_lags20_14000 --delta-desc "20 archivos contra 20 archivos: 2.000 vs 1.000 rondas (103,00)"
E "2/8 c246 3000 rondas"
python enviar.py --experimento c246_rondas3000 --submit c246_rondas3000_14000 --corte 14000 --enviar \
  --archivos "experimentos/c246_rondas3000/envios_14000/c246_rondas3000_14000_s*.csv" \
  --hipotesis "la curva de rondas con lr 0,005: 1.000 -> 104,16, 2.000 -> 105,51 (5 semillas). Si 3.000 sigue subiendo, el modelo estaba corto; si baja, 2.000 es el techo. Esta vez verificado en el log: 3000 rondas" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo 3.000 rondas (105,51)" \
  --descripcion "receta Denicolay sobre lags12, pesos 0,25, 03-06, 3.000 rondas, 5 semillas"
E "3/8 c251 comision deflactada"
python enviar.py --experimento c251_comdefl2000 --submit c251_comdefl2000_14000 --corte 14000 --enviar \
  --archivos "experimentos/c251_comdefl2000/envios_14000/c251_comdefl2000_14000_s*.csv" \
  --hipotesis "mcomisiones_mantenimiento tiene el drift mas grande del dataset (PSI 0,55/0,66 en 07/08) por cambio de tarifa +22%, no de conducta. Dividir las dos comisiones por la mediana mensual de los que pagan (tarifa estandar = 1,0 todos los meses) saca el umbral aprendido en pesos viejos. Deflactar TODO perdio (c127); esto toca solo 2 de 150" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo comisiones deflactadas (y sus lags) (105,51)" \
  --descripcion "lags12 con mcomisiones y mcomisiones_mantenimiento / mediana mensual de pagadores; receta, 2.000 rondas, 5 semillas"
E "4/8 c253 aguinaldo"
python enviar.py --experimento c253_aguinaldo2000 --submit c253_aguinaldo2000_14000 --corte 14000 --enviar \
  --archivos "experimentos/c253_aguinaldo2000/envios_14000/c253_aguinaldo2000_14000_s*.csv" \
  --hipotesis "el SAC de junio deja en agosto mpayroll__delta2 = -35% y cpayroll_trx__delta2 = -1 para todo asalariado: el modelo entrenado en 03-06 nunca vio esa firma en el mes a predecir y la lee como caida de sueldo. Normalizar junio (/1,5 y -1 trx) solo donde hay evidencia de SAC (2+ acreditaciones o salto > 30%) borra el artefacto de calendario" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo junio con SAC normalizado (105,51)" \
  --descripcion "lags12 con mpayroll/mpayroll2 /1,5 y cpayroll_trx -1 en 202106 donde hay SAC; receta, 2.000 rondas, 5 semillas"
E "5/8 c255 calendario"
python enviar.py --experimento c255_calendario2000 --submit c255_calendario2000_14000 --corte 14000 --enviar \
  --archivos "experimentos/c255_calendario2000/envios_14000/c255_calendario2000_14000_s*.csv" \
  --hipotesis "las tres reparaciones de calendario juntas: SAC de junio (c253) + cliente_vip de junio tomado de mayo (junio marca ~0 vip, artefacto) + Visa_delinquency = 0 donde Visa_Finiciomora = 0 en 05/08 (ceros inyectados). Si c253 solo ya gana, esto mide si vip y delinquency suman o restan" \
  --delta-contra c253_aguinaldo2000_14000 --delta-desc "c253 + vip + delinquency, contra c253 solo" \
  --descripcion "lags12 con SAC normalizado + vip de junio por coalesce con mayo + delinquency coherente con Finiciomora; receta, 2.000 rondas, 5 semillas"
E "6/8 c247 hessian x0,5"
python enviar.py --experimento c247_hess05 --submit c247_hess05_14000 --corte 14000 --enviar \
  --archivos "experimentos/c247_hess05/envios_14000/c247_hess05_14000_s*.csv" \
  --hipotesis "min_sum_hessian_in_leaf 25,65 es calibracion de Ramirez (12,79 x filas / 326.184), no de la catedra; Denicolay busca en 1e-6..0,1 absoluto. Bajarlo a 12,8 deja hojas mas chicas: c222 dijo que el hessiano chico de Optuna no transfirio a mayo, pero eso era 1e-6..8e-6 por fila, dos ordenes mas abajo" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo hessian x0,5 (105,51)" \
  --descripcion "receta con min_sum_hessian_in_leaf x0,5; lags12, 2.000 rondas, 5 semillas"
E "7/8 c248 hessian x2"
python enviar.py --experimento c248_hess2 --submit c248_hess2_14000 --corte 14000 --enviar \
  --archivos "experimentos/c248_hess2/envios_14000/c248_hess2_14000_s*.csv" \
  --hipotesis "el otro lado del barrido: hessian 51,3 regulariza mas (hojas mas pobladas). Si el modelo pierde ~50 M por drift entre meses, mas regularizacion deberia transferir mejor a agosto" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo hessian x2 (105,51)" \
  --descripcion "receta con min_sum_hessian_in_leaf x2; lags12, 2.000 rondas, 5 semillas"
E "8/8 c249 127 hojas"
python enviar.py --experimento c249_hojas127 --submit c249_hojas127_14000 --corte 14000 --enviar \
  --archivos "experimentos/c249_hojas127/envios_14000/c249_hojas127_14000_s*.csv" \
  --hipotesis "83 hojas es una de 12 configs al azar de la catedra; con 750 columnas y 2.000 rondas, 127 hojas da arboles mas expresivos. Mas rondas ayudo (capacidad faltante); mas hojas es la otra forma de capacidad" \
  --delta-contra c241_rondas2000_14000 --delta-desc "identico a c241 salvo 127 hojas (105,51)" \
  --descripcion "receta con num_leaves 127; lags12, 2.000 rondas, 5 semillas"
E "fin tanda"
