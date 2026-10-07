"""TAREA 2: acercar los datos a agosto. ¿Conviene entrenar TAMBIEN con 202107?
202107 tiene BAJA+1 etiquetado (1.103). Agregarlo acerca el train un mes, y medimos
que el modelo pierde ~3 puntos de captura por mes de distancia. El costo: ~1.100 de
sus "CONTINUA" son en realidad los BAJA+2 de julio (ausentes en 202109), o sea ruido
de etiqueta sobre la clase que mas nos importa. Hay que medirlo, no asumirlo.

Se mide en el FOLD C LIMPIO (gap 2): train [...] -> valida 202107 con BAJA+1.
Para no usar 202107 en train y validacion a la vez, el analogo es:
   train [03,04] vs train [03,04,05-solo-BAJA+1]  ->  valida 202106 BAJA+2 (fold B)
donde 202105 entra con BAJA+1 como positivo (peso 0,25) y sus BAJA+2 ESCONDIDOS como 0,
que es exactamente la situacion de 202107 en la entrega real.
"""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, lightgbm as lgb, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); K=9000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1,"verbose":-1}
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
CL=d["clase_ternaria"].to_numpy(); MES=d[c.fe.MES].to_numpy()

def armar(meses_full, mes_parcial=None):
    """meses_full: con BAJA+1 y BAJA+2 normales.
       mes_parcial: solo BAJA+1 es positivo; sus BAJA+2 se esconden como CONTINUA (0),
       que es lo que pasa con 202107 en la entrega real."""
    m = np.isin(MES, meses_full)
    if mes_parcial: m = m | (MES == mes_parcial)
    cl = CL[m]; mes = MES[m]
    y = np.isin(cl, ["BAJA+1","BAJA+2"]).astype("int8")
    w = np.where(cl=="BAJA+1", 0.25, 1.0)
    if mes_parcial is not None:
        esc = (mes == mes_parcial) & (cl == "BAJA+2")      # el futuro que NO se ve
        y[esc] = 0; w[esc] = 1.0
    return d.loc[m, pred], y, w, int(esc.sum()) if mes_parcial else 0

v = d[d[c.fe.MES]==202106]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")
CAR=c.EXPERIMENTOS/"c168_mas_reciente"; CAR.mkdir(parents=True,exist_ok=True)
res={}
for nom,(mf,mp) in {"train [03,04]": ([202103,202104],None),
                    "train [03,04] + 05 parcial": ([202103,202104],202105)}.items():
    X,y,w,esc = armar(mf,mp)
    g=[]
    for s in sem:
        mo=lgb.train({**P,"seed":s}, lgb.Dataset(X,label=y,weight=w), num_boost_round=250)
        g.append(c.ganancia_acumulada(mo.predict(v[pred]),es)[K-1]/1e6)
    res[nom]=np.array(g)
    print(f"  {nom:30s} {len(X):,} filas, {int(y.sum()):,} positivos"
          f"{f', {esc:,} BAJA+2 escondidos' if esc else ''}  ->  {res[nom].mean():6.1f}M   [{time.time()-t0:.0f}s]")
a,b=res["train [03,04] + 05 parcial"],res["train [03,04]"]
print(f"\n  agregar el mes mas reciente con solo BAJA+1: {a.mean()-b.mean():+5.1f}M  "
      f"{int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
print(f"\nlisto en {time.time()-t0:.0f}s")
