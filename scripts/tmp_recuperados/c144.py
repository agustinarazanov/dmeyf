"""La cadena de 4 eslabones con horizonte 1 (BAJA+1).
No sirve para elegir el modelo final (distancia 1, no 2), SI sirve para ver si el
cambio de regimen es tendencia o evento aislado: cuatro cohortes DISTINTAS de gente.
Comparo base contra 'sin prestamos' (el cambio tuneado a junio) en cada eslabon."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c144_cadena"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
d=c.cargar("competencia_01.parquet")
PRES=["mprestamos_personales","cprestamos_personales"]
SETS={"base":c.columnas_predictoras(d),
      "sin_prestamos":[x for x in c.columnas_predictoras(d) if x not in PRES]}
CADENA=[([202103],202104,202105),([202103,202104],202105,202106),
        ([202103,202104,202105],202106,202107),([202103,202104,202105,202106],202107,202108)]
print("horizonte 1 (BAJA+1), 10 semillas, captura a 10.000\n")
print(f"  {'eslabon':34s} {'base':>7} {'sin pr.':>8} {'dif':>7}  semillas      p")
for tr,va,se_va in CADENA:
    v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+1"); n=es.sum()
    out={}
    for nom,pred in SETS.items():
        X,y,w=c.preparar(d,tr,"pesos",0.25)   # peso: BAJA+2 del mes son "se va en 2", aun asi informan
        X=X[pred]
        out[nom]=np.array([100*es[np.argsort(-c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,nom+str(len(pred)),250,s)}")
            .predict(v[pred]))[:10000]].sum()/n for s in sem])
    a,b=out["sin_prestamos"],out["base"]
    etq=f"train{tr[0]%100:02d}-{tr[-1]%100:02d} -> val{va%100:02d} (se va {se_va%100:02d})"
    print(f"  {etq:34s} {b.mean():6.2f}% {a.mean():7.2f}% {a.mean()-b.mean():+6.2f}  "
          f"{int((a>b).sum()):2d}/10  p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
