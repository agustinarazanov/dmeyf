"""¿Que instrumento local predice al publico? Caso de prueba: el FE completo.
VERDAD CONOCIDA: contra el publico, al corte 9.000 con 20 archivos, el FE perdio
-3.12 M publicos (-5.2 sigma) = ~-12.5 M mensuales.
  fold 202105  -> dice +1.3 M   (se equivoca de signo)
  fold 202106  -> dice +15.5 M  (se equivoca de signo, y feo)
  cadena esl.4 -> ENTRENA CON 4 MESES como la entrega y valida la cohorte de agosto
Si la cadena acierta el signo, es el instrumento que hay que usar de ahora en mas."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c166_cadena_fe"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
TR=[202103,202104,202105,202106]      # los 4 meses, como la entrega
out={}
for nom,ds in [("base","competencia_01.parquet"),("fe228","competencia_01_fe.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    X,y,w=c.preparar(d,TR,"pesos",0.25); X=X[pred]
    v=d[d[c.fe.MES]==202107]
    es=(v["clase_ternaria"].to_numpy()=="BAJA+1")      # de SU propio v
    n=es.sum()
    out[nom]=np.array([100*es[np.argsort(-c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
        f"{nom}_{s}_{c.clave(P,TR,'pesos',0.25,nom+str(len(pred)),250,s)}")
        .predict(v[pred]))[:9000]].sum()/n for s in sem])
    print(f"  {nom:6s} ({len(pred)} col)  captura {out[nom].mean():5.2f}%   [{time.time()-t0:.0f}s]")
a,b=out["fe228"],out["base"]
print(f"\ncadena eslabon 4 (train 4 meses -> valida 202107, cohorte que se va en agosto):")
print(f"  fe228 contra base: {a.mean()-b.mean():+5.2f} puntos de captura  "
      f"{int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
print(f"\n  el publico dijo: fe228 PIERDE (-5.2 sigma)")
print(f"  la cadena dice:  fe228 {'PIERDE -> ACIERTA el signo' if a.mean()<b.mean() else 'GANA -> se equivoca igual que los folds'}")
print(f"\nlisto en {time.time()-t0:.0f}s")
