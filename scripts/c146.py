"""MISMA validacion (val 202107, BAJA+1 = cohorte que se va en agosto), misma gente.
Lo unico que cambia: si junio esta en el entrenamiento. Si sacar prestamos ayuda
entrenando [03,04] y perjudica entrenando [03,04,05,06], queda probado que lo que
decide no es el mes a predecir sino si el modelo VIO el cambio de regimen."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c144_cadena"; sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
d=c.cargar("competencia_01.parquet")
PRES=["mprestamos_personales","cprestamos_personales"]
SETS={"base":c.columnas_predictoras(d),
      "sin_prestamos":[x for x in c.columnas_predictoras(d) if x not in PRES]}
v=d[d[c.fe.MES]==202107]; es=(v["clase_ternaria"].to_numpy()=="BAJA+1"); n=es.sum()
print(f"validacion FIJA: 202107 BAJA+1 = {n} personas, las que se van en agosto\n")
print(f"  {'entrenamiento':26s} {'junio?':7s} {'base':>7} {'sin pr.':>8} {'dif':>7}  semillas     p")
for TR in ([202103,202104],[202103,202104,202105],[202103,202104,202105,202106]):
    out={}
    for nom,pred in SETS.items():
        X,y,w=c.preparar(d,TR,"pesos",0.25); X=X[pred]
        out[nom]=np.array([100*es[np.argsort(-c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{202107 if 202106 in TR else 'x'}_{'_'.join(map(str,TR))}_{s}_{c.clave(P,TR,'pesos',0.25,nom+str(len(pred)),250,s)}")
            .predict(v[pred]))[:10000]].sum()/n for s in sem])
    a,b=out["sin_prestamos"],out["base"]
    print(f"  {str(TR):26s} {'SI' if 202106 in TR else 'no':7s} {b.mean():6.2f}% {a.mean():7.2f}% "
          f"{a.mean()-b.mean():+6.2f}  {int((a>b).sum()):2d}/10  p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
