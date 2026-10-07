"""Historia MINIMA (25 col sobre 5 campos estables) contra el base.
Se mide con la cadena de horizonte 1, que es el unico instrumento donde las
columnas de historia estan definidas en el entrenamiento. El eslabon que importa
valida 202107 = la cohorte que se va en AGOSTO.
Se prueba con marzo y sin marzo, porque en marzo las nuevas son todas nulas."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c147_histmin"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
DS={"base":"competencia_01.parquet","hist_min":"competencia_01_histmin.parquet"}
datos={k:c.cargar(v) for k,v in DS.items()}
for k,d in datos.items(): print(f"  {k:9s} {len(c.columnas_predictoras(d))} predictoras")
print()
for TR in ([202103,202104,202105,202106],[202104,202105,202106]):
    out={}; n=None
    for nom,d in datos.items():
        pred=c.columnas_predictoras(d)
        X,y,w=c.preparar(d,TR,"pesos",0.25); X=X[pred]
        val=d[d[c.fe.MES]==202107]
        # es se calcula DE SU PROPIO val: los parquets no comparten orden de filas
        es=(val["clase_ternaria"].to_numpy()=="BAJA+1"); n=es.sum()
        out[nom]=np.array([100*es[np.argsort(-c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
            f"{nom}_{'_'.join(str(m%100) for m in TR)}_{s}_{c.clave(P,TR,'pesos',0.25,nom+str(len(pred)),250,s)}")
            .predict(val[pred]))[:10000]].sum()/n for s in sem])
    a,b=out["hist_min"],out["base"]
    print(f"  train {TR}  (valida 202107, {n} que se van en agosto)")
    print(f"    base      {b.mean():6.2f}%")
    print(f"    hist_min  {a.mean():6.2f}%   dif {a.mean()-b.mean():+5.2f} pts  "
          f"{int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
