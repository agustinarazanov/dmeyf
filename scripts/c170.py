"""Sacar los dos relojes de calendario: Master_fultimo_cierre y Visa_fultimo_cierre.
La validacion adversaria dice que separan 202108 del train con AUC 0,9998 y se llevan
el 97% del gain: son un PROXY DE foto_mes, la variable que sacamos a proposito.
Se mide en los TRES folds limpios."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); K=9000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
RELOJ=["Master_fultimo_cierre","Visa_fultimo_cierre"]
CAR=c.EXPERIMENTOS/"c170_reloj"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
d=c.cargar("competencia_01.parquet")
SETS={"base":c.columnas_predictoras(d),
      "sin_reloj":[x for x in c.columnas_predictoras(d) if x not in RELOJ]}
FOLDS=[([202103],202105,"BAJA+2"),([202103,202104],202106,"BAJA+2"),
       ([202103,202104,202105],202107,"BAJA+1")]
res={}
for tr,va,tg in FOLDS:
    v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()==tg); n=es.sum()
    for nom,pred in SETS.items():
        X,y,w=c.preparar(d,tr,"pesos",0.25)
        vals=[]
        for s in sem:
            m=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
                f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,nom+str(len(pred)),250,s)}")
            o=np.argsort(-m.predict(v[pred]))
            vals.append(100*es[o[:K]].sum()/n)       # captura: comparable entre folds
        res[(nom,va)]=np.array(vals)
    a,b=res[("sin_reloj",va)],res[("base",va)]
    print(f"  fold {va} ({tg}, {len(tr)} mes/es de train): base {b.mean():5.2f}% | "
          f"sin reloj {a.mean():5.2f}%  ->  {a.mean()-b.mean():+5.2f} pts  "
          f"{int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}   [{time.time()-t0:.0f}s]")
print(f"\nlisto en {time.time()-t0:.0f}s")
