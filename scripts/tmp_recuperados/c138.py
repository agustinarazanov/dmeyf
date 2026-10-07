"""Fold C aplicado a los tres cambios: valida 202107 (BAJA+1), el mes etiquetable
mas cercano a agosto. Mide CAPTURA (fraccion de los que se van que cae en el top K),
que es adimensional y no depende de que el target sea BAJA+1 en vez de BAJA+2."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
NBR=250; CAR=c.EXPERIMENTOS/"c138_foldC"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
TR=[202103,202104,202105]
res={}
for nom,ds in [("base","competencia_01.parquet"),("todo","competencia_01_todo.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    X,y,w=c.preparar(d,TR,"pesos",0.25); X=X[pred]
    v=d[d[c.fe.MES]==202107]; es=(v["clase_ternaria"].to_numpy()=="BAJA+1"); n=es.sum()
    cap={k:[] for k in (9000,10000,11000)}
    for s in sem:
        m=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"{nom}_{s}_{c.clave(P,TR,'pesos',0.25,f'{nom}{len(pred)}',NBR,s)}")
        o=np.argsort(-m.predict(v[pred]))
        for k in cap: cap[k].append(100*es[o[:k]].sum()/n)
    res[nom]=cap
    print(f"  {nom:5s} ({len(pred)} cols)  "+"  ".join(f"@{k//1000}k {np.mean(vv):5.2f}%" for k,vv in cap.items())+f"   [{time.time()-t0:.0f}s]  n={n} BAJA+1")
print("\n=== fold C (valida 202107, el mes mas cercano a agosto) ===")
for k in (9000,10000,11000):
    a=np.array(res["todo"][k]); b=np.array(res["base"][k])
    print(f"  @{k//1000:2d}k  captura {a.mean()-b.mean():+5.2f} puntos  {int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
print(f"listo en {time.time()-t0:.0f}s")
