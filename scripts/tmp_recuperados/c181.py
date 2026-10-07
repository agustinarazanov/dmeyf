"""Los hiperparametros de referencia de la CATEDRA (track R, z494) contra los nuestros.
Los nuestros vienen de z701, que es el notebook de DRIFT, no el del modelo final.
Presupuesto de aprendizaje: nuestro 0,0077x250 = 1,9 | catedra 0,02x1200 = 24."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, lightgbm as lgb, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
COMUN=dict(objective="binary",boosting_type="gbdt",first_metric_only=True,
           boost_from_average=True,feature_pre_filter=False,max_bin=31,
           bagging_freq=1,verbose=-1)
CFG={
 "nuestro (z701)":   dict(num_leaves=45,  learning_rate=0.0077, min_data_in_leaf=174,
                          feature_fraction=0.277, bagging_fraction=0.918, nbr=250),
 "catedra (z494)":   dict(num_leaves=750, learning_rate=0.02,   min_data_in_leaf=5000,
                          feature_fraction=0.5,   bagging_fraction=0.918, nbr=1200),
 "catedra lr/4":     dict(num_leaves=750, learning_rate=0.005,  min_data_in_leaf=5000,
                          feature_fraction=0.5,   bagging_fraction=0.918, nbr=1200),
 "intermedio":       dict(num_leaves=200, learning_rate=0.02,   min_data_in_leaf=1000,
                          feature_fraction=0.5,   bagging_fraction=0.918, nbr=600),
}
FOLDS=[([202103],202105,"BAJA+2"),([202103,202104],202106,"BAJA+2"),
       ([202103,202104,202105],202107,"BAJA+1")]
K=10_500; sem=c.SEMILLAS[:5]
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
res={}
for nom,cfg in CFG.items():
    nbr=cfg.pop("nbr"); P={**COMUN,**cfg}
    linea=f"  {nom:18s}"
    for tr,va,tg in FOLDS:
        X,y,w=c.preparar(d,tr,"pesos",0.25)
        v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()==tg); n=es.sum()
        caps=[]
        for s in sem:
            m=lgb.train({**P,"seed":s}, lgb.Dataset(X[pred],label=y,weight=w), num_boost_round=nbr)
            caps.append(100*es[np.argsort(-m.predict(v[pred]))[:K]].sum()/n)
        res[(nom,va)]=np.array(caps); linea+=f" | {va} {np.mean(caps):5.2f}%"
    print(linea+f"   [{time.time()-t0:.0f}s]",flush=True)
print(f"\ncontra el nuestro, pareado por semilla (captura a {K:,}):")
for nom in CFG:
    if nom.startswith("nuestro"): continue
    l=f"  {nom:18s}"
    for tr,va,tg in FOLDS:
        a,b=res[(nom,va)],res[("nuestro (z701)",va)]
        p=wilcoxon(a,b).pvalue if np.any(a!=b) else 1.0
        l+=f" | {va}: {a.mean()-b.mean():+5.2f} {int((a>b).sum())}/5 p={p:.3f}"
    print(l)
print(f"\nlisto en {time.time()-t0:.0f}s")
