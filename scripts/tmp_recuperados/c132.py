import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
import numpy as np, pandas as pd, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); KS=(9000,11000); NBR=250; N=10
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c132_hist"; CAR.mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=N-5,replace=False).tolist()
d=c.cargar("competencia_01_hist.parquet"); todas=c.columnas_predictoras(d)
SUF=("__vs_propio","__acel","__volrel","__vs_max")
sets={"base":[x for x in todas if not x.endswith(SUF)],
      "hist_v3":todas,
      "solo_vs_propio":[x for x in todas if not x.endswith(("__acel","__volrel","__vs_max"))]}
for k,v in sets.items(): print(f"  {k:16} {len(v)} cols")
res={}
for etq,pred in sets.items():
    for meses,mv in (([202103],202105),([202103,202104],202106)):
        X,y,w=c.preparar(d,meses,"pesos",0.25); X=X[pred]
        val=d[d[c.fe.MES]==mv]; es=(val[c.fe.CLASE].to_numpy()=="BAJA+2")
        a=[]
        for s in sem:
            cl=c.clave(P,meses,"pesos",0.25,f"{etq}{len(pred)}",NBR,s)
            m=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"{etq}_{mv}_{s}_{cl}")
            a.append(c.ganancia_acumulada(m.predict(val[pred]),es))
        res[(etq,mv)]=a
        print(f"  {etq:16} {mv} " + " ".join(f"{np.mean([x[k-1] for x in a])/1e6:>7,.1f}M" for k in KS)
              + f"  [{time.time()-t0:.0f}s]",flush=True)
print("\n=== contra base, FOLD POR FOLD ===")
for etq in sets:
    if etq=="base": continue
    for mv in (202105,202106):
        out=[]
        for k in KS:
            dd=np.array([x[k-1] for x in res[(etq,mv)]])-np.array([x[k-1] for x in res[("base",mv)]])
            out.append(f"@{k//1000}k {dd.mean()/1e6:+6.1f}M {(dd>0).sum():>2}/{len(dd)} p={wilcoxon(dd).pvalue:.3f}")
        print(f"  {etq:16} {mv}: " + " | ".join(out))
print(f"\nlisto en {time.time()-t0:.0f}s")
