import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
NBR=250; CAR=c.EXPERIMENTOS/"c134_mora"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
# arreglo: donde Finiciomora=0 el delinquency es inyectado
arr=d.copy()
for t in ["Visa","Master"]:
    m=arr[f"{t}_Finiciomora"].eq(0)
    arr.loc[m,f"{t}_delinquency"]=0
    print(f"{t}: {m.sum():,} filas con delinquency forzado a 0")
FOLDS=[([202103],202105),([202103,202104],202106)]
res={}
for nom,dd in [("crudo",d),("mora_arreglada",arr)]:
    for tr,va in FOLDS:
        X,y,w=c.preparar(dd,tr,"pesos",0.25); X=X[pred]
        v=dd[dd[c.fe.MES]==va]; real=c.aporte(v["clase_ternaria"].to_numpy()=="BAJA+2")
        g={k:[] for k in (9000,11000)}
        for s in sem:
            mo=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,nom,NBR,s)}")
            sc=mo.predict(v[pred]); o=np.argsort(-sc)
            for k in g: g[k].append(real[o[:k]].sum()/1e6)
        res[(nom,va)]=g
        print(f"  {nom:16s} {va}  @9k {np.mean(g[9000]):7.1f}M  @11k {np.mean(g[11000]):7.1f}M  [{time.time()-t0:.0f}s]")
print("\n=== arreglo de mora contra crudo, pareado por semilla ===")
for tr,va in FOLDS:
    linea=f"  {va}:"
    for k in (9000,11000):
        a=np.array(res[("mora_arreglada",va)][k]); b=np.array(res[("crudo",va)][k])
        p=wilcoxon(a,b).pvalue if np.any(a!=b) else 1.0
        linea+=f"  @{k//1000}k {a.mean()-b.mean():+6.1f}M {int((a>b).sum())}/10 p={p:.3f} |"
    print(linea + ("   <- mes ROTO, como agosto" if va==202105 else "   <- mes limpio, el arreglo no deberia hacer nada"))
print(f"listo en {time.time()-t0:.0f}s")
