import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import competencia as c
FOLDS=[([202103],202105),([202103,202104],202106)]
TARGET,PESO,NBR,CORTE="pesos",0.25,250,11_000
PARAMS={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,
 "boost_from_average":True,"feature_pre_filter":False,"max_bin":31,"num_leaves":45,
 "learning_rate":0.0077,"min_data_in_leaf":174,"feature_fraction":0.277,
 "bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c116_v2"; CAR.mkdir(parents=True,exist_ok=True)
t0=time.time()
np.random.seed(c.SEMILLAS[0])
sem=c.SEMILLAS+np.random.choice(1_000_000,size=5,replace=False).tolist()
d_base=c.cargar("competencia_01.parquet"); d_v101=c.cargar("competencia_01_v101.parquet")
d_v2=c.cargar("competencia_01_v2.parquet")
SETS={"base":(d_base,c.columnas_predictoras(d_base)),
      "v101_viejo":(d_v101,c.columnas_predictoras(d_v101)),
      "v2_corregido":(d_v2,c.columnas_predictoras(d_v2))}
res={}
for etq,(data,pred) in SETS.items():
    obs=[]
    for meses,mv in FOLDS:
        X,y,w=c.preparar(data,meses,TARGET,PESO); X=X[pred]
        val=data[data[c.fe.MES]==mv]; es=(val[c.fe.CLASE].to_numpy()=="BAJA+2")
        for s in sem:
            cl=c.clave(PARAMS,meses,TARGET,PESO,f"{etq}{len(pred)}",NBR,s)
            m=c.entrenar_o_cargar(PARAMS,X,y,w,NBR,s,CAR,f"{etq}_{mv}_{s}_{cl}")
            obs.append(c.ganancia_acumulada(m.predict(val[pred]),es)[CORTE-1])
    res[etq]=np.array(obs)
    print(f"  {etq:13} {len(pred):>3} cols  media={np.mean(obs)/1e6:>6,.1f}M  [{time.time()-t0:.0f}s]",flush=True)
print("\n=== pareado por (fold, semilla) ===")
for ref in ("base","v101_viejo"):
    print(f"  contra {ref}:")
    for etq,v in res.items():
        d=v-res[ref]
        if np.allclose(d,0): continue
        print(f"    {etq:13} {d.mean()/1e6:+7.1f} M  gana en {(d>0).sum()}/{len(d)}  p={wilcoxon(d).pvalue:.4f}")
pd.DataFrame(res).to_parquet(CAR/"por_semilla.parquet",index=False)
print(f"\nlisto en {time.time()-t0:.0f}s")
