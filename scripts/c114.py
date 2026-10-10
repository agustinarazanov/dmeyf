import time, glob
import numpy as np, lightgbm as lgb
import competencia as c
t0=time.time()
MESES=[202103,202104,202105,202106]; TARGET,PESO,NBR,CORTE="pesos",0.25,250,11_000
PARAMS={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,
 "boost_from_average":True,"feature_pre_filter":False,"max_bin":31,"num_leaves":45,
 "learning_rate":0.0077,"min_data_in_leaf":174,"feature_fraction":0.277,
 "bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c114_v101_20"; CAR.mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0])
semillas=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
data=c.cargar("competencia_01_v101.parquet"); pred=c.columnas_predictoras(data)
X,y,w=c.preparar(data,MESES,TARGET,PESO); X=X[pred]
fut=data[data[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
print(f"{len(pred)} cols, {len(X):,} filas, {len(semillas)} semillas")
scores={}
for s in semillas:
    cl=c.clave(PARAMS,MESES,TARGET,PESO,f"v101{len(pred)}",NBR,s)
    m=c.entrenar_o_cargar(PARAMS,X,y,w,NBR,s,CAR,f"fin_{s}_{cl}")
    scores[s]=m.predict(fut[pred])
print(f"entrenados [{time.time()-t0:.0f}s]")
ens=c.ensamble_por_rank(scores)
ruta=CAR/"envios"/"c114_v101_ens20.csv"
n=c.escribir_envios(c.top_k(ens,ids,CORTE),ruta,validos)
print(f"{ruta.name}: {n:,} envios")
import pandas as pd
pd.DataFrame({"numero_de_cliente":ids,"ensamble":ens,**{f"s{k}":v for k,v in scores.items()}}
             ).to_parquet(CAR/"scores_202108.parquet",index=False)
print(f"listo en {time.time()-t0:.0f}s")
