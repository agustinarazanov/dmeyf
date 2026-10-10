import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
MESES=[202103,202104,202105,202106]; TARGET,PESO,NBR="pesos",0.25,250
PARAMS={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,
 "boost_from_average":True,"feature_pre_filter":False,"max_bin":31,"num_leaves":45,
 "learning_rate":0.0077,"min_data_in_leaf":174,"feature_fraction":0.277,
 "bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c117_v2_final"; CAR.mkdir(parents=True,exist_ok=True)
t0=time.time(); np.random.seed(c.SEMILLAS[0])
sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
data=c.cargar("competencia_01_v2.parquet"); pred=c.columnas_predictoras(data)
X,y,w=c.preparar(data,MESES,TARGET,PESO); X=X[pred]
fut=data[data[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
print(f"{len(pred)} cols, {len(X):,} filas, {len(sem)} semillas")
scores={}
for s in sem:
    cl=c.clave(PARAMS,MESES,TARGET,PESO,f"v2{len(pred)}",NBR,s)
    scores[s]=c.entrenar_o_cargar(PARAMS,X,y,w,NBR,s,CAR,f"fin_{s}_{cl}").predict(fut[pred])
ens=c.ensamble_por_rank(scores)
pd.DataFrame({"numero_de_cliente":ids,"ensamble":ens,**{f"s{k}":v for k,v in scores.items()}}
             ).to_parquet(CAR/"scores_202108.parquet",index=False)
# el corte que midio AGOSTO (pico 9.000-9.500), no el minimax de junio
for k in (9000, 9500):
    n=c.escribir_envios(c.top_k(ens,ids,k), CAR/"envios"/f"c117_v2_ens_{k}.csv", validos)
    print(f"  c117_v2_ens_{k}.csv: {n:,} envios")
print(f"listo en {time.time()-t0:.0f}s")
