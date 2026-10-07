"""La entrega definitiva con el orden de filas canonico, en carpeta nueva para
forzar reentrenamiento (la llave de cache no incluye el orden)."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, pandas as pd, competencia as c
t0=time.time(); K=10_500
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c174_final_ordenado"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); SEM=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
X,y,w=c.preparar(d,[202103,202104,202105,202106],"pesos",0.25)
fut=d[d[c.fe.MES]==202108]; ids=fut[c.fe.ID].to_numpy()
print(f"{len(X):,} filas de train, {len(pred)} columnas, orden canonico")
sc={}
for i,s in enumerate(SEM,1):
    sc[s]=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,f"ord_{s}").predict(fut[pred])
    print(f"  {i:2d}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
ens=c.ensamble_por_rank(sc)
pd.DataFrame({"numero_de_cliente":ids,"ensamble":ens}).to_parquet(CAR/"scores_202108.parquet")
sel=c.top_k(ens,ids,K); c.verificar_seleccion(sel,d)
n=c.escribir_envios(sel,CAR/"envios"/"c174_final.csv",set(ids.tolist()))
viejo=set(pd.read_csv(c.EXPERIMENTOS/"c173_entrega_10500"/"envios"/"c173_entrega_10500.csv",header=None)[0])
print(f"\n{n:,} envios | solapamiento con la version sin ordenar: {len(set(sel)&viejo)/K:.2%}")
print(f"listo en {time.time()-t0:.0f}s")
