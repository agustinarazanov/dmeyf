import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
t0=time.time()
d=c.cargar("competencia_01_hist.parquet"); pred=c.columnas_predictoras(d)
print(f"solo FE historico: {len(pred)} predictoras (prestamos personales SI estan)")
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
NBR,CORTE=250,10_000; CAR=c.EXPERIMENTOS/"c135_hist"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
X,y,w=c.preparar(d,[202103,202104,202105,202106],"pesos",0.25); X=X[pred]
fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy()
sc={}
for s in sem:
    cl=c.clave(P,[202103,202104,202105,202106],"pesos",0.25,f"hist{len(pred)}",NBR,s)
    sc[s]=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"hist_{s}_{cl}").predict(fut[pred])
    print(f"  {len(sc)}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
sel=c.top_k(c.ensamble_por_rank(sc),ids,CORTE)
c.verificar_seleccion(sel,c.cargar("competencia_01.parquet"))
n=c.escribir_envios(sel,CAR/"envios"/"c135_hist.csv",set(ids.tolist()))
v=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
ref=set(c.top_k(v.ensamble.to_numpy(),v.numero_de_cliente.to_numpy().astype("int64"),CORTE))
print(f"\nc135_hist.csv: {n:,} envios, solapamiento con la entrega actual {len(set(sel)&ref)/CORTE:.1%}")
print(f"listo en {time.time()-t0:.0f}s")
