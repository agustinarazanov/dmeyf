"""fe_slope3 a dos cortes, 20 archivos de una semilla cada uno, mismas 20 semillas que c107.
Referencias ya medidas sobre la misma particion:
   corte  9.000 -> c152_corte9000 (base)  94,2232   y  c153_fe228  91,1061
   corte 11.000 -> c107_pesos25   (base)  91,3550
REGLA DE VETO FIJADA DE ANTEMANO: si fe3@9.000 <= 91 se descarta el FE y va base."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
MESES=[202103,202104,202105,202106]
CAR=c.EXPERIMENTOS/"c190_fe3"; CAR.mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); SEM=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
d=c.cargar("competencia_01_fe3.parquet"); pred=c.columnas_predictoras(d)
X,y,w=c.preparar(d,MESES,"pesos",0.25); X=X[pred]
fut=d[d[c.fe.MES]==202108]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
ref=c.cargar("competencia_01.parquet")
print(f"{len(pred)} predictoras, {len(X):,} filas de train")
sc={}
for i,s in enumerate(SEM,1):
    sc[s]=c.entrenar_o_cargar(P,X,y,w,250,s,CAR,f"fe3_{s}").predict(fut[pred])
    print(f"  {i:2d}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
pd.DataFrame({"numero_de_cliente":ids,"ensamble":c.ensamble_por_rank(sc)}).to_parquet(CAR/"scores_202108.parquet")
print()
for K in (9000,11000):
    E=CAR/f"envios_{K}"; E.mkdir(exist_ok=True)
    for j,s in enumerate(SEM):
        sel=c.top_k(sc[s],ids,K)
        if j==0: c.verificar_seleccion(sel,ref)
        c.escribir_envios(sel,E/f"c190_fe3_{K}_s{s}.csv",validos)
    print(f"  corte {K:6,}: 20 archivos listos")
print(f"listo en {time.time()-t0:.0f}s")
