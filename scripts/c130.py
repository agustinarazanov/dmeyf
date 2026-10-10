import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
t0=time.time()
MESES=[202103,202104,202105,202106]; NBR,CORTE=250,10_000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c130_entregas"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
PRES=["mprestamos_personales","cprestamos_personales"]
dd=c.cargar("competencia_01_defl.parquet"); todas=c.columnas_predictoras(dd)
defl=[x for x in todas if x.endswith("__defl")]; montos=[x[:-6] for x in defl]
crudas=[x for x in todas if not x.endswith("__defl")]
VAR={
 "c130_sinprest":   (crudas, [x for x in crudas if x not in PRES]),
 "c130_deflact":    (todas,  [x for x in crudas if x not in montos]+defl),
 "c130_ambos":      (todas,  [x for x in crudas if x not in montos+PRES]+defl),
}
fut=dd[dd[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); val=set(ids.tolist())
base_df=c.cargar("competencia_01.parquet")
for nom,(_,pred) in VAR.items():
    X,y,w=c.preparar(dd,MESES,"pesos",0.25); X=X[pred]
    sc={}
    for s in sem:
        cl=c.clave(P,MESES,"pesos",0.25,f"{nom}{len(pred)}",NBR,s)
        sc[s]=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"{nom}_{s}_{cl}").predict(fut[pred])
    ens=c.ensamble_por_rank(sc); sel=c.top_k(ens,ids,CORTE)
    c.verificar_seleccion(sel,base_df)
    n=c.escribir_envios(sel,CAR/"envios"/f"{nom}.csv",val)
    print(f"  {nom}.csv: {len(pred)} cols, {n:,} envios [{time.time()-t0:.0f}s]",flush=True)
v=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
ref=set(c.top_k(v.ensamble.to_numpy(), v.numero_de_cliente.to_numpy().astype("int64"), CORTE))
for nom in VAR:
    s=set(pd.read_csv(CAR/"envios"/f"{nom}.csv",header=None)[0])
    print(f"  solapamiento {nom} vs la entrega actual: {len(s&ref)/CORTE:.1%}")
print(f"listo en {time.time()-t0:.0f}s")
