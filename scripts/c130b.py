import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
t0=time.time()
MESES=[202103,202104,202105,202106]; NBR,CORTE=250,10_000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c130_entregas"
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
PRES=["mprestamos_personales","cprestamos_personales"]
dd=c.cargar("competencia_01_defl.parquet"); todas=c.columnas_predictoras(dd)
defl=[x for x in todas if x.endswith("__defl")]; montos=[x[:-6] for x in defl]
crudas=[x for x in todas if not x.endswith("__defl")]
# CORREGIDO: saca tambien la version deflactada del prestamo
pred=[x for x in crudas if x not in montos+PRES]+[x for x in defl if x[:-6] not in PRES]
assert not [x for x in pred if x in PRES or x[:-6] in PRES], "quedo un prestamo"
print(f"{len(pred)} columnas, prestamos personales restantes: 0")
X,y,w=c.preparar(dd,MESES,"pesos",0.25); X=X[pred]
fut=dd[dd[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); val=set(ids.tolist())
sc={}
for s in sem:
    cl=c.clave(P,MESES,"pesos",0.25,f"ambosB{len(pred)}",NBR,s)
    sc[s]=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"c130_ambosB_{s}_{cl}").predict(fut[pred])
ens=c.ensamble_por_rank(sc); sel=c.top_k(ens,ids,CORTE)
c.verificar_seleccion(sel, c.cargar("competencia_01.parquet"))
n=c.escribir_envios(sel,CAR/"envios"/"c130_ambos.csv",val)   # pisa el archivo malo
v=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
ref=set(c.top_k(v.ensamble.to_numpy(), v.numero_de_cliente.to_numpy().astype("int64"), CORTE))
print(f"c130_ambos.csv regenerado: {n:,} envios, solapamiento con la entrega actual {len(set(sel)&ref)/CORTE:.1%}")
print(f"listo en {time.time()-t0:.0f}s")
