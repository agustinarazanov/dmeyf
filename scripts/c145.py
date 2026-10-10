"""Las dos entregas de la pregunta de junio: identicas al base salvo que entrenan
con [03,04,05] en vez de [03,04,05,06]. Mismo dataset, mismas columnas, mismo
pesos 0,25, mismas 20 semillas, mismo ensamble por rank. Dos cortes para replica."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, competencia as c
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
MESES=[202103,202104,202105]            # <- SIN junio, esa es la unica diferencia
CAR=c.EXPERIMENTOS/"c145_sinjunio"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
X,y,w=c.preparar(d,MESES,"pesos",0.25); X=X[pred]
print(f"train {MESES} -> {len(X):,} filas, {int(y.sum()):,} positivos, {len(pred)} columnas")
fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
sc={}
for s in sem:
    sc[s]=c.entrenar_o_cargar(P,X,y,w,250,s,CAR,
        f"sinjunio_{s}_{c.clave(P,MESES,'pesos',0.25,'base150',250,s)}").predict(fut[pred])
    print(f"  {len(sc)}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
ens=c.ensamble_por_rank(sc)
pd.DataFrame({"numero_de_cliente":ids,"ensamble":ens}).to_parquet(CAR/"scores_202108.parquet")
ref=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
print()
for K,nombre,comparar in [(10000,"c145_sinjunio_10000","c118_corte_10000 = 92.62"),
                          (9000,"c145_sinjunio_9000","c107corte9000 = 97.24")]:
    sel=c.top_k(ens,ids,K)
    c.verificar_seleccion(sel,d)
    n=c.escribir_envios(sel,CAR/"envios"/f"{nombre}.csv",validos)
    base=set(c.top_k(ref.ensamble.to_numpy(), ref.numero_de_cliente.to_numpy().astype("int64"), K))
    print(f"  {nombre}.csv  {n:,} envios | solapa {len(set(sel)&base)/K:5.1%} con el base del mismo corte | contra {comparar}")
print(f"\nlisto en {time.time()-t0:.0f}s")
