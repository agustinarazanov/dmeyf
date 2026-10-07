"""rank_cero_fijo (metodo de drift de la catedra, z1401) contra el base.
Chequeo local para descartar si es obviamente malo, y armado de los 20 archivos."""
import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time(); K=9000
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c167_rcf"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
sem10=c.SEMILLAS+[112909,314159,562991,733517,951413]
res={}
for nom,ds in [("base","competencia_01.parquet"),("rcf","competencia_01_rcf.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    for tr,va in [([202103],202105),([202103,202104],202106)]:
        X,y,w=c.preparar(d,tr,"pesos",0.25)
        v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")   # de SU propio v
        res[(nom,va)]=np.array([c.ganancia_acumulada(c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
            f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,nom+str(len(pred)),250,s)}")
            .predict(v[pred]),es)[K-1]/1e6 for s in sem10])
    print(f"  {nom:5s} ({len(pred)} col) 202105 {res[(nom,202105)].mean():6.1f}M | "
          f"202106 {res[(nom,202106)].mean():6.1f}M   [{time.time()-t0:.0f}s]")
print(f"\n  contra base, corte {K:,}:")
for va in (202105,202106):
    a,b=res[("rcf",va)],res[("base",va)]
    print(f"    {va}: {a.mean()-b.mean():+6.1f}M  {int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
# los 20 archivos
np.random.seed(c.SEMILLAS[0]); SEM=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
d=c.cargar("competencia_01_rcf.parquet"); pred=c.columnas_predictoras(d)
X,y,w=c.preparar(d,[202103,202104,202105,202106],"pesos",0.25)
fut=d[d[c.fe.MES]==202108]; ids=fut[c.fe.ID].to_numpy(); validos=set(ids.tolist())
ref=c.cargar("competencia_01.parquet")
print()
for i,s in enumerate(SEM,1):
    sc=c.entrenar_o_cargar(P,X[pred],y,w,250,s,CAR,
        f"final_{s}_{c.clave(P,[202103,202104,202105,202106],'pesos',0.25,'rcf'+str(len(pred)),250,s)}").predict(fut[pred])
    sel=c.top_k(sc,ids,K)
    if i==1: c.verificar_seleccion(sel, ref)
    c.escribir_envios(sel, CAR/"envios"/f"c167_rcf_s{s}.csv", validos)
    print(f"    {i:2d}/20 [{time.time()-t0:.0f}s]",end="\r",flush=True)
print(f"\n  20 archivos de {K:,} envios listos   [{time.time()-t0:.0f}s]")
