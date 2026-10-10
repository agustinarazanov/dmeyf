import sys, time, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd, optuna, competencia as c, registro as r
optuna.logging.set_verbosity(optuna.logging.WARNING)
t0=time.time()
MESES=[202103,202104,202105,202106]; NBR=250
BASE={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
CAR=c.EXPERIMENTOS/"c126_sondas"; (CAR/"envios").mkdir(parents=True,exist_ok=True)
np.random.seed(c.SEMILLAS[0]); sem=c.SEMILLAS+np.random.choice(1_000_000,size=15,replace=False).tolist()
d=c.cargar("competencia_01.parquet"); pred=c.columnas_predictoras(d)
fut=d[d[c.fe.MES]==c.MES_COMPETENCIA]; ids=fut[c.fe.ID].to_numpy(); val=set(ids.tolist())

# --- A) el candidato de Optuna: modelo genuinamente DISTINTO (186 hojas vs 45) ---
est=optuna.load_study(study_name="c122_base",storage=r.dsn_url())
t22=max([t for t in est.trials if t.value is not None],key=lambda t:t.value)
P={**{k:v for k,v in BASE.items() if k not in t22.params},
   **{k:t22.params[k] for k in ("num_leaves","learning_rate","min_data_in_leaf","feature_fraction",
      "bagging_fraction","lambda_l1","lambda_l2","min_gain_to_split","max_bin") if k in t22.params}}
esq, peso, nbr = t22.params.get("esquema","baja2"), t22.params.get("peso_baja1",0.0), t22.user_attrs.get("best_iter",250)
X,y,w=c.preparar(d,MESES,esq,peso); X=X[pred]
sc={}
for s in sem[:10]:
    cl=c.clave(P,MESES,esq,peso,"t22",nbr,s)
    sc[s]=c.entrenar_o_cargar(P,X,y,w,nbr,s,CAR,f"t22_{s}_{cl}").predict(fut[pred])
eA=c.ensamble_por_rank(sc); print(f"A) t22 ({t22.params['num_leaves']} hojas, {nbr} rondas) [{time.time()-t0:.0f}s]")

# --- B) sin marzo: train [202104,202105,202106] ---
MB=[202104,202105,202106]
Xb,yb,wb=c.preparar(d,MB,"pesos",0.25); Xb=Xb[pred]
sc2={}
for s in sem:
    cl=c.clave(BASE,MB,"pesos",0.25,"sinmarzo",NBR,s)
    sc2[s]=c.entrenar_o_cargar(BASE,Xb,yb,wb,NBR,s,CAR,f"sinmarzo_{s}_{cl}").predict(fut[pred])
eB=c.ensamble_por_rank(sc2); print(f"B) sin marzo ({len(Xb):,} filas vs 654.066) [{time.time()-t0:.0f}s]")

# --- C) la cola de la curva, mismo ranking de base ---
eC=pd.read_parquet(c.EXPERIMENTOS/"c107_pesos25"/"scores_202108.parquet")
idc=eC.numero_de_cliente.to_numpy().astype("int64"); eC=eC.ensamble.to_numpy()

for nom, ens, iii, k in (("c126_t22_9000",eA,ids,9000), ("c126_t22_11000",eA,ids,11000),
                         ("c126_sinmarzo_11000",eB,ids,11000),
                         ("c126_corte_13000",eC,idc,13000), ("c126_corte_15000",eC,idc,15000)):
    sel=c.top_k(ens,iii,k); c.verificar_seleccion(sel,d)
    n=c.escribir_envios(sel,CAR/"envios"/f"{nom}.csv",val)
    print(f"  {nom}.csv: {n:,}")
v=pd.Series(eC,index=idc)
for nom,e in (("t22",pd.Series(eA,index=ids.astype('int64'))),("sin marzo",pd.Series(eB,index=ids.astype('int64')))):
    com=v.index.intersection(e.index)
    print(f"Spearman {nom} vs base: {v[com].corr(e[com],method='spearman'):.4f}")
print(f"listo en {time.time()-t0:.0f}s")
