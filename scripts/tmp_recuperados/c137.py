import sys, time
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday/competencia")
sys.path.insert(0,"/Users/agustina/Documents/segundo-cuatrimestre/dmeyf2026/monday")
import numpy as np, competencia as c
from scipy.stats import wilcoxon
t0=time.time()
P={"objective":"binary","boosting_type":"gbdt","first_metric_only":True,"boost_from_average":True,
 "feature_pre_filter":False,"max_bin":31,"num_leaves":45,"learning_rate":0.0077,
 "min_data_in_leaf":174,"feature_fraction":0.277,"bagging_fraction":0.918,"bagging_freq":1}
NBR=250; CAR=c.EXPERIMENTOS/"c137_combo"; CAR.mkdir(parents=True,exist_ok=True)
sem=c.SEMILLAS+[112909,314159,562991,733517,951413]
FOLDS=[([202103],202105),([202103,202104],202106)]
res={}
for nom,ds in [("base","competencia_01.parquet"),("todo","competencia_01_todo.parquet")]:
    d=c.cargar(ds); pred=c.columnas_predictoras(d)
    for tr,va in FOLDS:
        X,y,w=c.preparar(d,tr,"pesos",0.25); X=X[pred]
        v=d[d[c.fe.MES]==va]; es=(v["clase_ternaria"].to_numpy()=="BAJA+2")
        g={k:[] for k in (9000,10000,11000)}
        for s in sem:
            m=c.entrenar_o_cargar(P,X,y,w,NBR,s,CAR,f"{nom}_{va}_{s}_{c.clave(P,tr,'pesos',0.25,f'{nom}{len(pred)}',NBR,s)}")
            acum=c.ganancia_acumulada(m.predict(v[pred]),es)
            for k in g: g[k].append(acum[k-1]/1e6)
        res[(nom,va)]=g
        print(f"  {nom:5s} {va} ({len(pred)} cols)  "+"  ".join(f"@{k//1000}k {np.mean(gg):6.1f}M" for k,gg in g.items())+f"  [{time.time()-t0:.0f}s]")
print("\n=== los tres cambios JUNTOS contra base, pareado por semilla ===")
for tr,va in FOLDS:
    for k in (9000,10000,11000):
        a=np.array(res[("todo",va)][k]); b=np.array(res[("base",va)][k])
        print(f"  {va} @{k//1000:2d}k  {a.mean()-b.mean():+7.1f}M  {int((a>b).sum())}/10  p={wilcoxon(a,b).pvalue:.3f}")
print(f"listo en {time.time()-t0:.0f}s")
